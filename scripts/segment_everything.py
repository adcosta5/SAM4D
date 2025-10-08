"""Segment everything in image and pointcloud using SAM4D.

This script:
- builds the SAM4D predictor from a config and checkpoint
- initializes inference state from a sequence (image paths + point cloud npz)
- for frame 0, creates a full-image mask and a full-pointcloud mask and calls `add_new_mask`
- runs `propagate_in_video` to get per-frame masks for all objects (here a single object covering the whole scene)
- saves visualizations under `outputs/segment_everything/`.

Usage example:
    python scripts/segment_everything.py --config configs/sam4d_hieraS_mink34w32.py \
        --ckpt data/samples/sam4d_hieraS_mink34w32.pth \
        --data_dir data/samples/waymo \
        --seq segment-10868756386479184868_3000_000_3020_000_with_camera_labels \
        --cam FRONT

This is a simple helper for batch segmentation without interactive prompts.
"""

import os
import argparse
import pickle
import open3d as o3d
from PIL import Image
import numpy as np
import torch

from mmengine import Config
from mmengine.registry import MODELS

from sam4d.misc import _load_checkpoint
from sam4d.visualize import sam4d_viz


# Small helper copied/adapted from the notebook to build predictor

def build_sam4d_predictor(config_file, ckpt_path=None, device="cuda", apply_postprocessing=True, cfg_options_extra=None):
    cfg_options = {"model.type": "SAM4DPredictor"}
    cfg_options_extra = {} if cfg_options_extra is None else cfg_options_extra.copy()
    if apply_postprocessing:
        cfg_options_extra.update({
            "model.head.sam.sam_mask_decoder_extra_args.dynamic_multimask_via_stability": True,
            "model.head.sam.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta": 0.05,
            "model.head.sam.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh": 0.98,
            "model.binarize_mask_from_pts_for_mem_enc": False,
            "model.fill_hole_area": 8,
        })
    cfg_options.update(cfg_options_extra)
    cfg = Config.fromfile(config_file)
    cfg.merge_from_dict(cfg_options)
    model = MODELS.build(cfg.model)
    model.set_data_pipeline(cfg.data.test.pipeline)
    if ckpt_path is not None:
        _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    model.eval()
    return model


def make_full_masks_for_frame(inference_state, frame_idx=0):
    """Create a full-image mask and full-points mask (covering all pixels/points).

    Returns a dict suitable for `add_new_mask` where keys are 'img' and/or 'pts'.
    """
    masks = {}
    if 'images' in inference_state:
        h = inference_state['video_height'][frame_idx]['img']
        w = inference_state['video_width'][frame_idx]['img']
        # full mask: ones of shape (H, W)
        masks['img'] = torch.ones((h, w), dtype=torch.float32)
    if 'points' in inference_state:
        # for pts, the model expects a 2D mask of shape (N_pts, 1)? In add_new_mask the code
        # expects a 2D tensor for pts as (H, W) style; here the convention used is a 2D mask
        # with shape (n_pts, 1) stored as (N, ) maybe. We'll follow the `_use_mask_as_output`
        # path: for pts it expects a 2D mask (H, W) where H is number of points and W==1.
        n_pts = inference_state['video_height'][frame_idx]['pts']
        masks['pts'] = torch.ones((n_pts, 1), dtype=torch.float32)
    return masks


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--data_dir', required=True)
    p.add_argument('--seq', required=True)
    p.add_argument('--cam', default='FRONT')
    p.add_argument('--outdir', default='outputs/segment_everything')
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # build predictor
    predictor = build_sam4d_predictor(args.config, args.ckpt, device=args.device)

    # load meta info for sequence
    meta_pkl = os.path.join(args.data_dir, 'meta_infos', f'{args.seq}.pkl')
    if not os.path.exists(meta_pkl):
        raise FileNotFoundError(f"meta info not found: {meta_pkl}")
    with open(meta_pkl, 'rb') as f:
        meta_info = pickle.load(f)

    img_paths, metas, lidar_paths = [], [], []
    for frame_info in meta_info['frames']:
        img_path = os.path.join(args.data_dir, frame_info['cams_info'][args.cam]['data_path'])
        img_paths.append(img_path)
        camera_intrinsics = np.eye(4)
        camera_intrinsics[:3, :3] = frame_info['cams_info'][args.cam]['camera_intrinsics'][:3, :3]
        metas.append({'pose': frame_info['lidar2world'],
                      'camera_intrinsics': camera_intrinsics,
                      'camera2lidar': frame_info['cams_info'][args.cam]['camera2lidar']})
        lidar_paths.append(os.path.join(args.data_dir, frame_info['path']['pcd']))

    print(f'Loaded {len(img_paths)} frames from {args.seq}')

    # initialize inference state
    inference_state = predictor.init_state(img_paths=img_paths, pts_paths=lidar_paths, metas=metas)

    # create full masks at frame 0 and add as a single object id (e.g., obj_id=1)
    masks = make_full_masks_for_frame(inference_state, frame_idx=0)
    frame_idx, obj_ids, video_res_masks = predictor.add_new_mask(inference_state, frame_idx=0, obj_id=1, mask=masks)
    print('Added full masks for frame 0; obj ids:', obj_ids)

    # save raw masks for frame 0
    out_masks_dir = os.path.join(args.outdir, 'masks')
    os.makedirs(out_masks_dir, exist_ok=True)
    # save numpy versions of masks per frame 0
    for k, v in video_res_masks.items():
        # v should be a tensor; move to cpu
        try:
            arr = v.cpu().numpy()
        except Exception:
            arr = np.array(v)
        np.save(os.path.join(out_masks_dir, f'mask_frame0_{k}.npy'), arr)

    # Render an overlay image for frame 0 and save as PNG
    try:
        img0 = np.array(Image.open(img_paths[0]))
        pcd0 = np.load(lidar_paths[0])['data']
        # `video_res_masks` is a dict with keys like 'img' and 'pts' containing tensors of shape [B, n_obj, H, W] or similar.
        # sam4d_viz expects: (img, pcd, out_mask_logits, out_obj_ids, ...)
        # The predictor returned `video_res_masks` from add_new_mask as `video_res_masks` already (per-object logits).
        out_mask_logits = video_res_masks
        out_obj_ids = obj_ids
        viz_img = sam4d_viz(img0, pcd0, out_mask_logits, out_obj_ids, title='frame_0_overlay')
        overlay_path = os.path.join(args.outdir, 'frame0_overlay.png')
        Image.fromarray(viz_img).save(overlay_path)
        print(f'Saved overlay image to: {overlay_path}')
    except Exception as e:
        print('Failed to render overlay image:', e)

    # Display the frame-0 pointcloud using Open3D (no propagation for now)
    try:
        pcd0 = np.load(lidar_paths[0])['data']
        pcd_o3d = o3d.geometry.PointCloud()
        # assume point cloud is [N, >=3] with xyz in the first 3 channels
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd0[:, :3])
        print("Displaying point cloud for frame 0 (close the window to continue)...")
        o3d.visualization.draw_geometries([pcd_o3d])
    except Exception as e:
        print("Open3D display failed:", e)
        print("Tip: install Open3D via: pip install open3d")

    print(f'Saved frame-0 masks to {out_masks_dir}')


if __name__ == '__main__':
    main()
