# SAM4D
> Jianyun Xu†, Song Wang†, Ziqian Ni†, Chunyong Hu, Sheng Yang*, Jianke Zhu, Qiang Li

This is the official implementation of **SAM4D: Segment Anything in Camera and LiDAR Streams** (ICCV 2025)  [[Paper](https://arxiv.org/abs/2506.xxxxx)] [[Project Page](https://sam4d-project.github.io/)].

## Abstract
We present SAM4D, a multi-modal and temporal foundation model designed for promptable segmentation across camera and LiDAR streams. Unified Multi-modal Positional Encoding (UMPE) is introduced to align camera and LiDAR features in a shared 3D space, enabling seamless cross-modal prompting and interaction. Additionally, we propose Motion-aware Cross-modal Memory Attention (MCMA), which leverages ego-motion compensation to enhance temporal consistency and long-horizon feature retrieval, ensuring robust segmentation across dynamically changing autonomous driving scenes. To avoid annotation bottlenecks, we develop a multi-modal automated data engine that synergizes VFM-driven video masklets, spatiotemporal 4D reconstruction, and cross-modal masklet fusion. This framework generates camera-LiDAR aligned pseudo-labels at a speed orders of magnitude faster than human annotation while preserving VFM-derived semantic fidelity in point cloud representations. We conduct extensive experiments on the constructed Waymo-4DSeg, which demonstrate the powerful cross-modal segmentation ability and great potential in data annotation of proposed SAM4D.

<p align="center"> <a><img src="figs/teaser.png" width="90%"></a> </p>

## Acknowledgement

We gratefully acknowledge the developers of the following open-source projects and datasets, whose foundational tools enabled our research: [SAM2](https://github.com/facebookresearch/sam2), [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2), [verl](https://github.com/volcengine/verl), [Waymo Open Dataset](https://waymo.com/open), [VDBFusion](https://github.com/PRBonn/vdbfusion), among others.
