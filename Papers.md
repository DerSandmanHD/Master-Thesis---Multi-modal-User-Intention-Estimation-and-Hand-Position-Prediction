# Literature Master List: Master Thesis

## 1. Deep Learning Architectures & Models

### Core Architectures (Top Models)
*   **[GTN] Gated Transformer Networks**
    *   *Title:* Gated Transformer Networks for Multivariate Time Series Classification (IJCAI 2021)
    *   *Links:* [arXiv:2103.14438](https://arxiv.org/abs/2103.14438) | [Official GitHub](https://github.com/ZZUFaceBookDL/GTN)
    *   *Role in Thesis:* Primary architecture blueprint for modeling step-wise and channel-wise dependencies in multimodal sensor matrices.
    *   *Local note:* See [`references/GTN.md`](references/GTN.md). The active model is an independent, task-specific implementation and does not import the upstream source tree.
*   **[InceptionTime] 1D-CNN Benchmark**
    *   *Title:* InceptionTime: Finding AlexNet for Time Series Classification (Data Mining and Knowledge Discovery 2020)
    *   *Links:* [arXiv:1909.04939](https://arxiv.org/abs/1909.04939) | [Official GitHub](https://github.com/hfawaz/InceptionTime)
    *   *Role in Thesis:* Baseline model to compare Transformer performance against state-of-the-art deep convolutional neural networks on time series.
*   **[ST-GCN] Spatio-Temporal Graph Neural Networks**
    *   *Title:* Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition (AAAI 2018)
    *   *Links:* [arXiv:1801.07455](https://arxiv.org/abs/1801.07455) | [Official GitHub](https://github.com/yysijie/st-gcn)
    *   *Role in Thesis:* Theoretical framework for alternative spatial-temporal point-cloud tracking (Hand-Pose landmarks and ArUco objects).
*   **[MulT] Multimodal Cross-Attention Transformer**
    *   *Title:* Multimodal Transformer for Unaligned Multimodal Language Sequences (ACL 2019)
    *   *Links:* [arXiv:1906.00297](https://arxiv.org/abs/1906.00297) | [Official GitHub](https://github.com/yaohungt/Multimodal-Transformer)
    *   *Role in Thesis:* Reference architecture for advanced multi-sensor fusion using cross-attention mechanisms between human gaze and scene targets.

---

## 2. Hardware, Toolkits & Egocentric Datasets

### Meta Aria Ecosystem
*   **Project Aria Tools & SDK**
    *   *Documentation:* [Project Aria Tools Gen2 Docs](https://facebookresearch.github.io/projectaria_tools/gen2/)
    *   *Core Reference Paper:* *Project Aria: A New Tool for Egocentric Multimodal AI Research* (Meta AI, 2023)
    *   *Role in Thesis:* Foundation for the data acquisition pipeline, VRS file parsing, DEVICE_TIME nanosecond alignment, and MPS (Meta Perception Services) SLAM/Gaze tracking extraction.

### Related Large-Scale Benchmarks
*   **Ego4D / Ego-Exo4D Dataset**
    *   *Papers:* [Ego4D (CVPR 2022)](https://arxiv.org/abs/2110.07058) | [Ego-Exo4D (CVPR 2024)](https://arxiv.org/abs/2311.18259)
    *   *Role in Thesis:* Contextual state-of-the-art validation for egocentric vision tasks, short-term intention anticipation, and human-object interactions using wearable sensors.

---

## 3. Methodology & Auxiliary Foundations

*   **Attention Foundation:** *Attention Is All You Need* (Vaswani et al., NIPS 2017) – [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
*   **Multi-Task Learning:** *An Overview of Multi-Task Learning in Deep Neural Networks* (S. Ruder, 2017) – [arXiv:1706.05098](https://arxiv.org/abs/1706.05098) *(Framework for combining Intention Classification with Hand Position Regression).*
