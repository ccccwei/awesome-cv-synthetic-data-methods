# 基于仿真的合成数据精选集 (2022–2025+)

*[English Version](README.md)*

一个精心策划的**实用**工具、引擎和论文集合，涵盖**基于仿真的合成数据**在视觉、机器人和具身AI领域的应用。  
欢迎提交拉取请求！✨

> 为什么选择仿真？照片级真实感渲染 + 精确的真值标注 + 可控场景 = 可扩展的训练和评估数据（光流、检测、姿态、导航、抓取等）。

---

## 目录
- [生态系统与工具](#生态系统与工具)
  - [引擎/平台](#引擎平台)
  - [流水线与SDK](#流水线与sdk)
  - [Unity感知套件](#unity感知套件)
- [领域专用仿真平台](#领域专用仿真平台)
- [最新论文（重点）](#最新论文重点)
- [综述/概述论文](#综述概述论文)
- [精选列表与主题中心](#精选列表与主题中心)
- [如何使用此列表](#如何使用此列表)
- [贡献](#贡献)
- [许可证](#许可证)

---

## 生态系统与工具

### 引擎/平台

- **NVIDIA Omniverse Replicator** — 用于**物理精确**合成数据流水线的框架，自动真值标注、领域随机化和扩展渲染。文档：<https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html>。示例：<https://github.com/NVIDIA-Omniverse/synthetic-data-examples>。

- **Isaac Sim** — 大规模机器人仿真 + 合成数据（感知、操作、机器人基础模型）。<https://developer.nvidia.com/isaac/sim>。

- **AI Habitat / Habitat 2.0** — 高性能具身AI仿真器，具有交互式重排任务和照片级真实感室内场景。<https://aihabitat.org/>；论文：<https://arxiv.org/abs/2106.14405>。

- **BlenderProc / BlenderProc2** — 用于照片级真实感渲染的程序化Blender流水线；sim2real的长期主力工具。文档：<https://dlr-rm.github.io/BlenderProc/>；arXiv：<https://arxiv.org/abs/1911.01911>。

### 流水线与SDK

- **Kubric (CVPR'22)** — 可扩展数据集生成器（Blender + PyBullet），具有丰富的真值标注；用于NeRF、光流等。论文：<https://arxiv.org/abs/2203.03570>；PDF：<https://openaccess.thecvf.com/content/CVPR2022/papers/Greff_Kubric_A_Scalable_Dataset_Generator_CVPR_2022_paper.pdf>。

- **Unity Perception** — 大规模合成数据集生成（标注、随机化器、传感器）。<https://github.com/Unity-Technologies/com.unity.perception>。

- **Unity Dataset Insights** — 解析/检查Unity生成的数据集。<https://github.com/Unity-Technologies/datasetinsights>。

### Unity感知套件

- **SynthDet** — 端到端目标检测流水线，使用Unity合成数据 + Perception。<https://github.com/Unity-Technologies/SynthDet>。

- **PeopleSansPeople** — 以人为中心的合成数据生成器；输出2D/3D边界框、分割、COCO姿态。<https://unity-technologies.github.io/PeopleSansPeople/>。

---

## 领域专用仿真平台

- **CARLA / ℛ‑CARLA扩展** — 自动驾驶仿真器；最新工作增加了高保真传感器和数字孪生轨道。  
  • ℛ‑CARLA (2025)：<https://arxiv.org/html/2506.09629v1>  
  • PCLA (2025)：使用预训练CARLA代理的测试基础设施：<https://arxiv.org/html/2503.09385v2>

- **Omniverse Isaac Sim @ scale** — 示例：使用Replicator为机器人感知生成270万张图像。<https://arxiv.org/html/2410.21153v1>。

---

## 最新论文（重点）

> 重点关注2022–2025年**使用仿真合成数据**的论文（通常使用程序化场景、PBR渲染或物理），并报告下游收益。

| 论文名称 | 年份 | 会议/期刊 | 主要贡献 | 技术栈 | 链接 |
|---------|------|-----------|----------|--------|------|
| **Kubric: 可扩展数据集生成器** | 2022 | CVPR | Blender + PyBullet；可重现流水线和跨任务的丰富监督 | Blender, PyBullet | [ArXiv](https://arxiv.org/abs/2203.03570) · [PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Greff_Kubric_A_Scalable_Dataset_Generator_CVPR_2022_paper.pdf) |
| **机器人感知的大规模合成数据** | 2024 | - | Isaac Sim + Replicator生成**270万**张图像；显示真实世界收益 | Isaac Sim, Replicator | [ArXiv](https://arxiv.org/html/2410.21153v1) |
| **ℛ‑CARLA：具有数字孪生轨道的高保真传感器仿真** | 2025 | - | 扩展CARLA以获得更好的传感器保真度和场景真实感 | CARLA扩展 | [ArXiv](https://arxiv.org/html/2506.09629v1) |
| **PCLA：CARLA中自主代理测试框架** | 2025 | - | 基础设施和预训练代理，用于系统化**场景级**测试 | CARLA | [ArXiv](https://arxiv.org/html/2503.09385v2) |
| **BlenderProc / BlenderProc2** | 2019–2023+ | - | 强调照片真实感的程序化流水线，以**减少CV训练的现实差距** | Blender | [ArXiv](https://arxiv.org/abs/1911.01911) · [文档](https://dlr-rm.github.io/BlenderProc/) |
| **Habitat 2.0** | 2021 | - | 交互式重排任务；用于具身学习的照片级真实感室内场景，使用仿真数据 | Habitat | [ArXiv](https://arxiv.org/abs/2106.14405) |

> **提示：** 在调研论文时，注意*渲染器/引擎*、*标注类型*（2D/3D边界框、深度、法线、光流、姿态）、*随机化旋钮*和*真实世界验证*（零样本 vs 微调）。

---

## 综述/概述论文

| 论文名称 | 年份 | 主要内容 | 链接 |
|---------|------|----------|------|
| **计算机视觉中合成数据增强方法综述** | 2024 | 涵盖基于3D图形的合成、神经渲染、GAN/VAE等 | [ArXiv](https://arxiv.org/abs/2403.10075) · [PDF](https://arxiv.org/pdf/2403.10075) |
| **数据合成方法综述** | 2024 | 目标：多样性、平衡、领域偏移、边缘情况；广泛方法分类 | [ArXiv](https://arxiv.org/html/2407.03672v1) |
| **合成数据生成的机器学习：综述** | 2023 | 全面综述（多领域） | [ArXiv](https://arxiv.org/abs/2302.04062) |

---

## 精选列表与主题中心

- **awesome‑synthetic‑data** (Statice) — 跨领域工具和库。<https://github.com/statice/awesome-synthetic-data>

- **awesome‑synthetic‑data** (Gretel.ai) — 广泛的合成数据资源。<https://github.com/gretelai/awesome-synthetic-data>

- **awesome‑synthetic‑datasets** (davanstrien) — 构建合成数据集的实用资源。<https://github.com/davanstrien/awesome-synthetic-datasets>

- **awesome‑neural‑sbi** — **神经仿真推理**的论文/工具（无似然）。适用于仿真前向 + 神经后验工作流。<https://github.com/smsharma/awesome-neural-sbi>

- **awesome‑amortized‑inference** — 更广泛的摊销推理列表（与SBI重叠）。<https://github.com/bayesflow-org/awesome-amortized-inference>

- **LLM合成数据（用于NLP/代理）** — 如果您也跟踪文本/代理仿真器：<https://github.com/wasiahmad/Awesome-LLM-Synthetic-Data>。

---

## 如何使用此列表

1. **选择引擎**，匹配您的领域和资产：*Omniverse/Isaac*（机器人、PBR）、*Unity Perception*（视觉数据集）、*Habitat*（具身）、*BlenderProc*（通用渲染）。  
2. **从模板仓库开始**（Omniverse示例、SynthDet、Kubric配置），然后**参数化**资产/相机/灯光/材料以覆盖边缘情况。  
3. **丰富记录标注**（深度、法线、光流、分割、6D姿态）并以**标准模式导出**（COCO、BOP、KITTI）。  
4. **测量仿真→真实**：保留真实测试集；尝试*零样本*然后*少样本微调*；报告与仅真实基线的收益对比。  
5. **记录随机化旋钮**，以便其他人可以重现*分布偏移*研究。

---

## 贡献

- 添加链接时，用一行说明其用途和**提供什么标注**。  
- 优先选择**积极维护**的仓库和**最新（2022+）**论文。  
- 如果可用，包含最小的"如何重现"指针。

---

## 许可证

此列表基于MIT许可证发布。 