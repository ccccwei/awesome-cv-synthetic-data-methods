# 合成数据精选集（2022–2025+）

> *通过合成数据，教会你的模型作任何事情*

*[English Version](README.md)*

一个精心策划的**实用**工具、引擎和论文集合，覆盖计算机视觉的两大类数据合成方式：**仿真驱动（Simulation-based）** 与 **生成式合成（Model-based）**。  
欢迎提交 PR！欢迎点亮star，跟踪实时更新✨

> **合成数据优势**：PBR 渲染 & 物理 → 照片级真实感；**大规模合成**；**自动真值标注**（分割/深度/法线/光流/6D 姿态等）；**可控随机化**（光照、材质、摄像机、背景、遮挡）；批量复现、可大规模扩展，希望该列表能给您视觉项目带来一点帮助。

---

## 目录

- [选型速查表（已复现pipeline）](#选型速查表强烈推荐先看)
- [仿真驱动（Simulation-based）](#仿真驱动simulation-based)
  - [通用引擎与平台](#通用引擎与平台)
  - [机器人/具身AI套件](#机器人具身ai套件)
  - [领域专用仿真平台](#领域专用仿真平台)
  - [标注与流水线工具](#标注与流水线工具)
- [生成式合成（Model-based）](#生成式合成model-based)
  - [图像级生成与编辑](#图像级生成与编辑)
  - [视频/时序一致性](#视频时序一致性)
  - [3D 生成与神经渲染](#3d-生成与神经渲染)
- [最新论文（2022–2025，重点）](#最新论文20222025重点)
- [综述/概述论文](#综述概述论文)
- [经典合成数据集与基准（可选）](#经典合成数据集与基准可选)
- [如何使用此列表](#如何使用此列表)
- [常见坑与实践建议](#常见坑与实践建议)
- [精选列表与主题中心](#精选列表与主题中心)
- [贡献](#贡献)
- [许可证](#许可证)

---

## 选型速查表（已复现数据合成pipeline）

| 目标场景                    | 首选工具/平台                                                | 产出标注                                | 经验贴士                        | 官方链接                                                      |
| --------------------------- | ------------------------------------------------------------ | --------------------------------------- | ------------------------------- | ------------------------------------------------------------- |
| 机器人感知（多视角/标注全） | **Omniverse Replicator + Isaac Sim / Isaac Lab**             | RGB、深度、法线、分割、光流、姿态、接触 | PBR+物理+随机化齐全，云端可扩展 | [Replicator](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html) · [Isaac Sim](https://developer.nvidia.com/isaac/sim) |
| 2D 视觉数据集（检测/分割）  | **Unity Perception / SynthDet** / **BlenderProc2**           | COCO 检测、实例/语义分割、关键点        | 资产库丰富，随机化模块成熟      | [Unity Perception](https://github.com/Unity-Technologies/com.unity.perception) · [SynthDet](https://github.com/Unity-Technologies/SynthDet) · [BlenderProc2](https://dlr-rm.github.io/BlenderProc/) |
| 室内具身导航/重排           | **Habitat 2.0**                                              | 轨迹、状态、深度、语义                  | 大型室内场景 + 交互式任务       | [Habitat 2.0](https://aihabitat.org/) |
| 自动驾驶（传感器组合）      | **CARLA / ℛ-CARLA 扩展**                                     | 多传感器、时序、分割、深度              | 传感器与天气/交通高度可控       | [CARLA](https://carla.org/) · [ℛ-CARLA](https://arxiv.org/html/2506.09629v1) |
| 图像编辑式扩增              | **Stable Diffusion SDXL + ControlNet/IP-Adapter/ComfyUI**    | 文本→图像、图像→图像、蒙版编辑          | 可控引导，易批量                | [SDXL](https://github.com/CompVis/stable-diffusion) · [ControlNet](https://github.com/lllyasviel/ControlNet) · [ComfyUI](https://github.com/comfyanonymous/ComfyUI) |
| 单/多视角→3D                | **3D Gaussian Splatting / Nerfstudio / Zero123++ / TripoSR** | 网格/体积/点-高斯                       | 快速获取高质量 3D 资源          | [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) · [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) · [Zero123++](https://github.com/SUDO-AI-3D/zero123plus) |

---

## 仿真驱动（Simulation-based)

### 通用引擎与平台

- **NVIDIA Omniverse Replicator** — **物理精确**合成流水线；自动真值、领域随机化、分布式渲染。文档：<https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html> · 示例：<https://github.com/NVIDIA-Omniverse/synthetic-data-examples>  
- **Isaac Sim / Isaac Lab** — 机器人仿真与合成数据（感知、操作、基础模型，支持 GPU 大规模并行）。<https://developer.nvidia.com/isaac/sim> · <https://github.com/isaac-sim/IsaacLab>  
- **BlenderProc / BlenderProc2** — 面向 CV 的程序化 Blender 渲染与标注流水线（PBR、域随机化）。文档：<https://dlr-rm.github.io/BlenderProc/> · 论文：<https://arxiv.org/abs/1911.01911>  
- **Unity Perception** — 大规模 2D/3D 合成（检测/分割/姿态/关键点/深度），自带随机化器和传感器。<https://github.com/Unity-Technologies/com.unity.perception>  
- **AI Habitat / Habitat 2.0** — 高性能具身仿真器，支持重排与照片级室内场景。<https://aihabitat.org/> · 论文：<https://arxiv.org/abs/2106.14405>  
- **AirSim** — 面向无人机/自动驾驶的仿真平台（多传感器、天气）。<https://github.com/microsoft/AirSim>

### 机器人/具身AI套件

- **ManiSkill / ManiSkill2/3（基于 SAPIEN）** — 大量操作任务与可复现实验管线；相机/深度/分割标注。<https://github.com/HaoyiZhu/ManiSkill> · <https://sapien.ucsd.edu/>  
- **RoboSuite** — MuJoCo 机器人操作基准套件，易自定义任务与相机。<https://github.com/ARISE-Initiative/robosuite>  
- **RLBench + PyRep** — 机械臂操作任务套件（CoppeliaSim），丰富的抓取/放置任务。<https://github.com/stepjam/RLBench>  
- **Gibson / iGibson / BEHAVIOR-1K** — 室内交互与行为任务大集合。<https://github.com/StanfordVL/iGibson>  
- **ORBIT-Surgical（Isaac）** — 外科手术机器人合成任务与评测。<https://github.com/NVIDIA-Omniverse/orbit-surgical>

### 领域专用仿真平台

- **CARLA / ℛ-CARLA 扩展** — 自动驾驶仿真器；高保真传感器/数字孪生轨。  
  • ℛ-CARLA (2025)：<https://arxiv.org/html/2506.09629v1>  
  • PCLA (2025)：预训练 CARLA 代理的测试基础设施：<https://arxiv.org/html/2503.09385v2>  
- **Omniverse Isaac Sim @ scale** — 参考：用 Replicator 为机器人感知生成 **270 万**张图像。<https://arxiv.org/html/2410.21153v1>

### 标注与流水线工具

- **Unity Dataset Insights** — 解析/巡检 Unity 生成数据集结构。<https://github.com/Unity-Technologies/datasetinsights>  
- **NDDS（Unity 插件）** — 半自动标注（姿态/分割/深度）。<https://github.com/NVIDIA/Dataset_Synthesizer>  
- **Blender-BOP / BOP 格式导出** — 6D 姿态任务常用导出。<https://bop.felk.cvut.cz/>  

---

## 生成式合成（Model-based）

### 图像级生成与编辑

- **Stable Diffusion / SDXL** — 文本/图像到图像，开放生态。<https://github.com/CompVis/stable-diffusion>  
- **ControlNet** — 条件可控（边缘/姿态/深度/法线/分割/草图）。<https://github.com/lllyasviel/ControlNet>  
- **IP-Adapter** — 参考图风格/外观可控迁移。<https://github.com/tencent-ailab/IP-Adapter>  
- **InstructPix2Pix / GLIGEN** — 指令驱动/语言落地的定向编辑。<https://github.com/timothybrooks/instruct-pix2pix> · <https://gligen.github.io/>  
- **ComfyUI / InvokeAI** — 稳定、可编排的批量工作流引擎。<https://github.com/comfyanonymous/ComfyUI> · <https://github.com/invoke-ai/InvokeAI>  
- **(商业) DALL·E / Midjourney** — 快速获得高质量风格样本（注意许可与数据使用条款）。

### 视频/时序一致性

- **AnimateDiff** — 基于扩散的动作驱动一致性。<https://github.com/guoyww/AnimateDiff>  
- **Stable Video Diffusion (SVD)** — 文生视频/图生视频（开源权重）。<https://github.com/Stability-AI/generative-models>

### 3D 生成与神经渲染

- **3D Gaussian Splatting（3DGS）/ gsplat** — 实时神经渲染与重建，适合快速生成 3D 资产。<https://github.com/graphdeco-inria/gaussian-splatting> · <https://github.com/nerfstudio-project/gsplat>  
- **Nerfstudio** — NeRF/3DGS 一站式训练/评测/可视化。<https://github.com/nerfstudio-project/nerfstudio>  
- **Instant-NGP** — NeRF 快速训练与推理。<https://github.com/NVlabs/instant-ngp>  
- **Zero123++ / SyncDreamer / TripoSR / Wonder3D / Shap-E** — 单/少视角到 3D 的主流方法。  
  <https://github.com/SUDO-AI-3D/zero123plus> · <https://arxiv.org/abs/2307.00686> · <https://github.com/VAST-AI-Research/TripoSR> · <https://github.com/xxlong0/Wonder3D> · <https://github.com/openai/shap-e>  
- **GET3D / Magic3D** — 高质量纹理网格/文本到 3D。<https://github.com/nv-tlabs/GET3D> · <https://github.com/daveredrum/Magic3D>

---

## 最新论文（2022–2025，重点）

> 关注**仿真驱动**或**合成→真实收益**有定量评估的工作（零样本/少样本/微调）。

| 论文名称                            |      年份 | 会议/期刊 | 主要贡献                                                 | 技术栈            | 链接                                                         |
| ----------------------------------- | --------: | --------- | -------------------------------------------------------- | ----------------- | ------------------------------------------------------------ |
| **Kubric: 可扩展数据集生成器**      |      2022 | CVPR      | Blender+PyBullet，跨任务高质量标注                       | Blender, PyBullet | [ArXiv](https://arxiv.org/abs/2203.03570) · [PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Greff_Kubric_A_Scalable_Dataset_Generator_CVPR_2022_paper.pdf) |
| **InstaGen: 合成数据强化检测**      |      2024 | CVPR      | 用扩散生成多样训练样本，显著提升检测                     | Diffusion, Det    | [ArXiv](https://arxiv.org/abs/2402.09900)                    |
| **机器人感知的大规模合成数据**      |      2024 | —         | Isaac Sim + Replicator 生成 **270 万**图像，验证真实收益 | Isaac, Replicator | [ArXiv](https://arxiv.org/html/2410.21153v1)                 |
| **ORBIT-Surgical**                  | 2023–2024 | —         | 外科手术机器人合成任务/评测，仿真到现实                  | Isaac, Surgery    | [GitHub](https://github.com/NVIDIA-Omniverse/orbit-surgical) |
| **ℛ-CARLA：数字孪生与高保真传感器** |      2025 | —         | 更高保真度的自动驾驶仿真与传感器模型                     | CARLA 扩展        | [ArXiv](https://arxiv.org/html/2506.09629v1)                 |
| **PCLA：CARLA 代理测试框架**        |      2025 | —         | 预训练代理与系统化场景级测试                             | CARLA             | [ArXiv](https://arxiv.org/html/2503.09385v2)                 |
| **BlenderProc2**                    |     2023+ | —         | 面向 CV 的可复现实验与真实差距缩减                       | Blender           | [文档](https://dlr-rm.github.io/BlenderProc/)                |
| **Habitat 2.0**                     |      2021 | —         | 重排任务、具身学习仿真数据到真实验证                     | Habitat           | [ArXiv](https://arxiv.org/abs/2106.14405)                    |

> **阅读要点**：看清楚**随机化旋钮**（光照/材质/姿态/遮挡/背景）、**标注类型**、**真实验证协议**（零样本/微调）与**落地成本**（资产准备/算力/渲染吞吐）。

---

## 综述/概述论文

| 论文名称                             |      年份 | 主要内容                                 | 链接                                                         |
| ------------------------------------ | --------: | ---------------------------------------- | ------------------------------------------------------------ |
| **计算机视觉中合成数据增强方法综述** |      2024 | 3D 图形、神经渲染、GAN/扩散、任务覆盖    | [ArXiv](https://arxiv.org/abs/2403.10075) · [PDF](https://arxiv.org/pdf/2403.10075) |
| **数据合成方法综述**                 |      2024 | 多领域方法/目标：多样性、平衡、长尾/边界 | [ArXiv](https://arxiv.org/html/2407.03672v1)                 |
| **合成数据生成与机器学习综述**       |      2023 | 统摄多模态/多任务的全景综述              | [ArXiv](https://arxiv.org/abs/2302.04062)                    |
| **Sim2Real in Robotics（推荐）**     | 2022–2024 | 机器人领域的仿真到现实迁移综述           | *搜索关键词即可快速定位多篇*                                 |

---

## 经典合成数据集与基准（可选）

> 便于对标与 sanity-check（多为 2016–2021 的“基石”工作）。

- **GTA5 / SYNTHIA / Virtual KITTI 2** — 语义分割/驾驶场景。
- **SceneNet RGB-D / SunCG** — 室内合成与深度。
- **FlyingChairs / FlyingThings3D / MPI-Sintel** — 光流经典合成数据。
- **BOP Challenge** — 6D 姿态评测与格式标准化。<https://bop.felk.cvut.cz/>

---

## 如何使用此列表

1. **挑选引擎**：按领域与资源匹配（Omniverse/Isaac、Unity、Habitat、BlenderProc）。  
2. **从模板起步**：先跑通官方最小样例（Replicator/ SynthDet/ Kubric），再参数化**资产/相机/光照/材质**。  
3. **丰富标注**：同时导出**深度/法线/分割/光流/姿态**并统一到**COCO、BOP、KITTI**等标准格式。  
4. **评估 Sim→Real**：固定一套真实测试集；先**零样本**再**少样本微调**；与真实-only 基线对比。  
5. **记录随机化**：把“随机化旋钮”与采样分布写进 README，方便重现实验与研究分布偏移。

---

## 常见坑与实践建议

- **资产即效果上限**：材质（PBR）、法线/粗糙度贴图、HDRI 光照决定真实感；拒绝低质量模型。  
- **随机化不等于随机**：有目的地覆盖**背景/阴影/遮挡/长尾**；用拉丁超立方或分层采样提升覆盖率。  
- **摄像机与噪声**：尽量模拟真实**内参/畸变/曝光/噪声/运动模糊**；否则 domain gap 明显。  
- **批量与云渲染**：优先支持**无头渲染**与**分布式**；衡量**吞吐（fps/it/s）**与**单位成本**。  
- **标注一致性**：训练/评估格式对齐（类别、遮挡定义、IoU 度量）；保持数据版本与配置可追溯。  
- **许可与合规**：第三方资产/商用模型生成的数据**使用条款**务必确认。

---

## 精选列表与主题中心

- **awesome-synthetic-data**（Statice）— 跨领域工具与库。<https://github.com/statice/awesome-synthetic-data>  
- **awesome-synthetic-data**（Gretel.ai）— 合成数据资源总汇。<https://github.com/gretelai/awesome-synthetic-data>  
- **awesome-synthetic-datasets** — 构建合成数据集的实用资源。<https://github.com/davanstrien/awesome-synthetic-datasets>  
- **awesome-neural-sbi** — 神经仿真推理（无似然）工具与论文。<https://github.com/smsharma/awesome-neural-sbi>  
- **awesome-amortized-inference** — 摊销推理主题列表。<https://github.com/bayesflow-org/awesome-amortized-inference>  
- **LLM 合成数据（NLP/代理）** — <https://github.com/wasiahmad/Awesome-LLM-Synthetic-Data>

---

## 贡献

- 新增条目请用一句话说明**适用场景**与**可导出的标注**；附**最小上手链接**更佳。  
- 优先：**活跃维护**的仓库与**2022+** 的论文。  
- 若可能，请给出**重现脚本/配置**或数据示例。

---

## 许可证

MIT