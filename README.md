# Awesome Synthetic Data Collection (2022–2025+)

> *Through synthetic data, teach your models to do anything*

*[中文版本](README_CN.md)*

A carefully curated **practical** collection of tools, engines, and papers covering two major approaches to synthetic data in computer vision: **Simulation-based** and **Model-based generation**.  
Pull requests welcome! Star to track real-time updates ✨

> **Why synthetic data?** PBR rendering & physics → photo-realistic; **large-scale synthesis**; **automatic ground truth** (segmentation/depth/normals/optical flow/6D pose); **controllable randomization** (lighting, materials, cameras, backgrounds, occlusion); batch reproduction, scalable expansion. We hope this list brings some help to your vision projects.

---

## Contents

- [Selection Quick Reference (Strongly Recommended First)](#selection-quick-reference-strongly-recommended-first)
- [Simulation-based](#simulation-based)
  - [General Engines & Platforms](#general-engines--platforms)
  - [Robotics/Embodied AI Suites](#roboticsembodied-ai-suites)
  - [Domain-Specific Simulation Platforms](#domain-specific-simulation-platforms)
  - [Annotation & Pipeline Tools](#annotation--pipeline-tools)
- [Model-based Generation](#model-based-generation)
  - [Image-level Generation & Editing](#image-level-generation--editing)
  - [Video/Temporal Consistency](#videotemporal-consistency)
  - [3D Generation & Neural Rendering](#3d-generation--neural-rendering)
- [Recent Papers (2022–2025, Focus)](#recent-papers-20222025-focus)
- [Survey/Overview Papers](#surveyoverview-papers)
- [Classic Synthetic Datasets & Benchmarks (Optional)](#classic-synthetic-datasets--benchmarks-optional)
- [How to Use This List](#how-to-use-this-list)
- [Common Pitfalls & Practical Tips](#common-pitfalls--practical-tips)
- [Curated Lists & Topic Hubs](#curated-lists--topic-hubs)
- [Contributing](#contributing)
- [License](#license)

---

## Selection Quick Reference (Strongly Recommended First)

| Target Scenario | Preferred Tool/Platform | Output Annotations | Experience Tips | Official Links |
|-----------------|-------------------------|-------------------|-----------------|----------------|
| Robot Perception (Multi-view/Full Annotations) | **Omniverse Replicator + Isaac Sim / Isaac Lab** | RGB, depth, normals, segmentation, optical flow, pose, contact | PBR+physics+randomization complete, cloud scalable | [Replicator](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html) · [Isaac Sim](https://developer.nvidia.com/isaac/sim) |
| 2D Vision Datasets (Detection/Segmentation) | **Unity Perception / SynthDet** / **BlenderProc2** | COCO detection, instance/semantic segmentation, keypoints | Rich asset library, mature randomization modules | [Unity Perception](https://github.com/Unity-Technologies/com.unity.perception) · [SynthDet](https://github.com/Unity-Technologies/SynthDet) · [BlenderProc2](https://dlr-rm.github.io/BlenderProc/) |
| Indoor Embodied Navigation/Rearrangement | **Habitat 2.0** | Trajectories, states, depth, semantics | Large indoor scenes + interactive tasks | [Habitat 2.0](https://aihabitat.org/) |
| Autonomous Driving (Sensor Combinations) | **CARLA / ℛ-CARLA Extensions** | Multi-sensor, temporal, segmentation, depth | Highly controllable sensors & weather/traffic | [CARLA](https://carla.org/) · [ℛ-CARLA](https://arxiv.org/html/2506.09629v1) |
| Image Editing Expansion | **Stable Diffusion SDXL + ControlNet/IP-Adapter/ComfyUI** | Text→image, image→image, mask editing | Controllable guidance, easy batching | [SDXL](https://github.com/CompVis/stable-diffusion) · [ControlNet](https://github.com/lllyasviel/ControlNet) · [ComfyUI](https://github.com/comfyanonymous/ComfyUI) |
| Single/Multi-view→3D | **3D Gaussian Splatting / Nerfstudio / Zero123++ / TripoSR** | Mesh/volume/point-gaussian | Fast acquisition of high-quality 3D assets | [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) · [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) · [Zero123++](https://github.com/SUDO-AI-3D/zero123plus) |

---

## Simulation-based

### General Engines & Platforms

- **NVIDIA Omniverse Replicator** — **Physically accurate** synthetic data pipeline; automatic ground truth, domain randomization, distributed rendering. Docs: <https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html> · Examples: <https://github.com/NVIDIA-Omniverse/synthetic-data-examples>  
- **Isaac Sim / Isaac Lab** — Robotics simulation and synthetic data (perception, manipulation, foundation models, GPU-scale parallel support). <https://developer.nvidia.com/isaac/sim> · <https://github.com/isaac-sim/IsaacLab>  
- **BlenderProc / BlenderProc2** — CV-oriented procedural Blender rendering and annotation pipeline (PBR, domain randomization). Docs: <https://dlr-rm.github.io/BlenderProc/> · Paper: <https://arxiv.org/abs/1911.01911>  
- **Unity Perception** — Large-scale 2D/3D synthesis (detection/segmentation/pose/keypoints/depth), built-in randomizers and sensors. <https://github.com/Unity-Technologies/com.unity.perception>  
- **AI Habitat / Habitat 2.0** — High-performance embodied simulator, supports rearrangement and photo-realistic indoor scenes. <https://aihabitat.org/> · Paper: <https://arxiv.org/abs/2106.14405>  
- **AirSim** — UAV/autonomous driving simulation platform (multi-sensor, weather). <https://github.com/microsoft/AirSim>

### Robotics/Embodied AI Suites

- **ManiSkill / ManiSkill2/3 (SAPIEN-based)** — Large-scale manipulation tasks and reproducible experimental pipelines; camera/depth/segmentation annotations. <https://github.com/HaoyiZhu/ManiSkill> · <https://sapien.ucsd.edu/>  
- **RoboSuite** — MuJoCo robotics manipulation benchmark suite, easy custom task and camera setup. <https://github.com/ARISE-Initiative/robosuite>  
- **RLBench + PyRep** — Robotic arm manipulation task suite (CoppeliaSim), rich grasping/placing tasks. <https://github.com/stepjam/RLBench>  
- **Gibson / iGibson / BEHAVIOR-1K** — Large collection of indoor interaction and behavior tasks. <https://github.com/StanfordVL/iGibson>  
- **ORBIT-Surgical (Isaac)** — Surgical robotics synthetic tasks and evaluation. <https://github.com/NVIDIA-Omniverse/orbit-surgical>

### Domain-Specific Simulation Platforms

- **CARLA / ℛ-CARLA Extensions** — Autonomous driving simulator; high-fidelity sensors/digital twin tracks.  
  • ℛ-CARLA (2025): <https://arxiv.org/html/2506.09629v1>  
  • PCLA (2025): Testing infrastructure for pre-trained CARLA agents: <https://arxiv.org/html/2503.09385v2>  
- **Omniverse Isaac Sim @ scale** — Reference: Using Replicator to generate **2.7 million** images for robot perception. <https://arxiv.org/html/2410.21153v1>

### Annotation & Pipeline Tools

- **Unity Dataset Insights** — Parse/inspect Unity-generated dataset structures. <https://github.com/Unity-Technologies/datasetinsights>  
- **NDDS (Unity Plugin)** — Semi-automatic annotation (pose/segmentation/depth). <https://github.com/NVIDIA/Dataset_Synthesizer>  
- **Blender-BOP / BOP Format Export** — Common export for 6D pose tasks. <https://bop.felk.cvut.cz/>  

---

## Model-based Generation

### Image-level Generation & Editing

- **Stable Diffusion / SDXL** — Text/image-to-image, open ecosystem. <https://github.com/CompVis/stable-diffusion>  
- **ControlNet** — Conditional control (edges/pose/depth/normals/segmentation/sketches). <https://github.com/lllyasviel/ControlNet>  
- **IP-Adapter** — Reference image style/appearance controllable transfer. <https://github.com/tencent-ailab/IP-Adapter>  
- **InstructPix2Pix / GLIGEN** — Instruction-driven/language-grounded directed editing. <https://github.com/timothybrooks/instruct-pix2pix> · <https://gligen.github.io/>  
- **ComfyUI / InvokeAI** — Stable, composable batch workflow engines. <https://github.com/comfyanonymous/ComfyUI> · <https://github.com/invoke-ai/InvokeAI>  
- **(Commercial) DALL·E / Midjourney** — Quick access to high-quality style samples (note licensing and data usage terms).

### Video/Temporal Consistency

- **AnimateDiff** — Diffusion-based motion-driven consistency. <https://github.com/guoyww/AnimateDiff>  
- **Stable Video Diffusion (SVD)** — Text-to-video/image-to-video (open source weights). <https://github.com/Stability-AI/generative-models>

### 3D Generation & Neural Rendering

- **3D Gaussian Splatting (3DGS) / gsplat** — Real-time neural rendering and reconstruction, suitable for fast 3D asset generation. <https://github.com/graphdeco-inria/gaussian-splatting> · <https://github.com/nerfstudio-project/gsplat>  
- **Nerfstudio** — NeRF/3DGS one-stop training/evaluation/visualization. <https://github.com/nerfstudio-project/nerfstudio>  
- **Instant-NGP** — Fast NeRF training and inference. <https://github.com/NVlabs/instant-ngp>  
- **Zero123++ / SyncDreamer / TripoSR / Wonder3D / Shap-E** — Mainstream methods from single/few views to 3D.  
  <https://github.com/SUDO-AI-3D/zero123plus> · <https://arxiv.org/abs/2307.00686> · <https://github.com/VAST-AI-Research/TripoSR> · <https://github.com/xxlong0/Wonder3D> · <https://github.com/openai/shap-e>  
- **GET3D / Magic3D** — High-quality textured meshes/text-to-3D. <https://github.com/nv-tlabs/GET3D> · <https://github.com/daveredrum/Magic3D>

---

## Recent Papers (2022–2025, Focus)

> Focus on papers that use **simulation-based** or **synthetic→real benefits** with quantitative evaluation (zero-shot/few-shot/fine-tuning).

| Paper Title | Year | Conference/Journal | Main Contribution | Tech Stack | Links |
|-------------|------|-------------------|------------------|------------|-------|
| **Kubric: A Scalable Dataset Generator** | 2022 | CVPR | Blender+PyBullet, cross-task high-quality annotations | Blender, PyBullet | [ArXiv](https://arxiv.org/abs/2203.03570) · [PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Greff_Kubric_A_Scalable_Dataset_Generator_CVPR_2022_paper.pdf) |
| **InstaGen: Synthetic Data Boosting Detection** | 2024 | CVPR | Using diffusion to generate diverse training samples, significantly improving detection | Diffusion, Detection | [ArXiv](https://arxiv.org/abs/2402.09900) |
| **Large-Scale Synthetic Data for Robot Perception** | 2024 | — | Isaac Sim + Replicator generating **2.7 million** images, validating real-world benefits | Isaac, Replicator | [ArXiv](https://arxiv.org/html/2410.21153v1) |
| **ORBIT-Surgical** | 2023–2024 | — | Surgical robotics synthetic tasks/evaluation, simulation to reality | Isaac, Surgery | [GitHub](https://github.com/NVIDIA-Omniverse/orbit-surgical) |
| **ℛ-CARLA: Digital Twins & High-Fidelity Sensors** | 2025 | — | Higher fidelity autonomous driving simulation and sensor models | CARLA Extension | [ArXiv](https://arxiv.org/html/2506.09629v1) |
| **PCLA: CARLA Agent Testing Framework** | 2025 | — | Pre-trained agents and systematic scenario-level testing | CARLA | [ArXiv](https://arxiv.org/html/2503.09385v2) |
| **BlenderProc2** | 2023+ | — | CV-oriented reproducible experiments and reality gap reduction | Blender | [Docs](https://dlr-rm.github.io/BlenderProc/) |
| **Habitat 2.0** | 2021 | — | Rearrangement tasks, embodied learning simulation data to real-world validation | Habitat | [ArXiv](https://arxiv.org/abs/2106.14405) |

> **Reading Points**: Clearly understand **randomization knobs** (lighting/materials/pose/occlusion/background), **annotation types**, **real-world validation protocols** (zero-shot/fine-tuning) and **deployment costs** (asset preparation/compute/render throughput).

---

## Survey/Overview Papers

| Paper Title | Year | Main Content | Links |
|-------------|------|--------------|-------|
| **A Survey of Synthetic Data Augmentation Methods in Computer Vision** | 2024 | 3D graphics, neural rendering, GAN/diffusion, task coverage | [ArXiv](https://arxiv.org/abs/2403.10075) · [PDF](https://arxiv.org/pdf/2403.10075) |
| **A Survey of Data Synthesis Approaches** | 2024 | Multi-domain methods/objectives: diversity, balance, long-tail/boundaries | [ArXiv](https://arxiv.org/html/2407.03672v1) |
| **Synthetic Data Generation and Machine Learning: A Review** | 2023 | Comprehensive multi-modal/multi-task overview | [ArXiv](https://arxiv.org/abs/2302.04062) |
| **Sim2Real in Robotics (Recommended)** | 2022–2024 | Simulation to reality transfer survey in robotics | *Search keywords to quickly locate multiple papers* |

---

## Classic Synthetic Datasets & Benchmarks (Optional)

> Useful for benchmarking and sanity-checking (mostly "foundational" works from 2016–2021).

- **GTA5 / SYNTHIA / Virtual KITTI 2** — Semantic segmentation/driving scenes.
- **SceneNet RGB-D / SunCG** — Indoor synthesis and depth.
- **FlyingChairs / FlyingThings3D / MPI-Sintel** — Classic synthetic optical flow data.
- **BOP Challenge** — 6D pose evaluation and format standardization. <https://bop.felk.cvut.cz/>

---

## How to Use This List

1. **Choose Engine**: Match by domain and resources (Omniverse/Isaac, Unity, Habitat, BlenderProc).  
2. **Start from Templates**: First run official minimal examples (Replicator/SynthDet/Kubric), then parameterize **assets/cameras/lighting/materials**.  
3. **Rich Annotations**: Export **depth/normals/segmentation/optical flow/pose** simultaneously and unify to **COCO, BOP, KITTI** and other standard formats.  
4. **Evaluate Sim→Real**: Fix a real test set; try **zero-shot** then **few-shot fine-tuning**; compare with real-only baselines.  
5. **Document Randomization**: Write "randomization knobs" and sampling distributions into README for easy experiment reproduction and distribution shift research.

---

## Common Pitfalls & Practical Tips

- **Assets Determine Quality Ceiling**: Materials (PBR), normal/roughness maps, HDRI lighting determine realism; reject low-quality models.  
- **Randomization ≠ Random**: Purposefully cover **backgrounds/shadows/occlusion/long-tails**; use Latin hypercube or stratified sampling to improve coverage.  
- **Cameras & Noise**: Try to simulate real **intrinsics/distortion/exposure/noise/motion blur**; otherwise domain gap is obvious.  
- **Batch & Cloud Rendering**: Prioritize **headless rendering** and **distributed** support; measure **throughput (fps/it/s)** and **unit cost**.  
- **Annotation Consistency**: Align training/evaluation formats (categories, occlusion definitions, IoU metrics); maintain data version and configuration traceability.  
- **Licensing & Compliance**: Third-party assets/commercial model-generated data **usage terms** must be confirmed.

---

## Curated Lists & Topic Hubs

- **awesome-synthetic-data** (Statice) — Cross-domain tools and libraries. <https://github.com/statice/awesome-synthetic-data>  
- **awesome-synthetic-data** (Gretel.ai) — Comprehensive synthetic data resources. <https://github.com/gretelai/awesome-synthetic-data>  
- **awesome-synthetic-datasets** — Practical resources for building synthetic datasets. <https://github.com/davanstrien/awesome-synthetic-datasets>  
- **awesome-neural-sbi** — Neural simulation-based inference (likelihood-free) tools and papers. <https://github.com/smsharma/awesome-neural-sbi>  
- **awesome-amortized-inference** — Amortized inference topic lists. <https://github.com/bayesflow-org/awesome-amortized-inference>  
- **LLM Synthetic Data (NLP/Agents)** — <https://github.com/wasiahmad/Awesome-LLM-Synthetic-Data>

---

## Contributing

- New entries should use one sentence to explain **applicable scenarios** and **exportable annotations**; including **minimal getting-started links** is better.  
- Priority: **actively maintained** repositories and **2022+** papers.  
- If possible, please provide **reproduction scripts/configurations** or data examples.

---

## License

MIT
