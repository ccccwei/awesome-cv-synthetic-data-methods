# Awesome Simulation‑Based Synthetic Data (2022–2025+)

*[中文版本](README_CN.md)*

A curated, **practical** collection of tools, engines, and papers for **simulation‑based synthetic data** across vision, robotics, and embodied AI.  
Pull requests welcome! ✨

> Why simulation? Photo‑realistic rendering + precise ground truth + controllable scenarios = scalable data for training & evaluation (optical flow, detection, pose, navigation, grasping, etc.).

---

## Contents
- [Ecosystems & Tooling](#ecosystems--tooling)
  - [Engines / Platforms](#engines--platforms)
  - [Pipelines & SDKs](#pipelines--sdks)
  - [Unity Perception Suite](#unity-perception-suite)
- [Domain‑Focused Sim Platforms](#domainfocused-sim-platforms)
- [Recent Papers (spotlight)](#recent-papers-spotlight)
- [Survey / Overview Papers](#survey--overview-papers)
- [Awesome Lists & Topic Hubs](#awesome-lists--topic-hubs)
- [How to Use This List](#how-to-use-this-list)
- [Contributing](#contributing)
- [License](#license)

---

## Ecosystems & Tooling

### Engines / Platforms

- **NVIDIA Omniverse Replicator** — Framework for **physically‑accurate** synthetic data pipelines, automatic ground‑truth labels, domain randomization, and scale‑out rendering. Docs: <https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html>. Examples: <https://github.com/NVIDIA-Omniverse/synthetic-data-examples>.

- **Isaac Sim** — Robotics simulation + synthetic data at scale (perception, manipulation, robot foundation models). <https://developer.nvidia.com/isaac/sim>.

- **AI Habitat / Habitat 2.0** — High‑performance embodied‑AI simulator with interactive rearrangement tasks and photo‑realistic indoor scenes. <https://aihabitat.org/>; paper: <https://arxiv.org/abs/2106.14405>.

- **BlenderProc / BlenderProc2** — Procedural Blender pipeline for photo‑realistic rendering; long‑standing workhorse for sim2real. Docs: <https://dlr-rm.github.io/BlenderProc/>; arXiv: <https://arxiv.org/abs/1911.01911>.

### Pipelines & SDKs

- **Kubric (CVPR’22)** — Scalable dataset generator (Blender + PyBullet) with rich GT annotations; used across NeRF, optical flow, etc. Paper: <https://arxiv.org/abs/2203.03570>; PDF: <https://openaccess.thecvf.com/content/CVPR2022/papers/Greff_Kubric_A_Scalable_Dataset_Generator_CVPR_2022_paper.pdf>.

- **Unity Perception** — Large‑scale synthetic dataset generation (labels, randomizers, sensors). <https://github.com/Unity-Technologies/com.unity.perception>.

- **Unity Dataset Insights** — Parse/inspect Unity‑generated datasets. <https://github.com/Unity-Technologies/datasetinsights>.

### Unity Perception Suite

- **SynthDet** — End‑to‑end object detection pipeline with Unity synthetic data + Perception. <https://github.com/Unity-Technologies/SynthDet>.

- **PeopleSansPeople** — Human‑centric synthetic data generator; outputs 2D/3D boxes, segmentation, COCO pose. <https://unity-technologies.github.io/PeopleSansPeople/>.

---

## Domain‑Focused Sim Platforms

- **CARLA / ℛ‑CARLA extensions** — Autonomous driving simulator; recent works add high‑fidelity sensors and digital‑twin tracks.  
  • ℛ‑CARLA (2025): <https://arxiv.org/html/2506.09629v1>  
  • PCLA (2025): testing infra with pretrained CARLA agents: <https://arxiv.org/html/2503.09385v2>

- **Omniverse Isaac Sim @ scale** — Example: 2.7M images for robot perception with Replicator. <https://arxiv.org/html/2410.21153v1>.

---

## Recent Papers (spotlight)

> Focus on 2022–2025 papers that **use simulation to synthesize data** (often with procedural scenes, PBR rendering, or physics) and report downstream gains.

- **Kubric: A Scalable Dataset Generator** (CVPR 2022). Blender + PyBullet; reproducible pipelines and rich supervision across tasks.  
  ArXiv: <https://arxiv.org/abs/2203.03570> · PDF: <https://openaccess.thecvf.com/content/CVPR2022/papers/Greff_Kubric_A_Scalable_Dataset_Generator_CVPR_2022_paper.pdf>

- **Large‑Scale Synthetic Data for Robot Perception** (2024). Isaac Sim + Replicator to generate **2.7M** images; shows real‑world benefits.  
  ArXiv: <https://arxiv.org/html/2410.21153v1>

- **ℛ‑CARLA: High‑Fidelity Sensor Simulations with Digital‑Twin Tracks** (2025). Extends CARLA for better sensor fidelity and scenario realism.  
  ArXiv: <https://arxiv.org/html/2506.09629v1>

- **PCLA: Framework for Testing Autonomous Agents in CARLA** (2025). Infrastructure and pretrained agents to systematize **scenario‑level** testing.  
  ArXiv: <https://arxiv.org/html/2503.09385v2>

- **BlenderProc / BlenderProc2** (2019–2023+). Procedural pipeline that emphasizes photorealism to **reduce the reality gap** for CV training.  
  ArXiv: <https://arxiv.org/abs/1911.01911> · Docs: <https://dlr-rm.github.io/BlenderProc/>

- **Habitat 2.0** (2021, widely used 2022–2025). Interactive rearrangement tasks; photo‑realistic indoor scenes for embodied learning with sim data.  
  ArXiv: <https://arxiv.org/abs/2106.14405>

> **Tip:** When surveying papers, note *renderer/engine*, *label types* (2D/3D boxes, depth, normals, flow, pose), *randomization knobs*, and *real‑world validation* (zero‑shot vs fine‑tune).

---

## Survey / Overview Papers

- **A Survey of Synthetic Data Augmentation Methods in Computer Vision** (2024). Covers 3D graphics‑based synthesis, neural rendering, GAN/VAEs, etc.  
  ArXiv: <https://arxiv.org/abs/2403.10075> · PDF: <https://arxiv.org/pdf/2403.10075>

- **A Survey of Data Synthesis Approaches** (2024). Goals: diversity, balance, domain shift, edge cases; broad methods taxonomy.  
  ArXiv: <https://arxiv.org/html/2407.03672v1>

- **Machine Learning for Synthetic Data Generation: A Review** (2023). Comprehensive review (multi‑domain).  
  ArXiv: <https://arxiv.org/abs/2302.04062>

---

## Awesome Lists & Topic Hubs

- **awesome‑synthetic‑data** (Statice) — Tools & libs across domains. <https://github.com/statice/awesome-synthetic-data>

- **awesome‑synthetic‑data** (Gretel.ai) — Broad synthetic‑data resources. <https://github.com/gretelai/awesome-synthetic-data>

- **awesome‑synthetic‑datasets** (davanstrien) — Pragmatic resources to build synthetic datasets. <https://github.com/davanstrien/awesome-synthetic-datasets>

- **awesome‑neural‑sbi** — Papers/tools for **neural Simulation‑Based Inference** (likelihood‑free). Great for sim‑forward + neural posterior workflows. <https://github.com/smsharma/awesome-neural-sbi>

- **awesome‑amortized‑inference** — Broader amortized inference lists (overlaps SBI). <https://github.com/bayesflow-org/awesome-amortized-inference>

- **LLM synthetic data (for NLP/agents)** — If you also track text/agent simulators: <https://github.com/wasiahmad/Awesome-LLM-Synthetic-Data>.

---

## How to Use This List

1. **Pick an engine** that matches your domain and assets: *Omniverse/Isaac* (robotics, PBR), *Unity Perception* (vision datasets), *Habitat* (embodied), *BlenderProc* (general rendering).  
2. **Start from a template repo** (Omniverse examples, SynthDet, Kubric configs), then **parameterize** assets/cameras/lights/materials to cover edge cases.  
3. **Log labels richly** (depth, normals, flow, segmentation, 6D pose) and **export in standard schemas** (COCO, BOP, KITTI).  
4. **Measure sim→real**: hold out real test sets; try *zero‑shot* then *few‑shot fine‑tuning*; report gains vs real‑only baselines.  
5. **Document randomization knobs** so others can reproduce *distribution‑shift* studies.

---

## Contributing

- Add links with a one‑line purpose and **what labels** it provides.  
- Prefer **actively maintained** repos and **recent (2022+)** papers.  
- Include a minimal “how to reproduce” pointer if available.

---

## License

This list is released under the MIT License.
