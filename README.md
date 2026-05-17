# Awesome Physical AI [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> 🤖 Physical AI (Robotics & Embodied AI) 분야의 오픈소스 모델, 데이터셋, 시뮬레이터를 체계적으로 정리한 큐레이션 리스트.
> A curated list of open-source models, datasets, and simulators for Physical AI (Robotics & Embodied AI).

[![Models](https://img.shields.io/badge/Models-12-blue)](https://pytorchkorea.github.io/Awesome-Physical-AI)
[![Datasets](https://img.shields.io/badge/Datasets-10-green)](https://pytorchkorea.github.io/Awesome-Physical-AI)
[![Simulators](https://img.shields.io/badge/Simulators-9-purple)](https://pytorchkorea.github.io/Awesome-Physical-AI)
[![Organizations](https://img.shields.io/badge/Organizations-27-orange)](https://pytorchkorea.github.io/Awesome-Physical-AI)
[![Updated](https://img.shields.io/badge/Updated-2026-05-17-lightgrey)](https://github.com/PyTorchKorea/Awesome-Physical-AI)
[![Dashboard](https://img.shields.io/badge/🌐_Dashboard-Live-brightgreen)](https://pytorchkorea.github.io/Awesome-Physical-AI)

> **[👉 인터랙티브 대시보드에서 필터링 및 시각화 보기 | View Interactive Dashboard](https://pytorchkorea.github.io/Awesome-Physical-AI)**

---

## Contents

- [Models](#-models)
- [Datasets](#-datasets)
- [Simulators & Tools](#-simulators--tools)
- [How to Contribute](#-how-to-contribute)
- [Taxonomy](#-taxonomy)

---

## 🤖 Models

> 스타 수 기준 내림차순 정렬 | Sorted by GitHub stars (auto-updated weekly)

| Name | Organization | Year | Category | Hardware | Learning | ⭐ Stars | Links |
|------|-------------|------|----------|----------|----------|---------|-------|
| [LeRobot](https://github.com/huggingface/lerobot) | Hugging Face | 2024 | manipulation | manipulator | IL, diffusion, VLA | 24,061 | [📄](https://arxiv.org/abs/2408.01730) [🤗](https://huggingface.co/lerobot) |
| [π0 (pi-zero)](https://github.com/Physical-Intelligence/openpi) | Physical Intelligence | 2024 | manipulation, whole-body | manipulator, humanoid | VLA, diffusion | 11,865 | [📄](https://arxiv.org/abs/2410.24164) [🤗](https://huggingface.co/physical-intelligence/pi0) |
| [GR00T N1](https://github.com/NVIDIA/Isaac-GR00T) | NVIDIA | 2025 | manipulation, whole-body | humanoid | VLA, IL | 7,026 | [📄](https://arxiv.org/abs/2503.14734) [🤗](https://huggingface.co/nvidia/GR00T-N1-2B) |
| [OpenVLA](https://github.com/openvla/openvla) | Stanford / UC Berkeley | 2024 | manipulation | manipulator | VLA | 6,175 | [📄](https://arxiv.org/abs/2406.09246) [🤗](https://huggingface.co/openvla/openvla-7b) |
| [Mobile ALOHA](https://github.com/MarkFzp/mobile-aloha) | Stanford | 2024 | manipulation, navigation, whole-body | mobile, manipulator | IL | 4,422 | [📄](https://arxiv.org/abs/2401.02117)  |
| [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) | Columbia University | 2023 | manipulation | manipulator | IL, diffusion | 4,148 | [📄](https://arxiv.org/abs/2303.04137)  |
| [ACT (Action Chunking with Transformers)](https://github.com/tonyzhaozh/act) | Stanford | 2023 | manipulation, dexterous | manipulator | IL | 1,945 | [📄](https://arxiv.org/abs/2304.13705)  |
| [Octo](https://github.com/octo-models/octo) | UC Berkeley / Stanford / CMU / others | 2023 | manipulation | manipulator, mobile | IL, VLA | 1,652 | [📄](https://arxiv.org/abs/2405.12213) [🤗](https://huggingface.co/rail-berkeley/octo-base) |
| [HumanPlus](https://github.com/MarkFzp/humanplus) | Stanford | 2024 | manipulation, whole-body | humanoid | IL | 846 | [📄](https://arxiv.org/abs/2406.10454)  |
| [RoboFlamingo](https://github.com/RoboFlamingo/RoboFlamingo) | ByteDance | 2023 | manipulation | manipulator | VLA | 432 | [📄](https://arxiv.org/abs/2311.01378)  |
| [GR-1](https://github.com/bytedance/GR-1) | BAAI / Beijing Academy of AI | 2024 | manipulation | manipulator | VLA, IL | 310 | [📄](https://arxiv.org/abs/2312.13139)  |
| [CrossFormer](https://github.com/rail-berkeley/crossformer) | UC Berkeley / others | 2024 | manipulation | manipulator, mobile | IL, VLA | 282 | [📄](https://arxiv.org/abs/2408.11812) [🤗](https://huggingface.co/rail-berkeley/crossformer) |

---

## 📦 Datasets

> 스타 수 기준 내림차순 정렬 | Sorted by GitHub stars (auto-updated weekly)

| Name | Organization | Year | Category | Source | Modality | Trajectories | ⭐ Stars | Links |
|------|-------------|------|----------|--------|----------|-------------|---------|-------|
| [ManiSkill2](https://github.com/haosulab/ManiSkill) | UC San Diego / Shanghai AI Lab | 2023 | manipulation, dexterous | simulation | rgb, rgbd, proprioception | 36,000 | 2,874 | [📄](https://arxiv.org/abs/2302.04659) [🤗](https://huggingface.co/datasets/haosulab/ManiSkill) |
| [Open X-Embodiment (OXE)](https://github.com/google-deepmind/open_x_embodiment) | Google DeepMind / RT-X Team | 2023 | manipulation | real, teleoperation | rgb, rgbd, proprioception | 1,000,000 | 1,840 | [📄](https://arxiv.org/abs/2310.08864) [🤗](https://huggingface.co/datasets/jxu124/OpenX-Embodiment) |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | UMass Amherst / Bosch | 2023 | manipulation | simulation | rgb, proprioception | 130,000 | 1,840 | [📄](https://arxiv.org/abs/2306.03310) [🤗](https://huggingface.co/datasets/openvla/modified_libero_spatial) |
| [Meta-World](https://github.com/Farama-Foundation/Metaworld) | Stanford / Berkeley | 2019 | manipulation | simulation | proprioception | — | 1,817 | [📄](https://arxiv.org/abs/1910.10897)  |
| [HumanoidBench](https://github.com/carlosferrazza/humanoid-bench) | CMU / UC San Diego / MIT | 2024 | locomotion, manipulation, whole-body | simulation | rgb, proprioception | — | 759 | [📄](https://arxiv.org/abs/2403.10506)  |
| [RoboAgent Dataset](https://github.com/robopen/roboagent) | CMU | 2023 | manipulation | real, teleoperation | rgb, proprioception | 7,500 | 366 | [📄](https://arxiv.org/abs/2309.01918)  |
| [DROID](https://github.com/droid-dataset/droid) | UC Berkeley / Stanford / others | 2024 | manipulation | real, teleoperation | rgb, rgbd, proprioception | 76,000 | 361 | [📄](https://arxiv.org/abs/2403.12945) [🤗](https://huggingface.co/datasets/droid-dataset/droid) |
| [Language-Table](https://github.com/google-research/language-table) | Google | 2023 | manipulation | real, simulation, teleoperation | rgb, proprioception | 600,000 | 356 | [📄](https://arxiv.org/abs/2210.01911) [🤗](https://huggingface.co/datasets/google/language_table) |
| [BridgeData V2](https://github.com/rail-berkeley/bridge_data_v2) | UC Berkeley | 2023 | manipulation | real, teleoperation | rgb, proprioception | 60,096 | 280 | [📄](https://arxiv.org/abs/2308.12952) [🤗](https://huggingface.co/datasets/rail-berkeley/bridge_orig) |
| [RH20T](https://github.com/rh20t/rh20t_api) | Shanghai AI Lab | 2023 | manipulation, dexterous | real, teleoperation | rgb, rgbd, tactile, proprioception, audio | 110,000 | 107 | [📄](https://arxiv.org/abs/2307.00595)  |

---

## 🔬 Simulators & Tools

> 스타 수 기준 내림차순 정렬 | Sorted by GitHub stars (auto-updated weekly)

| Name | Organization | Year | Type | GPU | ROS2 | Language | ⭐ Stars | Paper |
|------|-------------|------|------|-----|------|----------|---------|-------|
| [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) | Genesis Authors (MIT/CMU/Stanford) | 2024 | physics_engine | ✅ | — | python | 28,792 | [📄](https://arxiv.org/abs/2501.00599) |
| [MuJoCo](https://github.com/google-deepmind/mujoco) | DeepMind / Google | 2022 | physics_engine | — | — | c, python, c++ | 13,495 | [📄](https://arxiv.org/abs/2103.09615) |
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab) | NVIDIA | 2023 | rl_framework | ✅ | ✅ | python | 7,182 | [📄](https://arxiv.org/abs/2301.04195) |
| [dm_control](https://github.com/google-deepmind/dm_control) | DeepMind | 2018 | benchmark | — | — | python | 4,581 | [📄](https://arxiv.org/abs/2006.12983) |
| [Brax](https://github.com/google/brax) | Google | 2021 | physics_engine | ✅ | — | python | 3,159 | [📄](https://arxiv.org/abs/2106.13281) |
| [Robosuite](https://github.com/ARISE-Initiative/robosuite) | Stanford / UT Austin | 2020 | benchmark | — | — | python | 2,412 | [📄](https://arxiv.org/abs/2009.12293) |
| [RoboCasa](https://github.com/robocasa/robocasa) | UT Austin | 2024 | full_stack | — | — | python | 1,400 | [📄](https://arxiv.org/abs/2406.02523) |
| [SimplerEnv](https://github.com/simpler-env/SimplerEnv) | UC Berkeley / Google | 2024 | benchmark | — | — | python | 1,062 | [📄](https://arxiv.org/abs/2405.05941) |
| [SAPIEN](https://github.com/haosulab/SAPIEN) | UC San Diego / others | 2020 | physics_engine | ✅ | — | python, c++ | 767 | [📄](https://arxiv.org/abs/2003.08515) |

---

## 🤝 How to Contribute

새 항목을 추가하려면 GitHub Issue를 열어주세요.
To add a new entry, please open a GitHub Issue:

- **[➕ Add a Model](https://github.com/PyTorchKorea/Awesome-Physical-AI/issues/new?template=add-model.yml)**
- **[➕ Add a Dataset](https://github.com/PyTorchKorea/Awesome-Physical-AI/issues/new?template=add-dataset.yml)**
- **[➕ Add a Simulator](https://github.com/PyTorchKorea/Awesome-Physical-AI/issues/new?template=add-simulator.yml)**

이슈가 등록되면 봇이 자동으로 PR을 생성하고, 관리자가 검토 후 머지합니다.
A bot will automatically create a PR from your issue for admin review.

---

## 📐 Taxonomy

### Models

| Field | Valid Values |
|-------|-------------|
| `categories` | `manipulation` · `locomotion` · `navigation` · `dexterous` · `whole-body` · `aerial` |
| `hardware` | `manipulator` · `humanoid` · `quadruped` · `biped` · `mobile` · `drone` · `hand` |
| `learning` | `VLA` · `IL` · `RL` · `diffusion` · `world_model` · `sim2real` |
| `framework` | `pytorch` · `jax` · `tensorflow` |
| `communication` | `ros2` · `grpc` · `lcm` · `zenoh` |

### Datasets

| Field | Valid Values |
|-------|-------------|
| `source` | `real` · `simulation` · `teleoperation` · `human_demo` · `mocap` |
| `modality` | `rgb` · `rgbd` · `depth` · `lidar` · `tactile` · `proprioception` · `audio` · `force_torque` |

---

<sub>📊 Stats auto-updated every Sunday via GitHub Actions · README auto-generated by <code>scripts/generate_site.py</code></sub>
