# Contributing to Awesome Physical AI

---

## 한국어 가이드

### 기여 방법

기여 유형은 크게 두 가지입니다.

- **데이터 기여** — 새 모델 / 데이터셋 / 시뮬레이터 추가
- **기능 기여** — 대시보드 UI, 스크립트, CI/자동화 개선

---

### 1. 데이터 기여 (모델 / 데이터셋 / 시뮬레이터)

#### 등록 기준

- GitHub 또는 HuggingFace에 코드나 가중치가 공개된 오픈소스 프로젝트
- 논문 또는 공식 프로젝트 페이지가 있는 연구

#### GitHub Issue로 추가 (권장)

| 유형 | 이슈 템플릿 |
|------|------------|
| 모델 | [➕ 모델 추가](https://github.com/PyTorchKR/Awesome-Physical-AI/issues/new?template=add-model.yml) |
| 데이터셋 | [➕ 데이터셋 추가](https://github.com/PyTorchKR/Awesome-Physical-AI/issues/new?template=add-dataset.yml) |
| 시뮬레이터 | [➕ 시뮬레이터 추가](https://github.com/PyTorchKR/Awesome-Physical-AI/issues/new?template=add-simulator.yml) |

이슈를 등록하면 봇이 자동으로 PR을 생성합니다. 관리자가 검토 후 머지합니다.

#### 직접 PR 제출

YAML 파일을 직접 수정해 PR을 제출할 수도 있습니다.

```
data/models.yaml    — 모델
data/datasets.yaml  — 데이터셋
data/tools.yaml     — 시뮬레이터 / 툴
```

PR 제출 전 로컬에서 검증을 먼저 실행하세요.

```bash
pip install -r requirements.txt
python scripts/validate_data.py   # YAML 스키마 검증
python -m pytest tests/ -v        # 단위 테스트
python scripts/generate_site.py   # README / docs/ 재생성 확인
```

---

### 2. 기능 기여 (대시보드 / 스크립트 / 자동화)

기능 기여는 반드시 **Issue → 논의 → PR** 순서로 진행해주세요.  

#### 흐름

```
1. 이슈 등록 — 무엇을, 왜 바꾸려는지 설명
2. 관리자와 방향 합의
3. 포크 → 브랜치 생성 → 코드 작성
4. 테스트 작성 또는 기존 테스트 통과 확인
5. PR 제출
```

#### 수정 범위별 파일 안내

| 수정하고 싶은 것 | 관련 파일 |
|----------------|-----------|
| 대시보드 UI / 필터 / 카드 레이아웃 | `docs/index.html` |
| 차트 | `docs/index.html` (Alpine.js `buildCharts()`) |
| README 자동 생성 | `scripts/generate_site.py` |
| 주간 스타수 업데이트 | `scripts/update_stats.py` |
| YAML 스키마 검증 규칙 | `scripts/validate_data.py` |
| 이슈 → PR 자동화 | `scripts/process_issue.py`, `.github/workflows/process-issue.yml` |
| CI 워크플로우 | `.github/workflows/` |

#### 스타일 규칙

- Python: 외부 포매터 규칙 없음, 기존 코드 스타일 유지
- HTML/JS: Tailwind CSS, Alpine.js 사용
- 새 Python 함수 추가 시 `tests/` 에 단위 테스트 함께 작성

---

### PR 요구사항

모든 PR은 머지 전에 아래 CI 체크를 **모두 통과**해야 합니다.

| CI 체크 | 내용 | Workflow |
|---------|------|----------|
| **Tests** | pytest 단위 테스트 | `test.yml` |
| **Validate Data** | YAML 스키마 검증 | `validate-pr.yml` |

CI가 실패하면 리뷰 요청을 할 수 없습니다.  
---

### 분류 체계 (Taxonomy)

YAML 파일 작성 시 아래 유효값만 사용해야 합니다.

**모델**

| 필드 | 유효값 |
|------|--------|
| `categories` | `manipulation` · `locomotion` · `navigation` · `dexterous` · `whole-body` · `aerial` |
| `hardware` | `manipulator` · `humanoid` · `quadruped` · `biped` · `mobile` · `drone` · `hand` |
| `learning` | `VLA` · `IL` · `RL` · `diffusion` · `world_model` · `sim2real` |
| `framework` | `pytorch` · `jax` · `tensorflow` |
| `communication` | `ros2` · `grpc` · `lcm` · `zenoh` · `other` |

**데이터셋**

| 필드 | 유효값 |
|------|--------|
| `source` | `real` · `simulation` · `teleoperation` · `human_demo` · `mocap` |
| `modality` | `rgb` · `rgbd` · `depth` · `lidar` · `tactile` · `proprioception` · `audio` · `force_torque` |

**시뮬레이터**

| 필드 | 유효값 |
|------|--------|
| `type` | `physics_engine` · `rl_framework` · `benchmark` · `full_stack` |

---

---

## English Guide

### How to Contribute

There are two types of contributions:

- **Data contributions** — adding new models / datasets / simulators
- **Feature contributions** — improving the dashboard UI, scripts, or CI automation

---

### 1. Data Contributions (Models / Datasets / Simulators)

#### Eligibility

- Open-source project with public code or weights on GitHub or HuggingFace
- Has an associated paper or official project page

#### Via GitHub Issue (recommended)

| Type | Template |
|------|----------|
| Model | [➕ Add a Model](https://github.com/PyTorchKR/Awesome-Physical-AI/issues/new?template=add-model.yml) |
| Dataset | [➕ Add a Dataset](https://github.com/PyTorchKR/Awesome-Physical-AI/issues/new?template=add-dataset.yml) |
| Simulator | [➕ Add a Simulator](https://github.com/PyTorchKR/Awesome-Physical-AI/issues/new?template=add-simulator.yml) |

A bot will automatically open a PR from your issue. Maintainers will review and merge.

#### Direct PR

You can also edit the YAML files directly and submit a PR.

```
data/models.yaml    — models
data/datasets.yaml  — datasets
data/tools.yaml     — simulators / tools
```

Run these checks locally before submitting:

```bash
pip install -r requirements.txt
python scripts/validate_data.py   # YAML schema validation
python -m pytest tests/ -v        # unit tests
python scripts/generate_site.py   # verify README / docs/ output
```

---

### 2. Feature Contributions (Dashboard / Scripts / Automation)

Feature contributions must follow the **Issue → Discussion → PR** flow.  

#### Flow

```
1. Open an issue — describe what you want to change and why
2. Align on direction with maintainers
3. Fork → create a branch → write code
4. Write new tests or confirm existing tests pass
5. Submit PR
```

#### File map

| What you want to change | Relevant files |
|------------------------|----------------|
| Dashboard UI / filters / card layout | `docs/index.html` |
| Charts | `docs/index.html` (Alpine.js `buildCharts()`) |
| README auto-generation | `scripts/generate_site.py` |
| Weekly stats update | `scripts/update_stats.py` |
| YAML schema validation rules | `scripts/validate_data.py` |
| Issue → PR automation | `scripts/process_issue.py`, `.github/workflows/process-issue.yml` |
| CI workflows | `.github/workflows/` |

#### Style rules

- Python: no external formatter required — match existing code style
- HTML/JS: uses Tailwind CSS and Alpine.js
- New Python functions must include unit tests in `tests/`

---

### PR Requirements

All PRs must pass the following CI checks before merging.

| CI Check | Description | Workflow |
|----------|-------------|----------|
| **Tests** | pytest unit tests | `test.yml` |
| **Validate Data** | YAML schema validation | `validate-pr.yml` |

If CI fails, you cannot request a review from maintainers.  

---

### Taxonomy

Only the following values are valid in YAML files.

**Models**

| Field | Valid values |
|-------|-------------|
| `categories` | `manipulation` · `locomotion` · `navigation` · `dexterous` · `whole-body` · `aerial` |
| `hardware` | `manipulator` · `humanoid` · `quadruped` · `biped` · `mobile` · `drone` · `hand` |
| `learning` | `VLA` · `IL` · `RL` · `diffusion` · `world_model` · `sim2real` |
| `framework` | `pytorch` · `jax` · `tensorflow` |
| `communication` | `ros2` · `grpc` · `lcm` · `zenoh` · `other` |

**Datasets**

| Field | Valid values |
|-------|-------------|
| `source` | `real` · `simulation` · `teleoperation` · `human_demo` · `mocap` |
| `modality` | `rgb` · `rgbd` · `depth` · `lidar` · `tactile` · `proprioception` · `audio` · `force_torque` |

**Simulators**

| Field | Valid values |
|-------|-------------|
| `type` | `physics_engine` · `rl_framework` · `benchmark` · `full_stack` |

---

<sub>Made with ❤️ by <a href="https://github.com/PyTorchKR">PyTorch KR</a></sub>
