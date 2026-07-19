# Awesome Physical AI — Roadmap & Progress

> 최종 업데이트: 2026-05-03

---

## ✅ 완료된 항목

### 데이터
- [x] 모델 시드 데이터 14개 (`data/models.yaml`)
- [x] 데이터셋 시드 데이터 10개 (`data/datasets.yaml`)
- [x] 분류 체계 정의 (categories / hardware / learning / source / modality)

### 자동화
- [x] 주간 GitHub 스타수·포크수 자동 갱신 (`weekly-update.yml` — 매주 일요일 02:00 UTC)
- [x] 주간 HuggingFace 다운로드수 자동 갱신
- [x] `data/` 변경 시 사이트 즉시 자동 재빌드 (`build-site.yml`)
- [x] PR 생성 시 YAML 스키마 자동 검증 (`validate-pr.yml`)
- [x] 이슈 등록 시 PR 자동 생성 (`process-issue.yml`)

### 대시보드 (GitHub Pages)
- [x] 모델·데이터셋 카드 그리드
- [x] 연도 / 카테고리 / 하드웨어 / 학습방식 / 데이터출처 필터
- [x] 텍스트 검색
- [x] 스타수·최신순·이름순 정렬
- [x] 연도별 막대 차트 + 카테고리 도넛 차트
- [x] 한/영 언어 전환
- [x] 다크/라이트 모드
- [x] 데이터 HTML 직접 embed (파일 프로토콜에서도 동작)

### 기여 워크플로우
- [x] "Add Model" GitHub 이슈 템플릿
- [x] "Add Dataset" GitHub 이슈 템플릿
- [x] `.gitignore`

---

## 🔧 알려진 이슈 / 즉시 해결 필요

- [x] `process-issue.yml` 동작을 위해 GitHub 레포에 라벨 `add-model`, `add-dataset` 수동 생성 필요
- [x] 차트 렌더링 수정 후 실제 GitHub Pages에서 동작 확인 필요
- [x] `process_issue.py`의 체크박스 파싱 로직 실제 이슈 폼으로 테스트 필요
- [x] 모델/데이터/시뮬레이터 추가해도 자동으로 PR 안 생기는 문제 해결
- [x] 모델/데이터/시뮬레이터 추가 시 URL 검증 기능 추가
- [x] HuggingFace Read 권한 토큰 추가해서 모델 정보 접근 불가 문제 해결
- [x] roboagent 데이터셋 깃허브(코드) 링크 접속 불가 문제 해결 및 추후 링크 접속 불가 문제 발생 시 이슈 자동 생성
- [ ] 프론티어 모델 성능 정보 시각화
- [ ] 모델/데이터셋/시뮬레이터 이슈를 만든 사람이 github 봇이 만든 PR의 공동 기여자로 자동 추가되기
- [x] PyTorchKR 조직명 변경에 따른 수정


---

## 📋 TODO — 데이터 확충
## @ 심기택 -> 5/23(토) robot foundation model 세미나 (가안), 5/10(일) 모델/데이터/시뮬레이터 관련 조사 내용 공유
- [ ] 모델 추가 (우선순위 높음) 
  - [x] RoboVLMs, CogACT, RoboMamba 등 2025년 신규 모델
  - [ ] Unitree / Fourier / Agility 등 하드웨어 업체 오픈소스
  - [ ] IsaacGym / MuJoCo Playground 등 시뮬레이터
- [ ] 데이터셋 추가
  - [ ] AgiBot World (2025)
  - [ ] π-data (Physical Intelligence)
  - [x] BEHAVIOR-1K
- [ ] 각 항목에 라이선스 정보 필드 추가 (`license: MIT / Apache-2.0 / CC-BY`)
- [ ] 모델 파라미터 수 필드 추가 (`params: "7B"`)

---

## 📋 TODO — 고도화

### A. 데이터 자동 수집 (난이도: 중) @정인호 5/10(일) 크롤링 동작 간단히 확인 후 동작 구체화
- [ ] `scripts/discover_new.py` 추가
- [ ] arXiv `cs.RO` 카테고리 주간 신규 논문 감지
- [ ] HuggingFace `robotics` 태그 신규 모델 감지
- [ ] 후보 목록을 GitHub Issue로 자동 등록

### B. 모델 비교 기능 (난이도: 중)
- [ ] 대시보드에서 모델 2~3개 선택 → 스펙 나란히 비교하는 뷰
- [ ] GitHub 스타, 하드웨어 타겟, 학습 방식 등 비교 테이블

### C. 필터 URL 파라미터화 (난이도: 하) @강정민 5/10(일) html 코드 관련해서 조사한 내용 + 진행할 방향에 대한 공유
- [x] 필터 상태를 URL 쿼리스트링에 반영
- [x] 예: `?tab=models&category=manipulation&hardware=humanoid`
- [x] SNS·슬랙에서 특정 필터 결과를 바로 공유 가능

### D. 월간 뉴스레터 자동 생성 (난이도: 중)
- [ ] 매월 1일 GitHub Actions 실행
- [ ] 지난 한 달간 신규 항목 + 스타수 급상승 모델 정리
- [ ] GitHub Discussions에 자동 포스팅

### E. 벤치마크 성능 테이블 (난이도: 상)
- [ ] `data/benchmarks.yaml` 추가 (CALVIN, MetaWorld, LIBERO 등 수치 수동 입력)
- [ ] 대시보드에 레이더 차트로 다차원 비교 시각화

### F. 기타
- [ ] 학회 정보
- [ ] 코랩 튜토리얼 파일 제공 여부를 필터링 기준으로 추가
- [ ] 도커/requirements/... 등 환경 파일 제공 여부를 필터링 기준으로 추가
- [ ] 하드웨어 종속성(GPU 사양) - 해당 모델이 어떤 gpu에서 테스트 했는지 태깅
- [ ] 모델별로 커뮤니티성 댓글창 만들기(이슈 다 안 눌러봐도 reproduction/실행 방법 전파 가능)

### 임예윤 5/10(일) 논문 abstract/코드 readme 룰베이스 필터링 기준 검토 (1차) + LLM 프롬프트 (2차)
- [ ] 태그나 요약문이 진실인지 확인하는 프롬프트/테스트 코드 추가 (LLM 프롬프트의 신생 루키로 세상에 이름을 알리고 싶은 경우)

---

## 🗓️ 우선순위

| 항목 | 예상 임팩트 | 난이도 | 추천 시기 |
|------|------------|--------|---------|
| C. URL 파라미터화 | ⭐⭐⭐ | 낮음 | 지금 바로 |
| A. 자동 신규 감지 | ⭐⭐⭐⭐ | 중간 | 데이터 어느정도 쌓인 후 |
| B. 모델 비교 | ⭐⭐⭐ | 중간 | 모델 20개 이상 됐을 때 |
| D. 월간 뉴스레터 | ⭐⭐ | 중간 | 커뮤니티 생긴 후 |
| E. 벤치마크 테이블 | ⭐⭐⭐⭐⭐ | 높음 | 장기 과제 |
