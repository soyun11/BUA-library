# Library Agent — Browser Use Agent (도서관 자동화)

QLoRA 파인튜닝 Qwen2.5-VL-32B 기반 충남대학교 도서관 웹사이트 자동화 에이전트

## Authors

- **박소윤** (담당: 도서관 자동화, QLoRA 파인튜닝 파이프라인) ← 본 repo 작성자
- 정재현 (담당: 사이버캠퍼스 자동화, FastAPI+WebSocket UI, QLoRA 파인튜닝 파이프라인)
- 진민혁 (담당: 학과홈페이지 자동화, QLoRA 파인튜닝 파이프라인)
- 최민우 (담당: 전자결재 자동화, FastAPI+WebSocket UI, QLoRA 파인튜닝 파이프라인)
- 박경서 (담당: 통합정보시스템 자동화, QLoRA 파인튜닝 파이프라인)
- 이영석 (지도교수)

> KCC 2026 제출 논문 — 충남대학교 Data Network 연구실

상용 LLM(Gemini 2.5 Flash)의 도서관 웹사이트 브라우저 조작 trajectory를 오픈소스 비전-언어 모델(Qwen2.5-VL-32B)에 전이하는 **Teacher-Student QLoRA 파인튜닝 파이프라인**입니다. 도서 검색, 전자자료 조회, MyLibrary 등 도서관 웹사이트 자동화를 자연어 명령만으로 수행합니다.

---

## Demo

> "도서관에서 '강화학습' 관련 추천 도서 검색해줘."

자연어 명령 한 줄로 키워드 검색부터 대출 가능 여부 필터링, 소장 위치·청구기호 조회까지 자동 수행

![Demo](./demo_library.gif)

> **KCC 2026 제출 논문** | 충남대학교 Data Network 연구실
[![Paper](https://img.shields.io/badge/Paper-KCC%202026-blue?style=flat-square)](./paper/kcc2026.pdf)

---

## Results

### 모델별 성능 비교 (도서관 100개 태스크)

| 모델 | 성공률(%) | 평균 시간(초) |
|---|---|---|
| ChatBrowserUse | 96.0 | 48.2 |
| Gemini 2.5 Flash | 76.0 | 94.7 |
| GPT-4o-mini | 46.0 | 155.4 |
| Qwen2.5-VL-32B (FT, 기존)† | 45.0 | 313.1 |
| **Qwen2.5-VL-32B (FT, 개선 후)‡** | **75.0** | **69.0** |

† 초기 평가 시나리오 기준
‡ 시나리오 및 실행 조건 보완 후 재평가 결과. 성공 태스크 기준 평균 시간은 57.2초.

### 카테고리별 성공률 (개선 후, v7)

| 카테고리 | 성공/전체 | 성공률 |
|---|---|---|
| 키워드검색 | 22/25 | 88% |
| 저자검색 | 8/10 | 80% |
| 소장자료조회 | 12/15 | 80% |
| 전자자료 | 7/10 | 70% |
| 도서관이용안내 | 14/20 | 70% |
| 대출반납조회 | 12/20 | 60% |
| **전체** | **75/100** | **75%** |

---

## Architecture

```
[ 오프라인 파인튜닝 단계 ]

Teacher 모델 (Gemini 2.5 Flash)
        │
        ▼
도서관 실행 이력 수집 (100개 태스크, 2,919 step)
        │
        ▼
유효 step 필터링 (반복 wait/동일 액션 제거, 605개 제거 → 20.7%)
        │
        ▼
QLoRA 파인튜닝 (Qwen2.5-VL-32B, r=16, α=32, 4-bit nf4)
  └─ NVIDIA B200 (183GB) 단일 GPU / 약 18분 (1,084초)

[ 실시간 에이전트 추론 단계 ]

사용자 자연어 명령
        │
        ▼
상태 인지 모듈 (DOM 트리 파싱 + 스크린샷)
        │
        ▼
파인튜닝 모델 추론 → 행동 결정 (JSON)
        │
        ▼
행동 제어 모듈 (browser-use / Chrome DevTools Protocol)
        │
        ▼
Observe → Reason → Act 루프 (최대 30 step)
```

---

## Key Methods

### Teacher-Student 파이프라인

- **Stage 1**: Gemini 2.5 Flash로 도서관 웹사이트 성공 trajectory 수집 (100건, 2,919 step)
- **Stage 2**: 반복 wait·동일 액션 반복 step을 비유효 step으로 필터링 (605개 제거)
- **Stage 3**: 약 200토큰의 축약 시스템 프롬프트 적용
- **Stage 4**: Qwen2.5-VL-32B-Instruct QLoRA 파인튜닝
  - LoRA 적용 모듈: attention(`q_proj`, `k_proj`, `v_proj`, `o_proj`) + FFN(`gate_proj`, `up_proj`, `down_proj`)
  - 학습 파라미터: 141.5M / 33.59B (0.42%)

### VLM 입력 데이터 구성

각 학습 샘플은 다음으로 구성됨:
- 사용자 명령
- 축약 시스템 프롬프트
- 현재 브라우저 상태 (`<agent_history>`, `<browser_state>`)
- 상호작용 가능한 DOM 요소 목록 (`[index]<type>text</type>` 형식)
- 스크린샷 이미지 (Base64)
- 정답 액션 (JSON: `{"memory": "...", "next_goal": "...", "actions": [...]}`)

### 실패 원인 분석

개선 후 재평가 기준 실패 태스크 25개의 주요 원인:

| 원인 | 개수 | 비율 |
|---|---|---|
| MyLibrary 내 중첩 메뉴 탐색 실패 | 8 | 32% |
| 정보 위치 파악 실패 및 최대 step 도달 | 6 | 24% |
| 미소장 도서 검색 / 필터링 기능 조작 실패 | 9 | 36% |
| 외국 저자 검색 결과 없음 | 2 | 8% |

수행 시간: step당 평균 4.13초 (중앙값 3.71초, 표준편차 5.92초, 총 1,671 step)

---

## Benchmark

- 충남대학교 도서관 웹사이트(library.cnu.ac.kr) 6개 카테고리, 100개 태스크
- 카테고리: 키워드검색, 저자검색, 전자자료, 소장자료조회, 도서관이용안내, 대출반납조회
- 평가 환경: NVIDIA B200 (183GB), browser-use 0.11.4, vLLM

---

## Tech Stack

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![vLLM](https://img.shields.io/badge/vLLM-009688?style=flat-square)](https://github.com/vllm-project/vllm)

- **모델**: Qwen2.5-VL-32B-Instruct
- **파인튜닝**: QLoRA (r=16, α=32, 4-bit nf4) via PEFT/bitsandbytes
- **브라우저 제어**: browser-use, Chrome DevTools Protocol
- **추론 서버**: vLLM
- **Teacher 모델**: Gemini 2.5 Flash
- **비교 모델**: ChatBrowserUse, GPT-4o-mini

---

## Repository Structure

```
.
├── browser-use/
│   ├── evaluate_finetuned_vl.py   # 파인튜닝 모델 평가 (v7)
│   ├── logged_agent.py            # 에이전트 로깅 wrapper
│   ├── trajectories_finetuned_vl7/  # v7 평가 trajectory (100개)
│   └── data/
│       └── ft_train_vl_gemini_v5.jsonl  # 학습 데이터 (제외, 재현 가능)
├── cnu-finetune/
│   ├── finetune_vl.py             # QLoRA 파인튜닝 스크립트
│   └── finetune_v7.log            # v7 학습 로그
└── .gitignore
```

---

## Setup

```bash
git clone <repo-url>
cd <repo-name>
pip install -r requirements.txt
```

```bash
# .env 설정
CNU_ID=your_student_id
CNU_PASSWORD=your_password
GOOGLE_API_KEY=your_gemini_key
```

---

## Usage

```bash
# Teacher 모델(Gemini)로 trajectory 수집 후 파인튜닝
cd cnu-finetune
python finetune_vl.py

# vLLM으로 파인튜닝 모델 서빙
vllm serve ./output/qwen2.5-vl-32b-cnu-qlora-v7 \
    --port 8001 --gpu-memory-utilization 0.85

# 파인튜닝 모델 평가 (100개 태스크)
cd ../browser-use
python evaluate_finetuned_vl.py 0 100
```

---

## License

MIT License