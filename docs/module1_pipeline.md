# 모듈 1 — 대조 쌍 생성 파이프라인

논문(Facet-Level Persona Control, arXiv:2602.19157) Section 3을 따름.
목표 규모: **facet당 250쌍 × 6 facet = 1,500쌍** (실제 1,488). 논문은 시나리오당 1쌍이므로
규모는 **고유 시나리오 수**로 채운다.

## 2단계 구조

### 0단계 — 시나리오 풀 구축 (`generate_scenarios.py`)
facet당 250개의 서로 다른 시나리오가 필요 → LLM이 (facet×도메인)마다 한 번에 M개 생성
(응답 내 중복 회피) + 누적 dedup(정규화 + 토큰 Jaccard≥0.8) + 사람 검토.
- 4 도메인(study/work/daily/leisure) × ~62개 = ~250/facet.
- 시나리오는 **상황만** 기술(외향/내향 반응 없음). 반응은 1단계에서.
- 기존 `scenarios_en.json` 사람 시드는 시작점 + 중복 회피 기준으로 재활용.
- 산출: `data/scenarios_pool.json` (병합 모드 — 셀별로 나눠 돌려 누적, 셀마다 증분 저장).

### 1단계 — 대조 쌍 생성 (`generate_pairs.py` / `batch_pairs.py`)
풀의 각 시나리오에 positive(high pole) 1 + minimal-edit negative(low pole) 1 = 1쌍.
- 시나리오당 정확히 1쌍(변형 없음 — 시나리오가 곧 식별자).
- 산출: `outputs/pairs_raw.jsonl`, 각 줄 `{facet, domain, scenario, positive, negative, gen_model}`.

## Hard rules (1단계 프롬프트 강제)
1. **MINIMAL EDIT**: pos/neg 같은 시나리오·구조 공유, 행동 동사구만 반대. 맥락 단어 동일.
2. pos = high pole, neg = low pole (cues 기준).
3. neg = **능동적 low-pole**(단순 부정·감소표현 금지). "less/little/rarely" 금지, 능동 반대 행동으로.
4. 1인칭 현재형, ≤18단어, 한 쌍 길이차 ~4단어.
5. 해당 facet만 표현, 누수 금지.

## cue-rotation (어휘 쏠림 방지)
단일 쌍 방식은 모델이 무난한 대조 어휘로 수렴("take charge/stay quiet" 9/15). 방지책:
`facets_en.json`의 cues.high/low(각 4~5개)를 **시나리오 i마다 `(high[i%n], low[i%n])`로
앵커 주입**해 행동구를 하위행동에 고르게 분산. facet 단위 카운터(도메인 경계 넘어 누적).
→ 적용 후 대조 어휘가 각 cue ~50회씩 균등.

## cues 스키마 (배열)
`facets_en.json`의 cues는 **배열**(문자열 콤마분리는 activity "keeps a slow, relaxed pace"
같은 내부 콤마에서 파싱 붕괴 → 배열로 전환). 원본은 `facets_en.orig.json`.
반응형 cue 2건 수정(→ [module1_dataset.md](module1_dataset.md) 참조):
- positive_emotions low: "rarely shows strong delight"→"responds with calm composure",
  "expresses little outward excitement"→"keeps a flat, neutral expression"
- excitement_seeking low: "finds loud or intense environments overwhelming"→"steers clear of loud or intense settings"

## 코드 맵
| 파일 | 역할 |
|---|---|
| `common.py` | 환경/모델 로딩, `chat_json`(백오프 재시도 + `QuotaExhausted` fast-fail), 텍스트 유사도, cue 분해 |
| `generate_scenarios.py` | 0단계 — 시나리오 풀 구축(증분 저장, 다양성 리포트) |
| `generate_pairs.py` | 1단계 동기 — 시나리오당 1쌍, cue-rotation, 품질 플래그, resume-safe append |
| `batch_pairs.py` | 1단계 Batch API — 동일 로직을 배치로(약 50% 저렴, 모델 검증/대량용) |

## 견고성 장치
- `chat_json`: rate limit/timeout/5xx는 지수 백오프(5·15·45·90s) 재시도, `insufficient_quota`는 즉시 중단.
- `generate_scenarios`: 셀마다 증분 저장(중간 실패 시 재실행으로 이어붙임).
- `generate_pairs --append`: 기존 출력의 (facet,domain,scenario) 건너뜀(중복 방지·재개 안전).

## 실행법
```bash
cp .env.example .env          # OPENAI_API_KEY 입력. GEN_MODEL=gpt-5.4
. .venv/bin/activate
# 0단계(전 facet 시나리오 풀)
python generate_scenarios.py --facets all --target 62
# 1단계 동기
python generate_pairs.py --facets all
# 1단계 Batch(저렴): 예 — 특정 facet
python batch_pairs.py --facets excitement_seeking --model gpt-5.4 --out outputs/pairs_x.jsonl
```
단계적 확장 권장: 소배치(--limit / --domains) → 검토 → 전체.
