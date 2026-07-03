# 모듈 1 — 대조 쌍 생성 (논문 arXiv:2602.19157 Section 3)

외향성 스티어링 벡터(CAA / 이후 기성 SAE) 추출용 **minimal-edit 대조 쌍**을 만든다.
논문대로 **시나리오당 정확히 1쌍**. 규모는 시나리오 수로 채운다.

## 목표 규모
- **facet당 250쌍 × 6 facet = 1,500쌍** (논문 정합). 4 도메인 × ~62/도메인.
- 벡터는 외향성 **도메인 수준 1개**로 추출. facet·도메인은 데이터를 외향성 전체에
  균형 있게 펼치는 구조화 틀(facet별 벡터 아님). 같은 1,500쌍을 CAA·SAE 공유.

## 파이프라인 (2단계)
- **0단계 — 시나리오 풀** (`generate_scenarios.py`): facet×도메인마다 고유 시나리오
  ~62개를 생성(응답 내 중복회피 + 누적 dedup) → 사람 검토 → `data/scenarios_pool.json`.
  시나리오는 **상황만** 기술(반응 없음). 기존 `scenarios_en.json` 시드는 시작점으로 재활용.
- **1단계 — 쌍 생성** (`generate_pairs.py`): 풀의 각 시나리오에 positive 1 + minimal-edit
  negative 1 = 1쌍 → `outputs/pairs_raw.jsonl`.

## Hard rules (1단계 프롬프트 강제)
1. MINIMAL EDIT: pos/neg 같은 시나리오·구조 공유, 행동 동사구만 반대. 맥락 단어 동일.
2. pos=high pole, neg=low pole (cues).  3. neg=능동적 low-pole(단순 부정 X).
4. 1인칭 현재형, ≤18단어, 길이차 ~4단어.  5. 해당 facet만, 누수 금지.

## 입력/출력
- 입력: `data/facets_en.json`(definition,cues), `data/scenarios_en.json`(시드),
  `data/scenarios_pool.json`(0단계 산출).
- 출력: `data/scenarios_pool.json`(0단계), `outputs/pairs_raw.jsonl`(1단계)
  각 줄 `{facet, domain, scenario, positive, negative}`.

## 모델
- `.env`: `OPENAI_API_KEY`, `GEN_MODEL`(기본 gpt-4o; **현재 gpt-5.5**), `GEN_TEMPERATURE`.
- **mini 금지**(minimal-edit 미준수). gpt-5 계열은 temperature 커스텀값 거부(기본 1 고정).

## 단계적 실행
```bash
. .venv/bin/activate
# 0단계 시험: assertiveness × study ~62개
python generate_scenarios.py --facets assertiveness --domains study --target 62
# 1단계 시험: 그 풀로 쌍 생성 (빠른 확인은 --limit 15)
python generate_pairs.py --facets assertiveness --domains study
# 확장: assertiveness 전 도메인(≈250) → 전 facet(≈1,500)
python generate_scenarios.py --facets assertiveness --target 62
python generate_pairs.py --facets assertiveness
python generate_scenarios.py --facets all --target 62
python generate_pairs.py --facets all
```

## 검증
- 0단계: 콘솔 다양성 리포트(유사쌍 수) + 사람 검토(중복·억지·반응누수).
- 1단계: 콘솔 품질 플래그(`>18w`, `len_diff>4`, `negation_in_neg`, `low_overlap`).
- 본격 검증은 **모듈 2**(생성과 다른 모델로 facet 정확성·극성·minimal-edit·누수 스크리닝).
