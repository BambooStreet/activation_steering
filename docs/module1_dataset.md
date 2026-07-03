# 모듈 1 — 최종 데이터셋 & 품질 리포트

## 데이터셋: `outputs/pairs_raw.jsonl`
- **1,488 대조 쌍** = 외향성 6 facet × 248 (= 4 도메인 × ~62).
- 각 줄: `{"facet","domain","scenario","positive","negative","gen_model"}`.
- 도메인 균등: study/work/daily/leisure 각 372.
- 생성 모델(provenance): **gpt-5.5 1,240** (5 facet) + **gpt-5.4 248** (excitement_seeking, 수정 cue).

## 시나리오 풀: `data/scenarios_pool.json`
- 1,488개 고유 시나리오(6 facet × 4 도메인 × ~62). 셀별 유사쌍(Jaccard≥0.6) 0건.

## 품질 리포트 (전체 1,488)
| 지표 | 값 |
|---|---|
| minimal-edit overlap 평균 | 0.64 |
| attenuation / negation | **0 / 0** (전 facet) |
| low_overlap (<0.3) | 18 (1.2%) |
| >18단어 | 109 (7%) |
| Δw(길이차) 평균 | ~0.9 |

facet별 overlap: assertiveness 0.72, activity 0.69, excitement_seeking 0.54(신규 cue),
warmth 0.63, gregariousness 0.62, positive_emotions 0.63.
(facet마다 양극 어휘 거리가 달라 overlap이 다름 — 결함 아님.)

## 지표 용어집 (왜 중요한가 — 전부 CAA 벡터 순도 점검)
CAA 벡터 = `mean(positive 활성화) − mean(negative 활성화)`. pos/neg가 행동만 빼고 같을수록
차이가 맥락을 상쇄하고 외향성 방향만 남는다.

- **minimal-edit overlap**: 한 쌍의 pos↔neg 토큰 Jaccard `|∩|/|∪|`. 맥락 공유도. 짧은 문장에서
  행동구는 반드시 달라야 하므로 **완벽해도 ~0.5~0.75가 자연 범위**(1.0 아님). 너무 높으면
  대조가 1단어 swap뿐 = 빈약. 절대 합격선 아니라 gross 붕괴 감지용 모니터.
- **low_overlap**: overlap<0.3인 개별 쌍 수(불량품 카운트). 평균보다 이게 더 의미 있음.
- **attenuation**: neg가 "less/little/rarely/mild"로 표현(약화). 저-극은 강도 약함이 아니라
  **반대 행동**이어야 함. 방치 시 neuroticism 등 외향성 밖 드리프트. → 행동구 diff에서 탐지.
- **negation**: neg가 "not/don't/never"(단순 부정). 행동이 아니라 부정어 방향을 벡터가 잡음.
  시나리오 절의 부정어(예 "can't")는 diff에서 제외해 오탐 방지.
- **>18단어 / Δw**: 문장이 짧고(≤18) 평행(길이차 ~4)해야 활성화가 행동에 집중·비교 가능.

> 진짜 판정은 다운스트림 — 모듈 5의 A·B·C 게이트에서 벡터가 실제로 외향성을 움직이는지로 결정.
> overlap 등은 그 전 cheap sanity check.

## 반응형 cue 수정 이력 (극성 순도)
minimal-edit·극성 검증 중 **일부 low cue가 "약화/반응형"**이라 negative가 능동 반대가 아니라
"덜함"·"압도됨"으로 생성되는 문제 발견 → active로 수정:
- positive_emotions low: "rarely shows strong delight"→"responds with calm composure",
  "expresses little outward excitement"→"keeps a flat, neutral expression"
- excitement_seeking low: "finds loud or intense environments overwhelming"→
  "steers clear of loud or intense settings" (neuroticism 드리프트 제거) → ES 5.4로 재생성.
- assertiveness 등은 원래 능동형이라 수정 불필요.
원본 cue는 `data/facets_en.orig.json` 보존.

## 모듈 2로 넘길 잔여 항목
1. low_overlap 18쌍 + >18단어 109쌍 필터/축약.
2. **진짜 구성개념 누수** 판정(surface 어휘 휴리스틱 말고 분류기/LLM). 특히 excitement_seeking↔
   gregariousness 인접성(단, 6 facet 모두 외향성이라 facet간 겹침은 도메인 벡터엔 덜 치명적,
   외향성 **밖**으로의 누수가 진짜 위험).
3. 검증은 **생성과 다른 모델**로.
