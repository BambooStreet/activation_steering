# 결정 로그 & 미해결 항목

## 주요 결정 (모듈 1)
1. **규모/방법**: 시나리오당 다중 변형 폐기 → 논문대로 **시나리오당 1쌍**, 규모는 고유 시나리오로.
   최종 facet당 250쌍(외향성 6 facet = 1,488).
2. **2단계 파이프라인**: 0단계 시나리오 풀 → 1단계 쌍 생성. 단계적 확장(소배치 검토 후 전체).
3. **cue-rotation**: 대조 어휘 쏠림("take charge/stay quiet") 방지 위해 cue를 시나리오별 순환 앵커.
4. **cues 배열화**: 콤마분리 문자열의 내부 콤마 파싱 붕괴(activity) → 배열로 전환.
5. **반응형 cue 수정**: PE 2건 + ES 1건을 능동형으로(약화/neuroticism 드리프트 제거). 원본 백업.
6. **생성 모델 gpt-5.4 확정**: mini는 minimal-edit 붕괴로 탈락, 5.4 채택(5.5보다 저렴). ES만 5.4 재생성,
   나머지 5 facet은 5.5 유지(혼합, `gen_model` 기록).
7. **견고성**: 백오프 재시도 + `QuotaExhausted` fast-fail + 증분 저장 + resume-safe append + Batch API.

## 사건: OpenAI 쿼터 소진
대형 런 중 `insufficient_quota`(429)로 0단계 중단 → 사용자 크레딧 충전 후 재개. 이후
`chat_json`에 쿼터 감지/재시도 추가. 교훈: 대량 런 전 1콜 프로브로 쿼터 확인.

## 미해결 / 사용자 결정 대기
- [ ] **실험 산출물 정리 여부**: `outputs/pairs_es_mini.jsonl`, `pairs_es_54.jsonl`,
  `pairs_test_pe*.jsonl`, `pairs_raw.bak_5.5ES.jsonl`, `batch_input.jsonl` — 기록 보존 vs 삭제.
- [ ] **5 facet도 5.4 통일 재생성 여부**: 현재 혼합(ES 5.4 / 나머지 5.5). 검증된 5.5 유지가 기본 권장.
- [ ] **gpt-5.4 실단가**: 확정 시 비용표(model_and_cost.md) 정확화.

## 모듈 2 인계 (검증)
논문 3중 검증의 파일럿 버전:
- (i) **2차 LLM 스크리닝** — 모듈 2에서 구현. facet 정확성·극성(pos=high/neg=low)·minimal-edit·누수.
  **생성과 다른 모델**로.
- (ii) 사람 판정 — 파일럿 생략(경계 사례만 표본 확인 권장).
- (iii) 30-way 분류기 — 파일럿은 학습 안 함. 대신 LLM 판정으로 6-way facet 분류 + Big Five 도메인 점검.

모듈 2가 처리해야 할 모듈 1 잔여:
1. low_overlap 18쌍 + >18단어 109쌍 필터/축약.
2. 진짜 구성개념 누수(휴리스틱 아닌 분류기) — 특히 외향성 **밖**으로의 누수.
3. facet 극성·정확성 전수 스크리닝 → 불합격 쌍 제거/재생성.

## 산출물 인벤토리
| 경로 | 내용 |
|---|---|
| `outputs/pairs_raw.jsonl` | **최종 1,488 대조 쌍** |
| `data/scenarios_pool.json` | 1,488 고유 시나리오 풀 |
| `data/facets_en.json` | facet 정의·cues(배열, 수정본) |
| `data/facets_en.orig.json` | 원본 cues 백업 |
| `data/scenarios_en.json` | 사람 시드 시나리오 |
| `common.py` / `generate_scenarios.py` / `generate_pairs.py` / `batch_pairs.py` | 파이프라인 |
| `outputs/pairs_es_{mini,54}.jsonl` | 모델 비교 실험 산출물 |
| `outputs/pairs_raw.bak_5.5ES.jsonl` | ES 교체 전 백업(옛 cue) |
