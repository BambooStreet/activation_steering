# Activation Steering Persona — 문서 인덱스

LLM 페르소나에 **활성화 스티어링**으로 성격(외향성)을 주입하고, 인간 UX 실험
(Vibe Check, CHI 2026)의 재현력을 프롬프트 방식과 비교하는 연구의 파일럿 코드베이스.

## 문서
- [research_overview.md](research_overview.md) — 연구 목적·가설·실험 설계(4조건)·측정 프레임
- [module1_pipeline.md](module1_pipeline.md) — 모듈 1(대조 쌍 생성) 파이프라인·코드 맵·실행법
- [module1_dataset.md](module1_dataset.md) — 최종 데이터셋 스펙·품질 리포트·지표 용어집
- [model_and_cost.md](model_and_cost.md) — 생성 모델 결정(mini/5.4/5.5 비교)·토큰/비용 분석
- [decisions_log.md](decisions_log.md) — 결정 로그·미해결 항목·모듈 2 인계

## 현재 상태 (2026-06)
- **모듈 1 완료**: 외향성 6 facet × 250 = **1,488 대조 쌍** 생성·검증 (`outputs/pairs_raw.jsonl`).
- 생성 모델 **gpt-5.4** 확정(향후). ES facet은 5.4+수정 cue, 나머지 5 facet은 5.5.
- **다음: 모듈 2(검증)** — 생성과 다른 모델로 facet 정확성·극성·minimal-edit·누수 스크리닝.

## 핵심 게이트 (파일럿 성공 기준)
스티어링 벡터가 A·B·C를 통과하는가 — A(Mini-IPIP 척도), B(대화 행동), C(멀티턴 안정성).
통과 못 하면 본 실험 무의미. 자세한 내용은 research_overview.md.
