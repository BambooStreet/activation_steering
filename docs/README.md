# Activation Steering Persona — 문서 인덱스

LLM 페르소나에 **활성화 스티어링**으로 성격(외향성)을 주입하는 연구의 파일럿 코드베이스.
원래 Vibe Check(CHI 2026) 재현이었으나, 스티어링 모델의 **행동↔자기보고 해리**를 발견해
그 조사로 방향을 전환했다. **→ 현재 위치·로드맵은 [STATUS.md](STATUS.md) 먼저 읽기.**

## 최상위 (오리엔테이션)
- [STATUS.md](STATUS.md) — 현재 상태·pivot 서사·로드맵 (새 세션 진입점)
- [research_overview.md](research_overview.md) — 원 연구 배경·가설(A·B·C)·실험 설계(4조건)
- [decisions_log.md](decisions_log.md) — 결정 로그·저장소 정리·Phase 판정

## steering/ — 행동↔자기보고 해리 조사 (현재 초점)
- [steering/Phase0_plan.md](steering/Phase0_plan.md) — Phase 0 자기보고 도달성(완료)
- [steering/phase1_selfreport.md](steering/phase1_selfreport.md) — Phase 1 v_selfreport + 이중해리(설계)

## module1/ — 대조 쌍 생성 (완료된 출발점)
- [module1/module1_pipeline.md](module1/module1_pipeline.md) — 모듈 1 파이프라인·코드 맵·실행법
- [module1/module1_dataset.md](module1/module1_dataset.md) — 데이터셋 스펙·품질 리포트·지표 용어집
- [module1/model_and_cost.md](module1/model_and_cost.md) — 생성 모델 결정(mini/5.4/5.5)·토큰/비용 분석

> 폴더 축은 코드(`module1/` vs `steering/`)와 정렬. 최상위 3개는 두 트랙에 걸치는 문서라 루트 유지.
