# STATUS — 지금 무엇을 하고 있나 (새 세션 먼저 읽기)

> 새 세션/핸드오프용 진입점. 상세는 각 문서로 링크. 배경 원계획은 [research_overview.md](research_overview.md).

## 한 줄 요약
원래 **모듈식 논문재현**(Vibe Check, 외향성 페르소나)으로 시작했으나, 스티어링 평가(모듈 4b)에서
**"주입 모델의 행동은 크게 바뀌는데 Likert 자기보고 숫자는 3.0에 고정"되는 해리**를 발견 →
그 **behavior ↔ self-report 해리를 파헤치는 쪽으로 연구 방향을 전환**했다. 현재 활성 작업은 전부 이 조사다.

## 방향 전환(pivot) 서사
1. **원계획**: Rahman & Desai(CHI 2026) *Vibe Check* 부분재현. 외향성 단일 축, Gemma 페르소나에 활성화 스티어링.
   모듈 1(대조쌍 생성)→2(검증)→…→5(A·B·C go/no-go 게이트)→8·9(본 대화). (`research_overview.md`)
2. **발견(모듈 4b, `steer_eval`)**: `v_behavior`를 layer 20에 주입하면 자유생성 **행동**은 단조로 움직이나
   (behavior_proj −17.6→+20.7), **자기보고 digit**은 α와 무관하게 ~3.0 고정(`dE≈0.017`).
   = `research_overview.md` **가설 2(척도만 바꾸나 vs 행동까지 — personality illusion)**의 illusion 시그니처.
3. **전환**: 모듈 5로 직진하는 대신, **이 해리 자체를 규명·활용**하는 방향으로. `v_selfreport`(자기보고 정렬 벡터)
   제작과 이중해리 검증이 새 주제.
4. **저장소 정리**(2026-07-08, `decisions_log.md`): 이 전환을 반영해 폴더 분리 —
   `module1/`(끝난 출발점) · `notebooks/`(Colab) · `steering/`(현재 조사 작업장).

## 현재 상태
| 트랙 | 항목 | 상태 |
|---|---|---|
| 원 모듈 | 모듈 1 대조쌍 생성 | ✅ 완료(`outputs/pairs_raw.jsonl` 1,488쌍) |
| 원 모듈 | 모듈 2 검증 / 5 게이트 / 8·9 대화 | ⏸ **보류**(해리 조사 결과 나오면 재개 판단) |
| 스티어링 | 모듈 3·4·4a·4b(추출·벡터·진단·평가) | ✅ 구축·실행됨(gemma-2-9b-it, layer 20) |
| **해리 조사** | **Phase 0** 자기보고 도달성 (gemma-2-9b) | ✅ **완료** → [Phase0_plan.md](steering/Phase0_plan.md), `phase0_reachability.json` |
| **해리 조사** | **Phase 0 교차모델** (Mistral·Qwen·Gemma) | ✅ **완료**(2026-07-12) → [phase0_crossmodel.md](steering/phase0_crossmodel.md), `artifacts/vectors/phase0_*.json` |
| **해리 조사** | **Phase 1** v_selfreport + 이중해리 | 🟡 **설계만**(미구현) → [phase1_selfreport.md](steering/phase1_selfreport.md) |
| **해리 조사** | **Phase 2** 신호소멸 추적(logit-lens 층 sweep) | 💡 **후보**(미계획) |
| **해리 조사** | **Phase 4** 교차모델 behavior 주입(착시 재현) | 🟡 **1차 완료**(2026-07-12): Gemma/Mistral 착시 유지, Qwen 벡터불량 무효 → [phase4_crossmodel_behavior.md](steering/phase4_crossmodel_behavior.md), `artifacts/vectors/steer_*.json` |
| **해리 조사** | **Phase 4b** Qwen α수정 · 채널 플립 | 🟡 **완료**(2026-07-12): "Qwen 통합" 기각(표준화 아티팩트) + **표준화=자기보고 채널** 발견 → [phase4b_standardize_channelflip.md](steering/phase4b_standardize_channelflip.md) |

## 핵심 발견 (조사의 출발점)
- **행동(B)은 살아있고 자기보고(A)는 죽어 있다** — 주입 시. → v_behavior가 digit 지배 방향과 어긋남.
- **Phase 0 판정**: 프롬프트 persona 유도는 digit을 **완전히** 움직임(`dE_ft≈4.0`, gen "5"/"1"), 주입은 안 움직임.
  ⇒ 자기보고 채널은 죽/게이팅/포화가 **아니라** v_behavior가 off-axis일 뿐. 추출 재료 추천 = **persona**.
- **Phase 0 교차모델 재현(N=3)**: gemma-2-9b·Mistral-7B·Qwen2.5-7B **전부** persona `dE_ft≈3.98~4.00`, cos_readout≈0 공통.
  ⇒ 도달성은 **모델 불문**(N=1 약점 해소). 단 completion 채점 dE_comp는 발산(Gemma 0.36 ≪ Qwen 1.86). 상세 [phase0_crossmodel.md](steering/phase0_crossmodel.md).
- **Phase 4 주입 착시 교차모델(1차)**: Gemma(dE_ft 0.045)·Mistral(−0.107) 착시 **유지**(행동은 내향↔외향 일관, digit 고정).
  Qwen은 **무효** — 과주입으로 생성 붕괴 + 벡터 잡음(cos_V_cue 0.018). 교훈: behavior_proj 절대값 교차비교 금지. 상세 [phase4_crossmodel_behavior.md](steering/phase4_crossmodel_behavior.md).
- **Phase 4b 채널 플립(중요)**: Qwen 붕괴 원인=**α 과대**(R=858 뻥튀기), 작은 α선 스티어링 됨. Qwen 자기보고 커플링 보였으나 **Gemma-표준화도 동일 커플링(+0.75)** → "최신=통합" **기각**(표준화 아티팩트).
  **발견**: 같은 Gemma에서 **raw 벡터=행동 채널**(행동O/자기보고X), **표준화 벡터=자기보고 채널**(행동X/자기보고O) → 이중해리에 근접(단 N 누수 −0.80). 교훈: **표준화 벡터엔 cos_V_cue 무효**, α는 R로 스케일 금지, 끝점 지표(dmaxmin) 비단조 놓침. **착시 모델의존/통합 질문은 여전히 미해결.** 상세 [phase4b_standardize_channelflip.md](steering/phase4b_standardize_channelflip.md).
- **Phase 1 가설**: persona high/low를 **답 위치**에서 뽑아 `v_selfreport` 제작 → 주입 시 digit이 움직이고,
  v_behavior와 **다른 방향**(이중해리)임을 2×2로 증명.

## 다음 할 일
1. **(가) 착시 테스트 제대로(Qwen)**: v_behavior를 **raw공간 거대차원 클리핑**(z-score 아님 → 행동 채널 보존) + **α 정상화**(R을 거대차원 제외 재계산/고정) → 조밀 스윕. "Qwen도 착시 있나?"에 답. plan mode 설계.
2. **(나) 자기보고 채널 추적**: 표준화 벡터의 E vs N 특이성 규명(N 직교화) → Phase 1 `v_selfreport` 실마리인지 검증.
3. **Phase 1 v_selfreport 구현**: `steering/phase1_selfreport.py` + `steer_eval.behavior(readout=)`·`phase0.answer_hidden(order=)` 1인자씩 + 이중해리 2×2.
4. 결과 판정 → **Phase 2**(신호소멸 추적) 착수 여부.

## 드리프트 주의 (문서 정정 필요)
- `research_overview.md`는 페르소나 모델을 **"Gemma 2 2B"**로 기술 → 실제 스티어링 실험은 **gemma-2-9b-it, layer 20**.
  (`gemma_common.py` 기본은 2b지만 `STEER_MODEL`/`STEER_LAYER` 환경변수로 9B·20 override.)
- 원 모듈 로드맵(5·8·9)은 해리 조사 결과에 따라 재개/수정될 수 있음 — 현재 활성 아님.

## 문서 지도
- [research_overview.md](research_overview.md) — 원 연구 배경·재현대상·가설(A·B·C).
- [Phase0_plan.md](steering/Phase0_plan.md) — Phase 0 설계(완료).
- [phase0_crossmodel.md](steering/phase0_crossmodel.md) — Phase 0 교차모델 결과(도달성 N=3 재현).
- [phase4_crossmodel_behavior.md](steering/phase4_crossmodel_behavior.md) — Phase 4 교차모델 주입 결과(Gemma/Mistral 착시 유지, Qwen 무효).
- [phase4b_standardize_channelflip.md](steering/phase4b_standardize_channelflip.md) — Phase 4b: Qwen α수정 + "통합" 기각 + 표준화=자기보고 채널 발견.
- [phase1_selfreport.md](steering/phase1_selfreport.md) — Phase 1 설계(현 초점).
- [decisions_log.md](decisions_log.md) — 결정·저장소 정리·Phase 판정 로그.
- [module1_pipeline.md](module1/module1_pipeline.md) / [module1_dataset.md](module1/module1_dataset.md) — 끝난 데이터 생성 모듈.
