# Phase 0 — 교차모델 자기보고 도달성 (결과)

> 실행 2026-07-12, `notebooks/Phase0_CrossModel_Colab.ipynb`.
> 원본 JSON: `artifacts/vectors/phase0_{gemma9b,mistral7b,qwen7b}.json`.
> 배경/단일모델 판정: [Phase0_plan.md](Phase0_plan.md), 루트 `phase0_reachability.json`(gemma 최초 실행).

## 질문
α=0(주입 없음)에서 **프롬프트 페르소나 유도**만으로 Likert 자기보고 digit이 움직이나? 이게 **Gemma 특성인가, 모델 불문인가?** (지난 N=1 약점 대응)

## 결과 — 3모델 전부 완전 도달 (persona 기준)

| 모델 | L | neutral E_ft (gen) | persona dE_ft | dE_comp | dN_ft | spec_ratio | cos_readout | rel_diff | 진단 |
|---|---|---|---|---|---|---|---|---|---|
| gemma-2-9b-it | 20 | 3.05 (`3`) | **3.998** | 0.357 | +0.81 | 0.202 | −0.011 | 0.200 | reachable ✅ |
| Mistral-7B-Instruct-v0.3 | 15 | 3.13 (`5`⚠) | **3.984** | 1.277 | −0.88 | 0.221 | +0.011 | 0.230 | reachable ✅ |
| Qwen2.5-7B-Instruct | 13 | 2.99 (`3`) | **3.993** | 1.863 | −0.46 | 0.115 | −0.007 | 0.081 | reachable ✅ |

**persona dE_ft = 3.98~4.00, 3모델 동일.** digit이 1↔5로 완전히 열림, specificity 전부 통과. → "문은 안 잠겼다"는 **모델 불문**(N=1→N=3).

## 강건하게 재현된 3가지
1. **neutral E_ft ≈ 3.0** — 3모델 계측 sane.
2. **persona dE_ft ≈ 4.0** — 프롬프트로 완전 도달, 3모델 동일.
3. **cos_readout ≈ 0** (−0.011 / +0.011 / −0.007) — "의미축 ≠ 숫자토큰 방향"이 cross-model로 재현. 두-방향(readout) 가설을 계속 지지.

## 유도별 dE_ft — persona/grounded 강건, fewshot은 Gemma 전용

| 유도 | gemma | mistral | qwen |
|---|---|---|---|
| persona | 3.998 | 3.984 | 3.993 |
| grounded | 3.983 | 3.903 | 3.921 |
| **fewshot** | **3.051** | **0.559** | **0.825** (spec fail 0.80) |

`recommended_for_vector` = **persona** (3모델 공통). persona가 최강·최청정 공통 추출 재료.

## 모델 차이 (다음 단계에 영향)
- **completion 채점(dE_comp) 발산:** Gemma 0.36 ≪ Mistral 1.28 < Qwen 1.86. first_token으론 셋 다 ~4인데, completion 판독기로 보면 **Gemma 자기보고가 가장 "stuck".** ⇒ Phase 1에서 v_selfreport가 어느 판독기를 타깃하느냐가 모델마다 난이도 다름.
- **Mistral neutral 편향:** 무개입 greedy 생성이 `5`(단 E_ft 3.13이라 dE 계산엔 무해). 기저선이 Gemma보다 덜 깨끗.
- **내부 자국(rel_diff):** Gemma 0.20 · Mistral 0.23 · Qwen 0.08. Qwen은 중간층 내부 신호가 작지만 출력은 완전히 움직임 → cos_readout≈0(분산·비선형 경로) 서사와 일관.

## 범위 / 다음
- 이 정찰은 **프롬프트 수준 도달성만**. 원래의 성격 착시(=**v_behavior 주입** 시 행동만 움직이고 자기보고 digit 고정 `dE≈0.017`)는 **주입 수준**이며 Mistral/Qwen 미검증.
- **다음:** 교차모델 **behavior 주입 실험**(extract→build→steer_eval)으로 착시가 모델 불문인지 검증(→ Phase 4). 주의: `pooled_hidden`의 BOS-드롭이 Qwen(no-BOS)에서 `extract_activations`에 영향 → Qwen 벡터 추출 전 수정 필요.
- 각 모델 주입 층은 `--layer-sweep`으로 재탐색(이 정찰의 auto 층은 도달성엔 무관했으나 주입엔 중요).
