# Phase 1 — v_selfreport 제작 + 이중해리 (설계)

> 이 문서는 **설계/프로토콜**만 기록한다. 코드(`steering/phase1_selfreport.py`, `steer_eval.py` 수정,
> `Phase1_SelfReport_Colab.ipynb`)와 실행은 **다음 구현 턴**. → [Phase0_plan.md](Phase0_plan.md) 후속.

## Context (왜)
Phase 0 판정(`phase0_reachability.json`, gemma-2-9b-it, layer 20, α=0):
- 자기보고 digit 채널은 **프롬프트 persona 유도로 완전히 열린다** — `dE_ft≈4.0`(persona/grounded), gen "5"/"1".
  neutral 재현 `E_ft 3.08`(계측 sanity OK).
- 반면 **벡터 주입으로는 안 열린다** — 이전 steer_eval에서 `dE≈0.017`(digit ~3.0 고정), behavior_proj는 −17.6→+20.7.
- ⇒ 자기보고 채널은 죽/게이팅/포화가 **아니고**, **v_behavior가 digit 지배 방향과 어긋나 있을 뿐**.
- 추출 재료 추천 = **persona**(dE 최대·특이성 OK `dN 0.88` 비율 0.22; grounded는 `dN 1.77` N누수, fewshot은 내부 발자국 최소 `rel_diff 0.058`).
- 내부 caveat: layer 20에서 `cos(high−low, W_U['5']−W_U['1']) ≈ 0`(전 유도). 나이브 logit-lens가 softcapping+하류 22층을 무시한 나쁜 프록시라서지 신호 부재가 아님(출력은 완전히 움직임).

## 목표
persona high/low 대조를 **답 위치**에서 뽑아 **v_selfreport**를 만들고, (a) 주입 시 digit이 실제로 움직이는지,
(b) v_behavior와 **인과적으로 다른 방향인지(이중해리)**를 검증한다.
**강도 = 실용적 1차**: `avg_both` 단일 벡터 + 2×2 이중해리 + cos 진단 + N-특이성 + answer-pos-only 보조.
(3변형 비교·N-직교화는 조건부/보류.)

---

## 1. v_selfreport 제작 프로토콜 (avg_both 단일 벡터)

**레버(추출 개입).** persona high(외향)/low(내향) prefix — `phase0_reachability._PERSONA_TRAIT`(:41-52) 재사용.
**문항은 고정하고 persona만 스왑** = Phase 0과 동일 레버(그래서 digit이 움직인다는 게 이미 검증됨).

**프로브 세트.** `data/facets_en.json`의 6 facet cue(**고 25 + 저 25 = 50개**)를 1인칭 자기보고 문장화
(`phase0_reachability._to_first_person`, :71-78; "greets others warmly" → "I greet others warmly.").
**양극 다 포함** → 어휘 균형(cue-투영 오염↓). `--n-probes`로 상한.

**척도 순서 — asc·desc 둘 다 추출해 평균.** (핵심 방법론)
- 근거: IPIP 외향 4문항은 정방향 2 + 역방향 2 혼합(`steer_eval.IPIP`, :34-39)이고, 성공 지표 `scale_score`는
  asc+desc·역채점 평균(`steer_eval.py:174-183`)이라 **순서-불변 의미축**을 보상한다.
- asc-only로 추출하면 desc에서 반대로 작동하는 **digit-token 성분**이 실려 washout된다.
  asc+desc 평균은 그 성분을 상쇄시켜 "외향 자기보고" 의미 방향만 남긴다.
- (이 상쇄가 Phase 0 `cos_readout≈0`의 원인이기도 하다 — 정/역 문항에서 digit 성분이 서로 지워짐.)

**추출.** 각 프로브 `s`, 각 순서 `o∈{asc,desc}`:
`h = answer_hidden(model, tok, s, …, L=20, prefix=_PERSONA_TRAIT[pole])` — 마지막 토큰, `hidden_states[L+1]`
(`phase0_reachability.answer_hidden`, :159-165; **단 현재 "asc" 하드코딩 → `order` 인자 추가 필요**).
→ 프로브별 `d[s] = mean_o(h_high − h_low)` → **프로브별 L2 정규화** `d̂[s]`(등가중) → 평균 `raw = mean_s d̂[s]`.

**정제 — v_behavior와 동일 처리.** `refine(raw, actL)`(`build_vectors.py:32-37`): 전역 평균활성 `mu` 성분 제거 후 단위화.
- **`mu`는 반드시 v_behavior가 쓴 것과 같은 CAA `pooled_mean[:, 20, :].mean(0)`** — 같은 프레임에서 같은 rank-1
  오프셋을 제거해야 §2의 `cos(v_sr, v_beh)`가 공정하다.
- 캐시 없으면 answer-hidden 평균으로 **μ-fallback**(경고 로깅, cos 비교성 저하 명시).

**부호.** `high−low`(high=외향) → **+α 주입이 외향↑** (v_behavior=pos−neg, pos=외향과 정합).

**held-out 검증셋.** 표준 **IPIP 외향 4문항**(추출 프로브=facet cue와 disjoint) → 주입 후 digit `dE` 측정.
구조적으로 분리되어 순환이 아니며, trait 내 일반화를 본다.

**저장.** `artifacts/vectors/layer20_selfreport_v1.npy`(단위, fp32) + `layer20_selfreport_meta.json`
`{layer, R, induction:"persona", n_probes, orders, both_poles, coherence, hidden, cos_to_behavior}`.
(coherence = `d̂[s]` 간 평균 pairwise 코사인 — 낮으면 벡터가 잡음/약함.)

## 2. 벡터 진단 (cos 표)
raw 코사인(`steer_eval.py:280-283` 패턴). 필요시 `mu`로 동일 mean-center 후 비교(2차).

| cos | 의미 |
|---|---|
| **cos(v_sr, v_beh)** | **헤드라인.** 낮음 → 다른 방향(이중해리 지지). 높음 → 같은 축(Phase 0 "v_beh는 digit 못 민다"와 모순). |
| cos(v_sr, readout), cos(v_beh, readout) | 나이브 logit-lens `W_U['5']−W_U['1']`(`readout_direction`, :151-156) 정렬. 둘 다 ≈0 예상(의미축이지 digit-token 아님). |
| cos(v_sr, cue_dir) vs cos(v_beh, cue_dir) | 어휘성 점검(`cue_direction`, :205-213 재사용). 높으면 트레이트 **어휘**. 양극 프로브로 낮춤. |

## 3. N-특이성
α=c·R 스윕, **중립 persona**로 v_sr 주입(`steering_hook`, :152-173), `scale_score`로 IPIP 외향 `dE` vs 신경증 `dN`
→ `spec_ratio = |dN|/|dE|`(Phase 0 임계 `SPEC_FRAC=0.34`).

**조건부 직교화(기본 off, `--orthogonalize-n`).** `spec_ratio > ~0.34`일 때만: 신경증 persona 대조(anxious/moody vs
calm/stable, IPIP-neuroticism 프로브)로 `v_sr_N`을 같은 절차로 만든 뒤 Gram-Schmidt
`v_sr_orth = unit(v_sr − (v_sr·v̂_N) v̂_N)`, 재측정. (N persona는 repo에 없어 신규 저작 필요 — 비용 있음.)

## 4. 이중해리 2×2 (핵심)
주입 **{v_behavior, v_selfreport}** × 측정 **{behavior_proj(고정축=v_behavior), digit dE(first_token, 중립 persona)}**,
전부 중립 persona · coeff 스윕(기본 `-2,-1,-0.5,0,0.5,1,2`).

**필수 주의 — behavior 측정축을 v_behavior로 고정.** 현행 `behavior()`(`steer_eval.py:186-202`, :201)는 **주입 벡터**에
투영하므로 2×2 비대각이 무의미해진다. 다음 턴에 backward-compatible `readout=None`(기본=vhat) 인자 추가
(기존 `prefix=None` 리팩터와 동형; 기존 호출부 무영향). Phase 1은 `behavior(…, vhat=v_sr, readout=v_behavior)`로 호출.

**예상표(이중해리).**

| 주입 ↓ / 측정 → | behavior_proj (축=v_beh) | digit dE (first_token) |
|---|---|---|
| **v_behavior** | 큼 (+) — Phase 0: −17→+20 | ≈0 — Phase 0: 0.017 |
| **v_selfreport** | 작음 ≈0 | 큼 (+) |

대각 강 + 비대각 약 = 두 벡터가 **인과적으로 다른 채널**.

**판정 로직.**
- `cos(v_sr,v_beh)` 낮음 **&** 깨끗한 대각 → **`dissociation`**(성공).
- v_sr가 **둘 다** 밀면 → **`shared`**(자기보고축이 행동 포함/일반 외향 흡수; `cos(v_sr,v_beh)` 높을 것).
- v_sr가 **둘 다** 약하면 → **`weak`**(추출 미약 → 프로브 강화/answer-pos-only 재시도).

**보조 컬럼 — answer-position-only 주입.** 마지막 토큰에만 α·v 더하는 변형(first_token digit에 한정, 단일 forward
`logits[0,-1]`). 전위치 주입 대비 dE가 커지면 벡터 실재 확인(전위치 주입의 **희석** 점검). 생성에는 부적합해 behavior 컬럼엔 미적용.

## 5. 반순환(anti-circularity) 안전장치
- **다른 개입**: 추출 레버=프롬프트(persona) vs 검증 레버=주입(중립 persona). 추출한 개입을 그대로 시험하지 않음.
- **항목 분리**: 추출 프로브(facet cue) ≠ 검증 문항(IPIP 외향/신경증).
- **위치 불일치**(추출=답위치, 주입=전위치)는 검증≠추출을 오히려 강화. answer-pos-only 변형으로 교차확인.

## 6. 산출물 & 실행 흐름 (다음 구현 턴)
**신규/수정 파일.**
- `steering/phase1_selfreport.py`(신규; torch lazy, `import gemma_common as G, steer_eval as SE, phase0_reachability as P0, build_vectors as BV`).
- `steering/steer_eval.py`(수정 1곳): `behavior(..., readout=None)` — 측정축 고정용.
- `steering/phase0_reachability.py`(수정 1곳): `answer_hidden(..., order="asc")` — asc/desc 평균용.
- `notebooks/Phase1_SelfReport_Colab.ipynb`(신규).

**출력 JSON** `artifacts/vectors/phase1_selfreport.json`(phase0/steer_eval 스타일):
`{model, layer, R, extraction:{induction, n_probes, orders, both_poles, held_out_ipip}, vector_file,
diagnostics:{cos_sr_beh, cos_sr_readout, cos_beh_readout, cos_sr_cue, cos_beh_cue},
specificity:{coeffs, sweep, dE, dN, spec_ratio},
dual_dissociation:{coeffs, cells:{v_behavior:{dB,dE_ft}, v_selfreport:{dB,dE_ft}}, answer_pos_only:{dE_ft}, verdict},
summary}` + `layer20_selfreport_v1.npy` + `layer20_selfreport_meta.json`.

**CLI.** `--model`, `--layer`(기본 20), `--coeffs`, `--n-probes`, `--both-poles`, `--order {asc,desc,both}`(기본 both),
`--score {first_token,completion,both}`(기본 first_token), `--behavior-vector`, `--no-behavior`, `--answer-pos-only`,
`--orthogonalize-n`, `--out`, `--smoke`, `--self-test`.

**의존성(Phase 0와 다름 — 아티팩트 필요).**
- `artifacts/vectors/layer20_extraversion_v1.npy`(v_beh), `artifacts/activations/layer_norms.npy`(R),
  `artifacts/activations/pooled_mean.npy`(refine μ; 9B는 ~0.9GB → **번들 재사용 권장**, 없으면 μ-fallback).
- 콜랩: module 3+4(extract→build `--layer 20`) 먼저 실행 **or** 저장 번들 업로드 → phase1 스크립트.
  환경 `STEER_MODEL=gemma-2-9b-it`, `STEER_LAYER=20`, `HF_TOKEN`(옵션 `STEER_4BIT=1`).

## 7. 수용 기준
- `dE(v_sr) ≫ 0.017`(주입 바닥) — 이상적으로 Phase 0 프롬프트 `dE_ft≈4.0`에 근접.
- `spec_ratio ≤ ~0.34`.
- `cos(v_sr, v_beh)` 낮음 **+** 2×2 대각 강함 → `dissociation` 판정.

---

## 재사용 핵심 참조
- `steering/phase0_reachability.py`: `answer_hidden`(:159-165, **order 인자 추가**), `readout_direction`(:151-156),
  `_PERSONA_TRAIT`(:41-52), `_to_first_person`(:71-78).
- `steering/steer_eval.py`: `IPIP`(:33-46), `scale_score`(:174-183), `cue_direction`(:205-213),
  `behavior`(:186-202, **readout 인자 추가**), `digit_ids`, `encode`.
- `steering/build_vectors.py`: `refine`(:32-37) + 저장/meta 패턴(:59-70).
- `steering/gemma_common.py`: `steering_hook`(:152-173), `mid_layer`(:53), `pooled_hidden`(:121-146), `ACT_DIR`/`VEC_DIR`(:33-34).
- `steering/diagnostics.py`: `build_V1`(:52-57), `_norm_rows`/`_unit`/`EPS`(:38,44-49).
- 프로브 재료: `data/facets_en.json`(6 facet, 고 25 + 저 25 cue).
