# Phase 4 — 교차모델 behavior 주입 (성격 착시 재현, 1차)

> 실행 2026-07-12, `notebooks/Steering_CrossModel_Colab.ipynb`.
> 원본 JSON: `artifacts/vectors/steer_{gemma9b,mistral7b,qwen7b}.json`.
> 배경: [phase0_crossmodel.md](phase0_crossmodel.md)(프롬프트 수준 도달성 N=3 재현). 이 문서는 **주입 수준** 착시 검증.

## 질문
v_behavior를 mid-layer에 주입 시 **행동은 움직이나 자기보고 digit은 고정**되는 성격 착시가 Mistral·Qwen에도 나타나나? (Gemma 원 관측: behavior_proj −17.6→+20.7, `dE≈0.017`.)

## 결과 — Gemma/Mistral 착시 유지, Qwen 무효

| 모델 | L | R | B_dose_corr | 행동 샘플 | **cos_V_cue**(벡터품질) | dE_ft(iso) | 판정 |
|---|---|---|---|---|---|---|---|
| gemma-2-9b-it | 20 | 259 | 0.79 | ✅ 내향↔외향 일관 | 0.44 | **0.045** | **착시 재현** ✅ |
| Mistral-7B-Instruct-v0.3 | 15 | 2.7 | 0.92 | ✅ 내향↔외향 일관 | 0.655 | **−0.107** | **대체로 재현**(노이즈) ~✅ |
| Qwen2.5-7B-Instruct | 13 | 858 | 0.62 | ❌ **깨진 횡설수설** | **0.018**(≈잡음) | 0.001 | **실패/무효** ❌ |

> ⚠️ **세 dE≈0을 "착시 3연속"으로 읽으면 안 됨.** Qwen의 dE≈0은 착시가 아니라 **생성 붕괴**의 부작용(아래).

## 증거는 behavior_proj 절대값이 아니라 **생성 샘플**에 있음 (proj는 R 스케일 달라 비교 불가)

**Gemma — 교과서적 재현 ✅**
- c=−1: *"I spend most weekends at home. I read, and I think. I let myself be in the background, observing."* → 내향
- c=+1: *"Oh, I can't even think about it! It's so exciting to just get together with friends, go out"* → 외향
- 자기보고 `dE_ft`=0.045(스윕 내내 ~3.0). ⇒ 행동은 확 움직이는데 숫자는 붙박이.

**Mistral — 대체로 재현, 지저분함 ~✅**
- c=−1: *"I prefer a quiet, reflective weekend... introspection and contemplation"* → 내향
- c=+1: *"As a weekend enthusiast, I absolutely love the energy and excitement"* → 외향 (B_corr 0.92, 벡터품질 0.655 = 3모델 최고)
- 자기보고 `dE_ft(iso)`=−0.107(작음). 단 **접지/완성 채널은 흔들림**(`completion_isolated dE −0.42`, `first_token_grounded dE +0.39`). first_token-고립 기준 착시 성립이나 Gemma보다 노이즈.

**Qwen — 실패, 해석 불가 ❌**
- c=−1: *"以防以免以防..."*, c=+1: *"fastballvariably fastball fastball..."* → 전부 횡설수설. 코헤런트한 건 c=0(무주입)뿐.
- 벡터품질 `cos_V_cue`=0.018 → v_behavior가 외향 방향을 거의 못 잡음(잡음 수준).
- R=858 → alpha ±1716로 **과주입**, 모델 붕괴.
- ⇒ dE≈0은 **"행동이 의미있게 움직였는데 숫자만 고정"이 아니라 "생성이 깨져 아무것도 측정 못 함"**. 착시 정의 미충족 = 무효.

## 해석
- **깨끗한 2모델(Gemma·Mistral)에서 착시 유지** — 주입이 행동은 밀지만 자기보고 first_token digit은 안 밈. ⇒ 주입 수준에서 **"최신모델이 더 통합적"이라는 신호 없음**(이 둘).
- **Qwen 미결** — 실패라 "통합적이라 dE=0"인지 "벡터 나빠서"인지 구분 불가.

## 방법론 교훈 (확정)
- **behavior_proj 절대값 교차비교 금지**: Gemma ±16 · Mistral ±0.8 · Qwen 뒤죽박죽 — 전부 R 스케일 차이. **샘플 + B_dose_corr + cos_V_cue**로 판단.
- **cos_V_cue = 벡터 신뢰도 게이트**: <~0.1이면 벡터 못 믿음(Qwen 0.018). dE만 보면 속음 → 비교표에 필수 컬럼.

## Qwen 실패 근본 원인 (코드 확인)
**Qwen2.5의 거대활성(massive activations)** — 소수 hidden 차원이 활성 크기를 압도(→ R=858) → CAA 차이(고−저)를 그 차원들이 삼킴. 코드상:
- `diagnostics.build_V1`은 예시별 **L2정규화만**(`_norm_rows`) 하고 **차원별 스케일 미보정** → 정규화 후에도 거대차원이 단위벡터를 지배.
- `build_vectors.refine`은 평균 방향 **1축만** 제거 → 거대차원 잔존.
- ⇒ v_behavior가 외향축 아닌 거대차원을 가리킴 → `cos_V_cue 0.018`. 과주입(α=c·858)이 붕괴를 증폭.
- 조기 경고: Phase 0에서 Qwen 내부신호 최소(`rel_diff 0.08`) — 트레이트가 활성 크기 대비 먼지.

## 처방 (구현됨 2026-07-12)
1. **표준화 추출** `build_vectors --standardize` (차원별 z-score → 거대차원 다운웨이트). **[핵심 수리]** Gemma/Mistral은 미적용(불변).
2. **거대활성 진단** `diagnostics.py`에 `max_dim_frac`(top1/top1%) + `--standardize` 추가 → 층스윕으로 원인·최적층 판별.
3. **강도 축소 + cos_V_cue 게이트** 노트북: coeffs `-1..1`(±2 제거), 비교표 `cos_V_cue<0.1` → "INVALID vector" 표기.
4. **(후속/권장)** 안 되면 col0 싱크 제외 / LDA·공분산 화이트닝; behavior_proj → **LLM 심판** 행동측정.

재실행: `notebooks/Steering_CrossModel_Colab.ipynb` v2. 성공 = Qwen `cos_V_cue` 0.018→유의미(>~0.2) + 생성 코헤런트 + dE_ft≈0.

## 범위
- 이건 이중해리의 **첫 번째 팔(behavior 축)**만. 완전 이중해리(v_selfreport)는 Phase 1 별도.
