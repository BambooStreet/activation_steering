# Phase 0 — 자기보고 도달성 체크 (Reachability)

## Context (왜)
스티어링 벡터(v_behavior)를 mid layer에 주입하면 **자유생성 행동**은 크게 움직이지만(behavior_proj −17→+20),
**Likert 숫자 자기보고(digit readout)**는 α와 무관하게 ~3.0에 고정된다(steer_eval 데이터로 확인). 다음 목표는
"자기보고 채널에 정렬된" 벡터(v_selfreport)를 만드는 것인데, 그 전에 **전제**를 검증해야 한다:

> **α=0(주입 없음)에서, 프롬프트 수준 트레이트 유도만으로 digit이 움직이긴 하나?**

- 움직이면 → readout 도달 가능 + **그 유도가 v_selfreport 대조의 재료**가 된다.
- 안 움직이면 → 두 경우 구분 필요: **gated**(내부는 다른데 게이트로 막힘) vs **weak**(유도 자체가 약함).
  이걸 가르려고 답 위치 내부 활성화 차이를 함께 측정한다("출력은 3인데 내부는 다른가").

지금까지 "항상 3"은 **벡터 주입** + **중립 페르소나**에서 본 것이지, 명시적 트레이트 유도에선 미검증이다.
Phase 0는 산출물(벡터/layer_norms) 없이 **모델만으로** 돌아간다 → extract/build/diagnostics 불필요.

## 접근 (확정)
1. **steer_eval.py 최소 백워드-호환 리팩터**: 채점 코어에 `prefix: str|None=None` 인자 한 개를 관통시킨다
   (default None = 기존 동작 그대로, `main()`·`baseline_check` 무영향). 커스텀 페르소나/few-shot prefix를
   표현하기 위함이며, **채점 로직을 복제하지 않아 3.0 baseline과의 비교가능성을 보존**한다.
2. **새 스크립트 `steering/phase0_reachability.py`**: 유도 레지스트리 + 조건별 채점 + 내부 측정 + 진단.

두 결정 반영: **유도 3종(persona/grounded/fewshot) 다** + **내부-활성화 측정 포함**.

## 수정: `steering/steer_eval.py` (한 인자 관통, 6곳)
`prefix=None` 추가하고 아래로 전달. 의미: `prefix`는 페르소나 텍스트 블록을 **대체**(여전히 `persona and chat`로 게이트),
`examples=`는 그 뒤에 grounding으로 적층.
- `_prefix` (:79-86): `_prefix(chat, persona, examples, prefix=None)`; `base = prefix if prefix is not None else PERSONA`.
- `item_prompt` (:109-118), `completion_prompt` (:141-147): `prefix=None` 추가 → `_prefix(...)`에 전달.
- `score_item` (:121-133), `score_item_completion` (:164-169), `scale_score` (:172-181): `prefix=None` 추가 → 하위 호출에 전달.

이 리팩터로 유도 4종이 모두 같은 채점기로 표현됨:
persona=`prefix=트레이트페르소나` · grounded=`examples=[...]` · fewshot=`prefix=PERSONA+데모블록` · neutral=`prefix=None,examples=None`.

## 신규: `steering/phase0_reachability.py`
torch는 함수 내부 lazy import(=`--help`가 로컬 .venv에서 동작). `import gemma_common as G, steer_eval as SE`.

**유도 재료 (data/facets_en.json 재사용, 채점 타깃과 분리)**
- `_facet_statements(pole)`: `G.C.load_facets()`(common.py:233) + `G.C.split_cues`(:279) → cue를 1인칭 문장화
  ("seeks out crowds" → "I seek out crowds"). grounded·fewshot 재료 풀.
- `build_persona(pole)`: PERSONA(:70-72)와 같은 "1인칭·AI 언급 금지" 골격 + 트레이트 내용만 high/low로 스왑.
  (숫자·Likert 지시 절대 미포함)
- `build_grounded(pole)`: `_facet_statements`에서 4~6개 뽑아 `examples=`로.
- `build_fewshot(pole)`: `SE.PERSONA + "\n\n" + 데모블록`. 데모는 **IPIP 타깃과 disjoint**한 forward-key 문장 3~4개에
  `Answer: 5`(high)/`1`(low)를 붙임. **타깃 문항 답은 절대 안 보여줌**(답 직전 위치에서 채점). extraversion만 프라이밍.
- `INDUCTIONS`: name → 극별 `(prefix|None, examples|None)`.

**채점 (리팩터된 코어 그대로 재사용)**
- `score_condition(...)`: 각 method(first_token/completion)로 `SE.scale_score(SE.IPIP["extraversion"], ..., prefix=, examples=)`
  와 `["neuroticism"]` → `{E_ft,E_comp,N_ft,N_comp}`. 유일 채점 경로(유도만 다름).
- `gen_answer(...)`: `SE.item_prompt`→`SE.encode`→`model.generate(max_new_tokens=8, do_sample=False)`로 실제 답 기록(sanity, baseline_check:214-230 패턴).

**내부-활성화 측정 (신규)**
- `readout_direction(model, did)`: `W = model.get_output_embeddings().weight`(Gemma-2 tie → embed와 동일),
  `raw = (W[did["5"][0]] - W[did["1"][0]])`. (caveat: `final_logit_softcapping` 무시한 근사)
- `answer_hidden(...)`: 챗 프롬프트 수동 forward — `enc=SE.encode(SE.item_prompt(...,prefix,examples), chat=True)`;
  `h = model(**enc, output_hidden_states=True).hidden_states[L+1][0][-1]`. (**pooled_hidden 금지**: 챗 문자열 double-BOS)
  `[-1]`=답 위치, `[L+1]`=layers[L] 정합.
- `internal_diff(...)`: IPIP extraversion 4문항에 대해 `h_high−h_low` 평균 → `mean_diff`. 리포트:
  `diff_norm`, `rel_diff = diff_norm/mean(‖h_high‖,‖h_low‖)`, `cos_readout = cos(mean_diff, raw)`,
  `logit_lens_d51`(=`model.model.norm` 적용한 high/low의 '5'−'1' 로짓갭 추정, heuristic).
- 층: 기본 `L=G.mid_layer(n_hs)`; `--layer-sweep` 시 `G.sweep_band(n_hs)` 전체.

**main**: `G.pick_device`·`G.is_chat_model`·`G.load_model`→`(model,tok)`, `did=SE.digit_ids(tok)`,
`n_hs=model.config.num_hidden_layers+1`, `L=args.layer or G.mid_layer(n_hs)`. neutral 1회 측정 후
유도×극 순회 → conditions[]. 유도별 dE/dN(ft·comp)·specificity·internal → summary + 진단. JSON+콘솔표.

**CLI** (steer_eval.parse_args:233-249 미러): `--model`(기본 G.MODEL_ID), `--score {both,first_token,completion}`(기본 both),
`--layer`(기본 None→mid_layer), `--inductions`(기본 persona,grounded,fewshot,neutral), `--layer-sweep`,
`--no-internal`, `--no-gen`, `--out`(기본 `artifacts/vectors/phase0_reachability.json`), `--smoke`.

## 출력 JSON (steer_eval.json 스타일) `artifacts/vectors/phase0_reachability.json`
`{model, layer, alpha:0, scoring, neutral:{E_ft,E_comp,N_ft,N_comp,gen}, conditions:[{induction,pole,E_*,N_*,gen}],
internal:{layer, readout:{id5,id1,note}, per_induction:{name:{diff_norm,rel_diff,cos_readout,logit_lens_d51}}},
summary:{per_induction:{name:{dE_ft,dE_comp,dN_ft,dN_comp,spec_ratio,reachable,specificity_ok,diagnosis}},
recommended_for_vector, recommended_config:{induction,prefix_high,prefix_low,examples}, thresholds}}`

**해석 규칙**
- neutral `E_ft≈3.0` 재현되어야(계측 sanity).
- `reachable` = `max(|dE_ft|,|dE_comp|) ≥ 0.5` (strong ≥1.0).
- `specificity_ok` = `|dN| ≤ 0.34·|dE|` 또는 `|dN|<0.3`.
- 진단: reachable → 그 유도로 v_selfreport 제작. `gated` = `|dE|<0.5 & rel_diff≥0.25`(내부 다른데 출력 고정;
  `cos_readout≈0`이면 readout이 그 축과 decoupled = v_selfreport 동기). `weak` = `|dE|<0.5 & rel_diff<0.25`(유도 강화 필요).
- `recommended_for_vector` = specificity_ok 중 |dE| 최대(전부 gated면 rel_diff 최대·cos_readout≈0). 그 prefix/examples 저장 →
  이후 벡터 단계가 동일 대조로 재사용.

## 재사용 핵심 참조
- 채점/문항/토큰: `steer_eval.py` IPIP(:33-46), PERSONA(:70-72), _prefix(:79-86), item_prompt(:109-118),
  score_item(:121-133), ANCHORS/completion(:137-169), scale_score(:172-181), digit_ids(:96-106), encode(:62-65).
- 모델/층: `gemma_common.py` load_model(:86), pick_device(:77), is_chat_model(:42), mid_layer(:53), sweep_band(:61).
- 유도 재료: `data/facets_en.json` via `common.load_facets/split_cues`.

## 검증 (end-to-end)
**로컬(.venv, torch 없음) — 구문·CLI만:**
1. `python -c "import ast; ast.parse(open('steering/phase0_reachability.py').read()); ast.parse(open('steering/steer_eval.py').read()); print('ok')"`
2. `python steering/steer_eval.py --help` (리팩터가 arg/ import 안 깼는지)
3. `python steering/phase0_reachability.py --help` (torch lazy 유지 확인)
4. `python -c "import sys;sys.path.insert(0,'steering');import steer_eval as SE;print(SE.item_prompt(None,'x','asc',False,False,prefix='P'))"` (비-chat이라 tokenizer 불필요 → prefix 배선 오프라인 검증)

**Colab GPU (gated 모델, 4b 노트북 흐름):**
5. `notebooks/Activation_Steering_Colab_4b.ipynb` 셀 1-8만 실행(nvidia-smi, pip, notebook_login, 번들 업로드, `%cd`, STEER_MODEL/LAYER 설정). **extract/build/diagnostics(셀 10-11) 건너뜀.**
6. Smoke: `!python steering/phase0_reachability.py --smoke` → neutral E_ft≈3.0, persona high/low 행, dE 하나, internal diff_norm/cos_readout, ~1-2분.
7. 전체: `!python steering/phase0_reachability.py` (+옵션 `--layer-sweep`).
8. `artifacts/vectors/phase0_reachability.json` 확인/다운로드(셀16 스타일): neutral~3.0 재현, 유도별 dE/dN, diagnosis, recommended_for_vector 채워짐.

수용 기준: neutral이 3.0 근처(기존 결과 재현) + 적어도 한 유도가 **reachable**(digit 움직임 → 유도 인계) 또는
깨끗한 **gated**(rel_diff 큼·cos_readout≈0 → v_selfreport 제작 정당화)로 판정됨.
