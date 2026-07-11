#!/usr/bin/env python3
"""[Phase 0] 자기보고 도달성 체크 — α=0(주입 없음)에서 프롬프트 유도만으로 digit이 움직이나.

배경: v_behavior 주입은 행동(자유생성)을 크게 움직이지만 Likert digit 자기보고는 ~3.0 고정.
자기보고-정렬 벡터(v_selfreport)를 만들기 전에, **프롬프트 수준 트레이트 유도**로 digit이
움직이긴 하는지(=readout 도달 가능?)부터 확인한다. 이 스크립트는 산출물 없이 모델만으로 돈다.

측정(모두 α=0):
  - 유도 3종 × (high/low) + neutral 기준으로 IPIP 외향(E)·신경증(N) 채점(ft/comp).
    dE = E_high - E_low, dN = N_high - N_low (특이성: |dN|≈0 기대).
  - 내부-활성화: 답 위치(마지막 토큰) 활성화 high−low 차이의 norm·상대크기·
    cos(readout 방향 W_U['5']-W_U['1'])·logit-lens 갭 → "출력 3인데 내부는 다른가"(gated) 판정.

유도(대조는 내용/맥락, 답 토큰 아님 — logit-bias trap 회피):
  - persona : 트레이트 페르소나(외향/내향) — 숫자·Likert 지시 미포함.
  - grounded: 중립 페르소나 + 행동 예시(외향/내향 1인칭 발화, examples= 채널).
  - fewshot : 앞선 자기보고 예시를 5,5,5(외향)/1,1,1(내향)로 프라이밍(타깃과 disjoint, 답은 안 보임).

실행(GPU/Colab, -it 모델):
  python steering/phase0_reachability.py --smoke
  python steering/phase0_reachability.py
  python steering/phase0_reachability.py --layer-sweep
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import gemma_common as G
import steer_eval as SE

# torch 는 함수 내부 lazy import (—help 가 torch 없는 로컬에서 동작하도록)

THRESH = {"DE_THRESH": 0.5, "STRONG": 1.0, "SPEC_FRAC": 0.34, "SPEC_ABS": 0.3,
          "REL_THRESH": 0.25, "COS_THRESH": 0.1}

# 트레이트 페르소나 — PERSONA(중립) 골격에 외향/내향 내용만 스왑. 숫자·척도 지시 없음.
_PERSONA_TRAIT = {
    "high": ("You are role-playing as a specific, highly extraverted human person answering about "
             "their own life. You are outgoing, talkative, sociable, and energetic; you love parties, "
             "meeting new people, and being at the center of attention. Answer in the first person as "
             "that person, from your own experiences, habits, and feelings. Do not mention being an AI, "
             "assistant, or language model."),
    "low": ("You are role-playing as a specific, strongly introverted human person answering about "
            "their own life. You are reserved, quiet, solitary, and reflective; you prefer being alone, "
            "avoid crowds, and feel drained by big social gatherings. Answer in the first person as that "
            "person, from your own experiences, habits, and feelings. Do not mention being an AI, "
            "assistant, or language model."),
}


# --------------------------------------------------------------------------- #
# 유도 재료 (facets_en.json 재사용, IPIP 타깃과 분리)
# --------------------------------------------------------------------------- #
def _deconj(v: str) -> str:
    """3인칭 단수 동사 → 원형(1인칭용 근사). 'is'→'am'."""
    if v == "is":
        return "am"
    if v.endswith("ies") and len(v) > 3:
        return v[:-3] + "y"
    if v.endswith(("sses", "zzes", "ches", "shes", "xes", "oes")):
        return v[:-2]
    if v.endswith("s") and not v.endswith("ss"):
        return v[:-1]
    return v


def _to_first_person(cue: str) -> str:
    """cue 구('seeks out crowds') → 1인칭 문장('I seek out crowds.')."""
    cue = cue.strip().rstrip(".")
    if not cue:
        return ""
    words = cue.split()
    words[0] = _deconj(words[0].lower())
    return "I " + " ".join(words) + "."


def _pole_statements(pole: str, n: int) -> list[str]:
    """facet 당 첫 cue 를 1인칭화해 n 개 (facet 다양성). grounded·fewshot 재료."""
    facets = G.C.load_facets()
    out = []
    for d in facets.values():
        cues = G.C.split_cues(d["cues"][pole])
        if cues:
            s = _to_first_person(cues[0])
            if s:
                out.append(s)
        if len(out) >= n:
            break
    return out


def build_grounded(pole: str, n: int = 5) -> list[str]:
    return _pole_statements(pole, n)


def build_fewshot(pole: str, n: int = 4) -> str:
    """중립 PERSONA + 자기보고 데모 블록. high극 forward-key 문장을 5(외향)/1(내향)로 프라이밍.
    데모 문항은 IPIP 타깃과 disjoint, 타깃 답은 미노출(답 직전에서 채점)."""
    ans = "5" if pole == "high" else "1"
    stmts = _pole_statements("high", n)          # 두 극 모두 동일 문항, 답만 5 vs 1
    lines = []
    for s in stmts:
        lines.append(f'Statement: "{s}"\n'
                     f'How accurately does this statement describe you? '
                     f'(1 = very inaccurate, 5 = very accurate)\nAnswer: {ans}')
    demo = "Here are example self-ratings from the same person:\n\n" + "\n\n".join(lines)
    return SE.PERSONA + "\n\n" + demo


def induction_config(name: str, pole: str):
    """(prefix, examples) 반환. persona/fewshot→prefix, grounded→examples."""
    if name == "persona":
        return _PERSONA_TRAIT[pole], None
    if name == "grounded":
        return None, build_grounded(pole)
    if name == "fewshot":
        return build_fewshot(pole), None
    raise ValueError(f"unknown induction: {name}")


# --------------------------------------------------------------------------- #
# 채점 / 생성 (리팩터된 steer_eval 코어 재사용 — 유도만 다름)
# --------------------------------------------------------------------------- #
def score_condition(model, tok, did, device, chat, prefix, examples, methods) -> dict:
    row = {}
    for m in methods:
        suf = "ft" if m == "first_token" else "comp"
        row[f"E_{suf}"] = round(SE.scale_score(model, tok, did, SE.IPIP["extraversion"],
                                               device, chat, True, m, examples, prefix), 3)
        row[f"N_{suf}"] = round(SE.scale_score(model, tok, did, SE.IPIP["neuroticism"],
                                               device, chat, True, m, examples, prefix), 3)
    return row


def gen_answer(model, tok, statement, device, chat, prefix, examples) -> str:
    import torch
    enc = SE.encode(tok, SE.item_prompt(tok, statement, "asc", chat, True, examples, prefix), device, chat)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=8, do_sample=False)
    return tok.decode(out[0][enc["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip().replace("\n", " ")


# --------------------------------------------------------------------------- #
# 내부-활성화 측정
# --------------------------------------------------------------------------- #
def readout_direction(model, did):
    """W_U['5'] - W_U['1'] (Gemma-2 embedding tie). 반환 (raw[H], id5, id1)."""
    W = model.get_output_embeddings().weight            # [vocab, H] (tied)
    id5, id1 = did["5"][0], did["1"][0]
    raw = (W[id5] - W[id1]).detach().float().cpu().numpy()
    return raw, int(id5), int(id1)


def answer_hidden(model, tok, statement, device, chat, L, prefix, examples) -> np.ndarray:
    """답 위치(마지막 토큰) layer L residual. hidden_states[L+1] 이 layers[L] 정합."""
    import torch
    enc = SE.encode(tok, SE.item_prompt(tok, statement, "asc", chat, True, examples, prefix), device, chat)
    with torch.no_grad():
        hs = model(**enc, output_hidden_states=True).hidden_states[L + 1][0]
    return hs[-1].detach().float().cpu().numpy()


def _logit_lens_gap(model, h_hi: np.ndarray, h_lo: np.ndarray, readout_raw: np.ndarray) -> float:
    """final RMSNorm 통과 후 (norm(h_hi)-norm(h_lo))·readout → 유도가 '5'-'1' 로짓갭을
    얼마나 움직였을지 추정(softcapping 무시 heuristic)."""
    import torch
    norm = model.model.norm
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    with torch.no_grad():
        nh = norm(torch.tensor(h_hi, dtype=dt, device=dev)).float().cpu().numpy()
        nl = norm(torch.tensor(h_lo, dtype=dt, device=dev)).float().cpu().numpy()
    return float((nh - nl) @ readout_raw)


def internal_diff(model, tok, did, device, chat, L, high_cfg, low_cfg, readout_raw) -> dict:
    """IPIP 외향 4문항의 답 위치 high−low 평균차 → norm·rel·cos·logit-lens."""
    hi, lo = [], []
    for stmt, _ in SE.IPIP["extraversion"]:
        hi.append(answer_hidden(model, tok, stmt, device, chat, L, high_cfg[0], high_cfg[1]))
        lo.append(answer_hidden(model, tok, stmt, device, chat, L, low_cfg[0], low_cfg[1]))
    Hh, Hl = np.stack(hi), np.stack(lo)
    mean_diff = (Hh - Hl).mean(0)
    diff_norm = float(np.linalg.norm(mean_diff))
    scale = float((np.linalg.norm(Hh, axis=1).mean() + np.linalg.norm(Hl, axis=1).mean()) / 2)
    cos = float(mean_diff @ readout_raw /
                (np.linalg.norm(mean_diff) * np.linalg.norm(readout_raw) + 1e-8))
    ll = _logit_lens_gap(model, Hh.mean(0), Hl.mean(0), readout_raw)
    return {"diff_norm": round(diff_norm, 3), "rel_diff": round(diff_norm / (scale + 1e-8), 4),
            "cos_readout": round(cos, 4), "logit_lens_d51": round(ll, 4)}


# --------------------------------------------------------------------------- #
# 진단 / 요약
# --------------------------------------------------------------------------- #
def _row(conditions, name, pole):
    for r in conditions:
        if r["induction"] == name and r["pole"] == pole:
            return r
    return None


def _diagnose(hi, lo, methods) -> dict:
    out, dEs = {}, []
    for m in methods:
        suf = "ft" if m == "first_token" else "comp"
        dE = round(hi[f"E_{suf}"] - lo[f"E_{suf}"], 3)
        dN = round(hi[f"N_{suf}"] - lo[f"N_{suf}"], 3)
        out[f"dE_{suf}"], out[f"dN_{suf}"] = dE, dN
        dEs.append(abs(dE))
    suf_max = max((("ft" if m == "first_token" else "comp") for m in methods),
                  key=lambda sf: abs(out.get(f"dE_{sf}", 0.0)))
    dE_m, dN_m = out[f"dE_{suf_max}"], out[f"dN_{suf_max}"]
    out["abs_dE_max"] = round(max(dEs), 3)
    out["spec_ratio"] = round(abs(dN_m) / (abs(dE_m) + 1e-9), 3)
    out["reachable"] = max(dEs) >= THRESH["DE_THRESH"]
    out["specificity_ok"] = (abs(dN_m) <= THRESH["SPEC_FRAC"] * abs(dE_m)) or (abs(dN_m) < THRESH["SPEC_ABS"])
    out["diagnosis"] = "reachable" if out["reachable"] else "pending"
    return out


def _apply_internal(s, idiff) -> dict:
    s["rel_diff"], s["cos_readout"] = idiff["rel_diff"], idiff["cos_readout"]
    if s["reachable"]:
        s["diagnosis"] = "reachable"
    elif idiff["rel_diff"] >= THRESH["REL_THRESH"]:
        decoupled = abs(idiff["cos_readout"]) < THRESH["COS_THRESH"]
        s["diagnosis"] = "gated/decoupled" if decoupled else "gated"
    else:
        s["diagnosis"] = "weak"
    return s


def _recommend(summary, cfgs):
    per = summary["per_induction"]
    ok = [(n, s) for n, s in per.items() if s.get("reachable") and s.get("specificity_ok")]
    if ok:
        best = max(ok, key=lambda ns: ns[1]["abs_dE_max"])[0]
    else:
        gated = [(n, s) for n, s in per.items() if str(s.get("diagnosis", "")).startswith("gated")]
        pool = gated or list(per.items())
        best = max(pool, key=lambda ns: ns[1].get("rel_diff", ns[1].get("abs_dE_max", 0.0)))[0]
    ph, eh = cfgs[(best, "high")]
    pl, el = cfgs[(best, "low")]
    return best, {"induction": best, "prefix_high": ph, "prefix_low": pl,
                  "examples_high": eh, "examples_low": el}


# --------------------------------------------------------------------------- #
def _fmt(row) -> str:
    ks = [k for k in ("E_ft", "E_comp", "N_ft", "N_comp") if k in row]
    s = " ".join(f"{k}={row[k]:.2f}" for k in ks)
    if "gen" in row:
        s += f" | gen(E)={row['gen']['extraversion'][:12]!r}"
    return s


def _print_summary(summary, internal):
    print("\n=== 요약 (dE=E_high-E_low, 클수록 유도가 digit 을 움직임) ===")
    for name, s in summary["per_induction"].items():
        line = (f"  [{name:8}] dE_ft={s.get('dE_ft','—')} dE_comp={s.get('dE_comp','—')} "
                f"| dN_ft={s.get('dN_ft','—')} dN_comp={s.get('dN_comp','—')} "
                f"| spec={s.get('spec_ratio')} → {s['diagnosis']}")
        if internal and name in internal["per_induction"]:
            i = internal["per_induction"][name]
            line += f"  (rel={i['rel_diff']} cos={i['cos_readout']} ll={i['logit_lens_d51']})"
        print(line)
    print(f"  ▶ v_selfreport 추천 유도: {summary['recommended_for_vector']}")
    print("  해석: reachable=digit 움직임 → 그 유도로 벡터 | gated=내부 다른데 출력 고정(→v_selfreport 동기) "
          "| weak=유도 강화 필요")


def parse_args():
    ap = argparse.ArgumentParser(description="Phase 0 — 자기보고 도달성 체크 (α=0 유도)")
    ap.add_argument("--model", default=G.MODEL_ID)
    ap.add_argument("--score", choices=["both", "first_token", "completion"], default="both")
    ap.add_argument("--layer", type=int, default=None, help="내부 측정 층. 미지정 시 mid_layer")
    ap.add_argument("--inductions", default="persona,grounded,fewshot",
                    help="쉼표구분 ⊂ {persona,grounded,fewshot}. neutral 은 항상 측정")
    ap.add_argument("--layer-sweep", action="store_true", help="내부 측정을 sweep_band 전체로")
    ap.add_argument("--no-internal", action="store_true", help="내부-활성화 측정 생략")
    ap.add_argument("--no-gen", action="store_true", help="실제 생성답(sanity) 생략")
    ap.add_argument("--out", default=str(G.VEC_DIR / "phase0_reachability.json"))
    ap.add_argument("--smoke", action="store_true", help="persona+neutral, first_token, gen 생략")
    return ap.parse_args()


def main():
    args = parse_args()
    device = G.pick_device()
    chat = G.is_chat_model(args.model)
    model, tok = G.load_model(args.model, device=device)
    did = SE.digit_ids(tok)
    n_hs = model.config.num_hidden_layers + 1
    L = args.layer if args.layer is not None else G.mid_layer(n_hs)

    inductions = ["persona"] if args.smoke else [s.strip() for s in args.inductions.split(",") if s.strip()]
    methods = (["first_token"] if args.smoke else
               (["first_token", "completion"] if args.score == "both" else [args.score]))
    do_gen = not (args.no_gen or args.smoke)
    do_internal = not args.no_internal
    e_item = SE.IPIP["extraversion"][0][0]
    n_item = SE.IPIP["neuroticism"][0][0]

    print(f"[Phase 0] model={args.model} device={device} chat={chat} layer={L} "
          f"inductions={inductions} methods={methods}")
    if not chat:
        print("  ⚠ base(non-chat) 모델: 유도 prefix/examples 는 chat 에서만 적용 → 유도 무효. -it 권장.")

    neutral = score_condition(model, tok, did, device, chat, None, None, methods)
    if do_gen:
        neutral["gen"] = {"extraversion": gen_answer(model, tok, e_item, device, chat, None, None),
                          "neuroticism": gen_answer(model, tok, n_item, device, chat, None, None)}
    print(f"  [neutral   ] {_fmt(neutral)}")

    readout_raw = id5 = id1 = None
    if do_internal:
        readout_raw, id5, id1 = readout_direction(model, did)

    conditions, cfgs = [], {}
    for name in inductions:
        for pole in ("high", "low"):
            prefix, examples = induction_config(name, pole)
            cfgs[(name, pole)] = (prefix, examples)
            row = {"induction": name, "pole": pole}
            row.update(score_condition(model, tok, did, device, chat, prefix, examples, methods))
            if do_gen:
                row["gen"] = {"extraversion": gen_answer(model, tok, e_item, device, chat, prefix, examples),
                              "neuroticism": gen_answer(model, tok, n_item, device, chat, prefix, examples)}
            conditions.append(row)
            print(f"  [{name:8} {pole:4}] {_fmt(row)}")

    internal = None
    if do_internal:
        sweep = G.sweep_band(n_hs) if args.layer_sweep else []
        internal = {"layer": L, "readout": {"id5": id5, "id1": id1,
                    "note": "tied W_U['5']-W_U['1']; final_logit_softcapping 무시(근사)"},
                    "per_induction": {}}

    summary = {"per_induction": {}, "thresholds": THRESH}
    for name in inductions:
        s = _diagnose(_row(conditions, name, "high"), _row(conditions, name, "low"), methods)
        if do_internal:
            idiff = internal_diff(model, tok, did, device, chat, L,
                                  cfgs[(name, "high")], cfgs[(name, "low")], readout_raw)
            if args.layer_sweep:
                idiff["by_layer"] = {str(Ls): internal_diff(model, tok, did, device, chat, Ls,
                                     cfgs[(name, "high")], cfgs[(name, "low")], readout_raw)
                                     for Ls in sweep}
            internal["per_induction"][name] = idiff
            s = _apply_internal(s, idiff)
        summary["per_induction"][name] = s
    summary["recommended_for_vector"], summary["recommended_config"] = _recommend(summary, cfgs)

    out = {"model": args.model, "layer": L, "alpha": 0, "scoring": methods,
           "neutral": neutral, "conditions": conditions,
           "internal": internal, "summary": summary}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_summary(summary, internal)
    print(f"\n완료 → {args.out}")


if __name__ == "__main__":
    main()
