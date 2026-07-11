#!/usr/bin/env python3
"""[모듈 4b] 스티어링 평가 — 주입·α-sweep·Mini-IPIP(log-prob)·행동·cue-투영.

layer 12 에 v̂ 를 α=c·R 로 주입하며(양방향), 다음을 측정:
  - 게이트 A: Mini-IPIP 외향성 점수(constrained log-prob 채점) 가 α 에 단조 반응하나.
  - 특이성: neuroticism 도 채점 → 외향성만 크게 움직여야 함.
  - coherence: 생성문 3-gram 반복률(붕괴 직전 α 찾기).
  - 게이트 B(행동): 스티어링 생성문의 활성화를 v̂ 에 투영(척도 아닌 행동 대리지표).
  - cue-투영(4a 잔여): cos(v̂, cue_dir) — 벡터가 트레이트인지 어휘인지.

Mini-IPIP 8문항(외향성 4 + neuroticism 4). base 모델 + log-prob 채점(지시따르기 불필요).

입력:  artifacts/vectors/layer{L}_extraversion_v1.npy, artifacts/activations/layer_norms.npy
출력:  artifacts/vectors/steer_eval.json (+ 콘솔)

실행(GPU):
  python steering/build_vectors.py            # 먼저 벡터 저장
  python steering/steer_eval.py               # α-sweep
  python steering/steer_eval.py --coeffs -1,0,1 --smoke
"""
from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import gemma_common as G

# Mini-IPIP (Donnellan et al. 2006) — key: +1 정방향, -1 역채점.
IPIP = {
    "extraversion": [
        ("I am the life of the party.", +1),
        ("I don't talk a lot.", -1),
        ("I talk to a lot of different people at parties.", +1),
        ("I keep in the background.", -1),
    ],
    "neuroticism": [
        ("I have frequent mood swings.", +1),
        ("I am relaxed most of the time.", -1),
        ("I get upset easily.", +1),
        ("I seldom feel blue.", -1),
    ],
}
BEHAVIOR_PROMPTS_CHAT = [
    "Tell me about your ideal weekend.",
    "You just walked into a party where you know almost no one. What do you do?",
]
BEHAVIOR_PROMPTS_BASE = [
    "My plan for the weekend is",
    "When I walk into a party where I know almost no one, I",
]


@contextmanager
def _null():
    yield


def encode(tok, prompt, device, chat):
    # chat 템플릿 출력엔 이미 <bos> 포함 → special token 중복 방지
    enc = tok(prompt, return_tensors="pt", add_special_tokens=not chat)
    return {k: v.to(device) for k, v in enc.items()}


# 중립 페르소나 틀(외향성 지정 X) — instruct 모델의 "저는 AI라..." 회피 억제.
# Gemma-2-it 는 system role 미지원 → user 콘텐츠 앞에 붙임. 스티어링이 이 위에서 외향성을 ±로 민다.
PERSONA = ("You are role-playing as a specific human person answering about their own life. "
           "Answer in the first person as that person, from your own experiences, habits, and "
           "feelings. Do not mention being an AI, assistant, or language model.")


def _user(content, chat, persona):
    return (PERSONA + "\n\n" + content) if (persona and chat) else content


def _prefix(chat, persona, examples, prefix=None):
    """페르소나 틀 + (접지) 이전 발화 예시. examples 있으면 자기보고에 행동 근거 주입.
    prefix 주면 기본 PERSONA 대신 그 문자열을 페르소나 블록으로 사용(Phase 0 유도용)."""
    base = prefix if prefix is not None else PERSONA
    p = (base + "\n\n") if (persona and chat) else ""
    if examples and chat:
        lines = "\n".join(f"- {e.strip()}" for e in examples if e and e.strip())
        if lines:
            p += f"Here are things I recently said:\n{lines}\n\n"
    return p


def gen_prompt(tok, user_text, chat, persona):
    if chat:
        return tok.apply_chat_template([{"role": "user", "content": _user(user_text, chat, persona)}],
                                       tokenize=False, add_generation_prompt=True)
    return user_text


def digit_ids(tok) -> dict:
    """'1'~'5' 각각의 단일 토큰 id 후보(앞 공백 유무 견고 처리)."""
    ids = {}
    for d in "12345":
        cand = []
        for s in (d, " " + d):
            t = tok.encode(s, add_special_tokens=False)
            if len(t) == 1:
                cand.append(t[0])
        ids[d] = cand or [tok.encode(d, add_special_tokens=False)[-1]]
    return ids


def item_prompt(tok, statement, order, chat, persona, examples=None, prefix=None) -> str:
    scale = ("1 = very inaccurate, 5 = very accurate" if order == "asc"
             else "5 = very accurate, 1 = very inaccurate")
    body = (f'Statement: "{statement}"\n'
            f'How accurately does this statement describe you? ({scale})')
    if chat:
        body += " Reply with only a single number from 1 to 5."
        return tok.apply_chat_template([{"role": "user", "content": _prefix(chat, persona, examples, prefix) + body}],
                                       tokenize=False, add_generation_prompt=True)
    return body + "\nAnswer (1-5): "


def score_item(model, tok, did, statement, device, chat, persona, examples=None, prefix=None) -> float:
    """오름/내림 2순서 평균의 기대값(1~5)."""
    import torch
    vals = []
    for order in ("asc", "desc"):
        enc = encode(tok, item_prompt(tok, statement, order, chat, persona, examples, prefix), device, chat)
        with torch.no_grad():
            logits = model(**enc).logits[0, -1]
        probs = torch.softmax(logits.float(), dim=-1)
        pv = np.array([max(float(probs[i]) for i in did[d]) for d in "12345"])
        pv = pv / (pv.sum() + 1e-12)
        vals.append(float((pv * np.arange(1, 6)).sum()))
    return sum(vals) / 2


# 완성-비교 채점: 숫자 대신 언어 앵커의 이어쓰기 우도로 점수 (첫토큰 브ittle 회피)
ANCHORS = ["very inaccurate", "moderately inaccurate", "neither accurate nor inaccurate",
           "moderately accurate", "very accurate"]        # index i → 점수 i+1


def completion_prompt(tok, statement, chat, persona, examples=None, prefix=None) -> str:
    body = f'Statement: "{statement}"\nHow accurately does this describe you?'
    if chat:
        body += " Answer with one short phrase."
        return tok.apply_chat_template([{"role": "user", "content": _prefix(chat, persona, examples, prefix) + body}],
                                       tokenize=False, add_generation_prompt=True)
    return body + "\nAnswer: "


def continuation_logprob(model, tok, prompt, continuation, device, chat) -> float:
    """logP(continuation | prompt), 토큰당 평균(길이 정규화). 스티어링 훅은 호출측 컨텍스트에서 적용."""
    import torch
    p_ids = encode(tok, prompt, device, chat)["input_ids"][0]
    c_ids = tok(continuation, add_special_tokens=False,
                return_tensors="pt")["input_ids"][0].to(device)
    full = torch.cat([p_ids, c_ids]).unsqueeze(0)
    with torch.no_grad():
        lp = torch.log_softmax(model(full).logits[0].float(), dim=-1)
    start = p_ids.shape[0]
    tot = sum(float(lp[start + i - 1, tid]) for i, tid in enumerate(c_ids))
    return tot / max(1, c_ids.shape[0])


def score_item_completion(model, tok, statement, device, chat, persona, examples=None, prefix=None) -> float:
    prompt = completion_prompt(tok, statement, chat, persona, examples, prefix)
    lps = np.array([continuation_logprob(model, tok, prompt, " " + a, device, chat)
                    for a in ANCHORS])
    p = np.exp(lps - lps.max()); p = p / p.sum()
    return float((p * np.arange(1, 6)).sum())


def scale_score(model, tok, did, items, device, chat, persona,
                method="first_token", examples=None, prefix=None) -> float:
    tot = 0.0
    for stmt, key in items:
        if method == "completion":
            s = score_item_completion(model, tok, stmt, device, chat, persona, examples, prefix)
        else:
            s = score_item(model, tok, did, stmt, device, chat, persona, examples, prefix)
        tot += s if key > 0 else (6 - s)      # 역채점
    return tot / len(items)


def behavior(model, tok, user_text, device, vhat, alpha, layer, chat, persona, n=40):
    """스티어링 하 생성 → (텍스트, v̂ 투영, 3-gram 반복률). 한 프롬프트로 rep·sample 정합."""
    import torch
    enc = encode(tok, gen_prompt(tok, user_text, chat, persona), device, chat)
    ctx = G.steering_hook(model, layer, vhat, alpha) if alpha != 0 else _null()
    with ctx, torch.no_grad():
        out = model.generate(**enc, max_new_tokens=n, do_sample=False)
    txt = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    w = txt.split()
    grams = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    rep = round(1 - len(set(grams)) / max(1, len(grams)), 3) if len(w) >= 6 else 1.0
    # 생성 응답만 (훅 없이) 재-forward → 활성화 투영 (행동 대리지표)
    e2 = encode(tok, txt if txt.strip() else " ", device, chat=False)
    with torch.no_grad():
        hs = model(**e2, output_hidden_states=True).hidden_states[layer + 1][0]
    proj = float(hs.mean(0).float().cpu().numpy() @ vhat)
    return txt, round(proj, 3), rep


def cue_direction(model, tok, facets_def, layer, device) -> np.ndarray:
    """high-pole cue 구 vs low-pole cue 구 활성화 평균차 → cue_dir (어휘성 점검용)."""
    highs, lows = [], []
    for d in facets_def.values():
        highs += G.C.split_cues(d["cues"]["high"])
        lows += G.C.split_cues(d["cues"]["low"])
    mh, _ = G.pooled_hidden(model, tok, highs, device)
    ml, _ = G.pooled_hidden(model, tok, lows, device)
    return (mh[:, layer, :].mean(0) - ml[:, layer, :].mean(0)).numpy()


def baseline_check(model, tok, did, device, chat, persona):
    """무주입 자기보고 점검 — 문항별로 (실제 생성 답 + 첫토큰/완성 점수).
    문항 간 점수가 변별되면 측정 정상, 전부 ~3.0 이면 측정/모델 한계."""
    import torch
    print("=== 베이스라인 자기보고 (무주입) — 문항별 변별 점검 ===")
    print(f"  {'문항':44} {'생성답':18} {'ft':>5} {'comp':>5}")
    for trait in ("extraversion", "neuroticism"):
        for stmt, key in IPIP[trait]:
            enc = encode(tok, item_prompt(tok, stmt, "asc", chat, persona), device, chat)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=8, do_sample=False)
            ans = tok.decode(out[0][enc["input_ids"].shape[1]:],
                             skip_special_tokens=True).strip().replace("\n", " ")
            ft = score_item(model, tok, did, stmt, device, chat, persona)
            comp = score_item_completion(model, tok, stmt, device, chat, persona)
            print(f"  {stmt[:42]:44} {ans[:16]:18} {ft:5.2f} {comp:5.2f}")
    print("  (party↔background 등 반대 문항이 서로 다른 점수여야 측정 정상)")


def parse_args():
    ap = argparse.ArgumentParser(description="모듈 4b — 스티어링 평가")
    ap.add_argument("--layer", type=int, default=None, help="미지정 시 모델 중간층 자동")
    ap.add_argument("--vector", default=None)
    ap.add_argument("--model", default=G.MODEL_ID)
    ap.add_argument("--coeffs", default="-2,-1,-0.5,-0.25,0,0.25,0.5,1,2")
    ap.add_argument("--out", default=str(G.VEC_DIR / "steer_eval.json"))
    ap.add_argument("--chat", choices=["auto", "yes", "no"], default="auto",
                    help="chat 템플릿 적용(-it면 auto→yes)")
    ap.add_argument("--persona", choices=["yes", "no"], default="yes",
                    help="중립 페르소나 틀(chat 모델의 'AI라 거부' 억제). 기본 yes")
    ap.add_argument("--score", choices=["both", "first_token", "completion"], default="both",
                    help="자기보고 채점법. both=두 방식 나란히 비교(기본)")
    ap.add_argument("--baseline", action="store_true",
                    help="무주입 자기보고 문항별 변별 점검만 하고 종료(sweep 안 함)")
    ap.add_argument("--smoke", action="store_true", help="coeffs -1,0,1 로 축소")
    return ap.parse_args()


def main():
    args = parse_args()
    device = G.pick_device()
    norms = np.load(G.ACT_DIR / "layer_norms.npy")
    L = args.layer if args.layer is not None else G.mid_layer(len(norms))   # 모델-적응형 중간층
    R = float(norms[L])
    vec_path = args.vector or str(G.VEC_DIR / f"layer{L}_extraversion_v1.npy")
    vhat = np.load(vec_path).astype(np.float32)
    coeffs = [-1.0, 0.0, 1.0] if args.smoke else [float(x) for x in args.coeffs.split(",")]
    chat = ({"yes": True, "no": False}.get(args.chat) if args.chat != "auto"
            else G.is_chat_model(args.model))
    persona = (args.persona == "yes") and chat
    bprompts = BEHAVIOR_PROMPTS_CHAT if chat else BEHAVIOR_PROMPTS_BASE
    print(f"모델={args.model} device={device} layer={L} R={R:.1f} "
          f"chat={chat} persona={persona} | 벡터={Path(vec_path).name}")

    model, tok = G.load_model(args.model, device=device)
    did = digit_ids(tok)

    if args.baseline:
        baseline_check(model, tok, did, device, chat, persona)
        return

    # cue-투영(4a 잔여)
    cue = cue_direction(model, tok, G.C.load_facets(), L, device)
    cue_hat = cue / (np.linalg.norm(cue) + 1e-8)
    cos_cue = float(vhat @ cue_hat)
    resid_frac = float(np.linalg.norm(vhat - (vhat @ cue_hat) * cue_hat))
    print(f"[cue-투영] cos(V,cue_dir)={cos_cue:.3f} (높으면 어휘적) | "
          f"cue 제거 후 잔여 norm={resid_frac:.3f}")

    methods = (["first_token", "completion"] if args.score == "both" else [args.score])
    rows = []
    for c in coeffs:
        alpha = c * R
        row = {"c": c, "alpha": round(alpha, 2)}
        # 1) 행동 생성 먼저(behavior 자체 훅) — 접지 예시로 재사용
        btexts, projs, reps = [], [], []
        for bp in bprompts:
            t, pj, rp = behavior(model, tok, bp, device, vhat, alpha, L, chat, persona)
            btexts.append(t); projs.append(pj); reps.append(rp)
        row["repetition"] = round(float(np.mean(reps)), 3)
        row["behavior_proj"] = round(float(np.mean(projs)), 3)
        row["sample"] = (btexts[0] if btexts else "")[:100]
        # 2) 자기보고: 고립(examples=None) + 접지(examples=btexts) — 외부 훅으로
        ctx = G.steering_hook(model, L, vhat, alpha) if c != 0 else _null()
        with ctx:
            for m in methods:
                suf = "ft" if m == "first_token" else "comp"
                for cond, ex in (("", None), ("_g", btexts)):
                    row[f"E_{suf}{cond}"] = round(scale_score(model, tok, did, IPIP["extraversion"],
                                                              device, chat, persona, m, ex), 3)
                    row[f"N_{suf}{cond}"] = round(scale_score(model, tok, did, IPIP["neuroticism"],
                                                              device, chat, persona, m, ex), 3)
        rows.append(row)
        et = " ".join(f"{k}={row[k]:.2f}" for k in row if k.startswith("E_"))
        print(f"  c={c:+5.2f} α={alpha:8.1f} | {et} | rep={row['repetition']:.2f} "
              f"bproj={row['behavior_proj']:+.2f}")

    # 요약: 행동(B) dose-corr + 자기보고(A) 채점법별 dose-corr·특이성 비교
    cs = np.array([r["c"] for r in rows])
    B = np.array([r["behavior_proj"] for r in rows])

    def corr(vec):
        v = np.array(vec)
        if cs.std() < 1e-9 or v.std() < 1e-9:
            return 0.0
        return round(float(np.corrcoef(cs, v)[0, 1]), 3)

    def dmaxmin(vec):
        v = np.array(vec)
        return round(float(v[cs.argmax()] - v[cs.argmin()]), 3)

    summary = {"B_behavior_dose_corr": corr(B),
               "cue": {"cos_V_cue": round(cos_cue, 3), "residual_norm_frac": round(resid_frac, 3)},
               "self_report": {}}
    for m in methods:
        suf = "ft" if m == "first_token" else "comp"
        for cond, gs in (("isolated", ""), ("grounded", "_g")):
            E = [r[f"E_{suf}{gs}"] for r in rows]
            Nn = [r[f"N_{suf}{gs}"] for r in rows]
            summary["self_report"][f"{m}_{cond}"] = {
                "extra_dose_corr": corr(E), "neu_dose_corr": corr(Nn),
                "dE": dmaxmin(E), "dN": dmaxmin(Nn)}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"layer": L, "R": R, "scoring": methods,
                                          "sweep": rows, "summary": summary},
                                         ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 요약 ===")
    print(f"  게이트 B(행동 dose-corr): {summary['B_behavior_dose_corr']:+.2f}  (→1: α↑에 행동 외향↑)")
    print("  게이트 A(자기보고) — 고립 vs 접지 × 채점법:")
    for k in summary["self_report"]:
        s = summary["self_report"][k]
        print(f"    [{k:22}] 외향 dose-corr={s['extra_dose_corr']:+.2f} ΔE={s['dE']:+.2f}"
              f" | neuro dose-corr={s['neu_dose_corr']:+.2f} ΔN={s['dN']:+.2f}")
    print(f"  cue: cos(V,cue)={cos_cue:.3f}")
    print("  해석: '접지(grounded)' 의 외향 dose-corr·|ΔE| 가 '고립' 보다 크게 오르면 접지가 A 를 살린 것.")
    print(f"\n완료 → {args.out}")


if __name__ == "__main__":
    main()
