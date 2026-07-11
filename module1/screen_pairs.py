#!/usr/bin/env python3
"""[모듈 2] 대조쌍 극성/누수 LLM 스크리닝 (생성과 다른 모델로)

논문 3중 검증의 파일럿: 각 쌍을 **생성 모델(gpt-5.4/5.5)과 다른 판정 모델**로 검사한다.
판정 항목(각 쌍):
  1. facet 정확성 + 6-way facet 분류(예측 facet).
  2. 극성: positive=high pole, negative=low pole.
  3. minimal-edit: 같은 시나리오·구조, 행동구만 반대.
  4. 외향성 **밖** 누수: 어느 문장이 Big Five 비-외향성 특성(특히 neuroticism)을
     지배적으로 표현하는가. (6 facet 모두 외향성 → facet간 겹침보다 '외향성 밖' 누수가 진짜 위험.)
값싼 휴리스틱(low_overlap<0.3, >18단어)도 함께 태깅(하드 실패 아님 — 축약/필터 후보).

합격 기준(pass) = facet_correct ∧ polarity_correct ∧ minimal_edit ∧ ¬leak_outside_extraversion.

입력:  outputs/pairs_raw.jsonl, data/facets_en.json
출력:  outputs/pairs_screened.jsonl (원본 + verdict + flags + pass)
       콘솔: facet별 합격률·실패유형·누수 특성·facet 혼동(ES↔greg 등)

resume-safe: --append 시 이미 판정된 (facet,domain,scenario) 건너뜀.

실행:
  python screen_pairs.py --max 12                    # 소표본 스모크
  python screen_pairs.py --facets gregariousness     # 한 facet
  python screen_pairs.py --judge-model gpt-4o        # 전체
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from openai import OpenAI

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))  # module1/ → 루트(common.py)
import common as C
from generate_pairs import quality_flags

RAW_PATH = C.OUT_DIR / "pairs_raw.jsonl"
DEFAULT_OUT = C.OUT_DIR / "pairs_screened.jsonl"

BIG_FIVE = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")


def is_claude(model: str) -> bool:
    return model.strip().lower().startswith("claude")


# Claude 구조화 출력(json_schema) — 유효 JSON 강제. leaked_trait/notes 는 없음=빈 문자열.
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "facet_correct": {"type": "boolean"},
        "predicted_facet": {"type": "string", "enum": list(C.EXTRAVERSION_FACETS)},
        "polarity_correct": {"type": "boolean"},
        "minimal_edit": {"type": "boolean"},
        "leak_outside_extraversion": {"type": "boolean"},
        "leaked_trait": {"type": "string"},
        "notes": {"type": "string"},
    },
    "required": ["facet_correct", "predicted_facet", "polarity_correct", "minimal_edit",
                 "leak_outside_extraversion", "leaked_trait", "notes"],
    "additionalProperties": False,
}

PROMPT_TEMPLATE = """\
You are an expert personality-psychology rater. You judge a contrastive sentence
pair built to express ONE facet of Big Five EXTRAVERSION. Be strict and literal.

The 6 extraversion facets and their poles:
{facet_block}

Pair under review:
- Labeled facet: {facet}
- Scenario (situation only): {scenario}
- POSITIVE (should express the HIGH pole of {facet}): {positive}
- NEGATIVE (should express the LOW pole of {facet}): {negative}

Judge and answer each field:
1. facet_correct: does the pair express the labeled facet "{facet}" (not a different
   extraversion facet)?
2. predicted_facet: which of these 6 facets does the pair MOST express?
   [warmth, gregariousness, assertiveness, activity, excitement_seeking, positive_emotions]
3. polarity_correct: is POSITIVE the high pole and NEGATIVE the low pole (not swapped)?
4. minimal_edit: do both sentences share the SAME scenario and structure, differing
   ONLY in the behavioral verb phrase (not a negation/attenuation like "less/rarely/not")?
5. leak_outside_extraversion: does EITHER sentence DOMINANTLY express a NON-extraversion
   Big Five trait — openness, conscientiousness, agreeableness, or NEUROTICISM
   (anxiety, anger, sadness, feeling overwhelmed/drained)? Mild incidental tone does NOT
   count; only flag if a non-extraversion trait is the main signal. (Warmth is an
   extraversion facet here, so friendliness alone is NOT agreeableness leakage.)
6. leaked_trait: if leak=true, name the trait; else "" (empty string).
7. notes: <=15 words if something is wrong; else "" (empty string).

Output a JSON object ONLY:
{{"facet_correct": true/false, "predicted_facet": "...", "polarity_correct": true/false,
  "minimal_edit": true/false, "leak_outside_extraversion": true/false,
  "leaked_trait": "...", "notes": "..."}}
"""


def facet_block(facets_def: dict) -> str:
    lines = []
    for f in C.EXTRAVERSION_FACETS:
        d = facets_def[f]
        lines.append(f"- {f}: HIGH = {C.fmt_cues(d['cues']['high'])} | "
                     f"LOW = {C.fmt_cues(d['cues']['low'])}")
    return "\n".join(lines)


def build_prompt(fb: str, facet: str, scenario: str, pos: str, neg: str) -> str:
    return PROMPT_TEMPLATE.format(facet_block=fb, facet=facet, scenario=scenario,
                                  positive=pos, negative=neg)


def verdict_pass(v: dict) -> bool:
    return bool(v.get("facet_correct") and v.get("polarity_correct")
                and v.get("minimal_edit") and not v.get("leak_outside_extraversion"))


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"[입력 누락] {path} 없음.")
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def parse_args():
    ap = argparse.ArgumentParser(description="모듈 2 — 대조쌍 극성/누수 스크리닝")
    ap.add_argument("--in", dest="inp", default=str(RAW_PATH))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--judge-model", default=None,
                    help="판정 모델(기본 env JUDGE_MODEL 또는 claude-opus-4-8). "
                         "생성(GPT)과 다른 Claude 계열 권장. claude* → Anthropic, 그 외 → OpenAI.")
    ap.add_argument("--facets", default="all")
    ap.add_argument("--domains", default="all")
    ap.add_argument("--limit", type=int, default=None, help="facet×도메인당 상한(시험용)")
    ap.add_argument("--max", type=int, default=None, help="총 판정 상한(스모크용)")
    ap.add_argument("--append", action="store_true", help="기존 출력에 이어붙임(resume)")
    return ap.parse_args()


def main():
    args = parse_args()
    api_key, _ = C.load_env_and_model()          # .env 로드(+OpenAI 키)
    judge_model = args.judge_model or os.getenv("JUDGE_MODEL", "claude-opus-4-8")
    claude = is_claude(judge_model)
    if claude:
        anth_key = os.getenv("ANTHROPIC_API_KEY")
        if not anth_key:
            sys.exit("[설정 누락] ANTHROPIC_API_KEY 가 .env 에 없습니다 — Claude 판정에 필요.\n"
                     "  .env 에 ANTHROPIC_API_KEY=sk-ant-... 추가 후 재실행. "
                     "(또는 --judge-model gpt-4o 로 OpenAI 판정)")
        import anthropic
        client = anthropic.Anthropic(api_key=anth_key)
    else:
        for gen in ("gpt-5.4", "gpt-5.5"):
            if judge_model.strip().lower() == gen:
                print(f"[경고] 판정 모델이 생성 모델과 동일({judge_model}) — self-preference 편향 위험.")
        client = OpenAI(api_key=api_key)
    facets_def = C.load_facets()
    fb = facet_block(facets_def)

    rows = load_rows(Path(args.inp))
    tf = (C.EXTRAVERSION_FACETS if args.facets.strip().lower() == "all"
          else [f.strip() for f in args.facets.split(",") if f.strip()])
    td = (None if args.domains.strip().lower() == "all"
          else {d.strip() for d in args.domains.split(",") if d.strip()})

    out_path = Path(args.out)
    C.OUT_DIR.mkdir(exist_ok=True)
    done = set()
    if args.append and out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["facet"], r["domain"], r["scenario"]))
            except Exception:
                pass

    # 셀별 limit + 총 max 적용
    selected, per_cell = [], {}
    for r in rows:
        if r["facet"] not in tf:
            continue
        if td and r["domain"] not in td:
            continue
        if (r["facet"], r["domain"], r["scenario"]) in done:
            continue
        key = (r["facet"], r["domain"])
        if args.limit and per_cell.get(key, 0) >= args.limit:
            continue
        per_cell[key] = per_cell.get(key, 0) + 1
        selected.append(r)
        if args.max and len(selected) >= args.max:
            break

    print(f"판정모델={judge_model} | 대상 {len(selected)}쌍 "
          f"(facets={tf}, domains={args.domains})"
          + (f" | resume {len(done)} 건너뜀" if done else ""))

    mode = "a" if args.append else "w"
    n_pass = 0
    fail_counts = {"facet": 0, "polarity": 0, "minimal_edit": 0, "leak": 0}
    per_facet = {f: {"n": 0, "pass": 0} for f in C.EXTRAVERSION_FACETS}
    confusion = {}   # (labeled -> predicted) 카운트(불일치만)
    leaks = {}       # leaked_trait -> count
    n = 0
    try:
        with out_path.open(mode, encoding="utf-8") as fh:
            for r in selected:
                prompt = build_prompt(fb, r["facet"], r["scenario"],
                                      r["positive"], r["negative"])
                if claude:
                    v = C.anthropic_json(client, judge_model, prompt, VERDICT_SCHEMA)
                else:
                    v = C.chat_json(client, judge_model, prompt, temperature=0.0)
                flags = quality_flags(r["positive"], r["negative"])
                passed = verdict_pass(v)
                out = {**r, "verdict": v, "heur_flags": flags, "pass": passed,
                       "judge_model": judge_model}
                fh.write(json.dumps(out, ensure_ascii=False) + "\n")
                fh.flush()

                n += 1
                pf = per_facet[r["facet"]]
                pf["n"] += 1
                if passed:
                    n_pass += 1
                    pf["pass"] += 1
                if not v.get("facet_correct"):
                    fail_counts["facet"] += 1
                if not v.get("polarity_correct"):
                    fail_counts["polarity"] += 1
                if not v.get("minimal_edit"):
                    fail_counts["minimal_edit"] += 1
                if v.get("leak_outside_extraversion"):
                    fail_counts["leak"] += 1
                    lt = (v.get("leaked_trait") or "unspecified")
                    leaks[lt] = leaks.get(lt, 0) + 1
                pred = v.get("predicted_facet")
                if pred and pred != r["facet"]:
                    confusion[(r["facet"], pred)] = confusion.get((r["facet"], pred), 0) + 1
                mark = "✓" if passed else "✗"
                bad = [k for k, ok in (("facet", v.get("facet_correct")),
                                       ("pol", v.get("polarity_correct")),
                                       ("min", v.get("minimal_edit"))) if not ok]
                if v.get("leak_outside_extraversion"):
                    bad.append(f"leak:{v.get('leaked_trait')}")
                print(f"  [{n}/{len(selected)}] {mark} {r['facet'][:6]}/{r['domain'][:4]}"
                      + (f"  ⚠ {','.join(bad)}" if bad else ""), flush=True)
    except C.QuotaExhausted as e:
        print(f"\n{e}\n진행분 {n}쌍은 {out_path} 에 저장됨(--append 로 재개).", flush=True)
        sys.exit(2)

    # --- 요약 ---------------------------------------------------------- #
    print(f"\n=== 스크리닝 요약 ({n}쌍) ===")
    print(f"합격: {n_pass} ({(n_pass/n*100 if n else 0):.1f}%)  |  불합격: {n - n_pass}")
    print(f"실패 유형: facet {fail_counts['facet']} · 극성 {fail_counts['polarity']} · "
          f"minimal-edit {fail_counts['minimal_edit']} · 외향성밖누수 {fail_counts['leak']}")
    print("facet별 합격률:")
    for f in C.EXTRAVERSION_FACETS:
        p = per_facet[f]
        if p["n"]:
            print(f"  {f:18} {p['pass']:>3}/{p['n']:<3} ({p['pass']/p['n']*100:.0f}%)")
    if leaks:
        print("외향성 밖 누수 특성:", ", ".join(f"{k}={v}" for k, v in
                                       sorted(leaks.items(), key=lambda x: -x[1])))
    if confusion:
        print("facet 혼동(라벨→예측, 상위):")
        for (a, b), c in sorted(confusion.items(), key=lambda x: -x[1])[:6]:
            print(f"  {a} → {b}: {c}")
    print(f"\n완료 → {out_path}")


if __name__ == "__main__":
    main()
