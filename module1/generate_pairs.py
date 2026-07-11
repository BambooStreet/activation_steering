#!/usr/bin/env python3
"""[모듈 1 · 1단계] 대조 쌍 생성 (Contrastive Pair Generation) — 시나리오당 1쌍

논문(arXiv:2602.19157) Section 3-3을 그대로 따름:
  "For each scenario, the model produces a positive sample and a minimal-edit
   negative counterpart." → **시나리오 하나당 정확히 1쌍**(변형 없음).

규모: facet당 250쌍 × 6 = 1,500쌍. 시나리오가 모두 고유하므로 변형이 아니라
0단계에서 만든 **고유 시나리오 풀**(data/scenarios_pool.json)로 규모를 채운다.

핵심 원칙
  - LLM은 시나리오를 발명하지 않는다 — 0단계 풀의 시나리오만 슬롯에 주입.
  - positive=high pole, negative=low pole. 두 문장은 같은 시나리오·구조 공유,
    행동 동사구만 반대로(=minimal edit). negative 는 능동적 low-pole(단순 부정 X).
  - 약한 모델(mini) 금지.

입력:
  data/facets_en.json       (definition, cues)
  data/scenarios_pool.json  (0단계 산출 — 없으면 generate_scenarios.py 먼저)
출력:
  outputs/pairs_raw.jsonl   각 줄 {facet, domain, scenario, positive, negative}

단계적 사용:
  # 1단계 시험: assertiveness × study 풀로 쌍 생성
  python generate_pairs.py --facets assertiveness --domains study
  # assertiveness 전 도메인(≈250)
  python generate_pairs.py --facets assertiveness
  # 전 facet(≈1,500)
  python generate_pairs.py --facets all
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from openai import OpenAI

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))  # module1/ → 루트(common.py)
import common as C

POOL_PATH = C.DATA_DIR / "scenarios_pool.json"

PROMPT_TEMPLATE = """\
Write a pair of short, first-person sentences that express a single Big Five
facet, differing only in the behavior.

Facet: {facet_title} (a facet of Extraversion)
Definition: {definition}
Cues:
- High pole (positive): {cues_high}
- Low pole (negative): {cues_low}
Context (domain): {domain}
Scenario: {scenario}

Rules:
1. MINIMAL EDIT: positive and negative share the SAME scenario and SAME sentence
   structure; change ONLY the behavioral verb phrase; shared context words
   identical in both.
2. Positive = high pole, Negative = low pole (per cues).
3. Negative is an ACTIVE low-pole behavior, NOT a negation, and NOT a reduced
   amount of the positive. Never phrase it as "less/little/rarely/barely/hardly"
   of the high pole; use a concrete opposite action (a calm, composed, neutral,
   reserved, or alternative behavior).
   (e.g. "I stay quiet and let others decide", NOT "I don't speak up";
    "I keep a flat, neutral expression", NOT "I show little delight".)
4. First person, present tense, each sentence <= 18 words, both within ~4 words.
5. Express ONLY {facet} ; no leakage of other traits/facets.
6. ANCHOR the behavioral verb phrase on this specific behavior pair (do not fall
   back to generic wording): positive expresses "{anchor_high}", negative
   expresses its opposite pole "{anchor_low}". Phrase it naturally for the scenario.

Output a JSON object only:
{{"positive": "...", "negative": "..."}}
"""


# --------------------------------------------------------------------------- #
def load_pool(path) -> list[dict]:
    if not path.exists():
        sys.exit(f"[입력 누락] {path} 없음. 0단계(generate_scenarios.py)를 먼저 실행.")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw.get("scenarios", [])


def build_prompt(facet, fdef, domain, scenario, anchor_high, anchor_low) -> str:
    return PROMPT_TEMPLATE.format(
        facet=facet, facet_title=facet.replace("_", " ").title(),
        definition=fdef["definition"],
        cues_high=C.fmt_cues(fdef["cues"]["high"]),
        cues_low=C.fmt_cues(fdef["cues"]["low"]),
        domain=domain, scenario=scenario,
        anchor_high=anchor_high, anchor_low=anchor_low,
    )


NEGATION_TOKS = {"not", "no", "never", "don't", "doesn't", "didn't", "won't",
                 "can't", "cannot", "isn't", "aren't", "wouldn't"}
ATTENUATION_TOKS = {"less", "little", "rarely", "barely", "hardly", "seldom",
                    "minimal", "fewer", "slightly", "somewhat", "mild", "mildly"}


def quality_flags(positive: str, negative: str) -> list[str]:
    """하드 실패 아님 — 사람 검토 + 모듈 2 검증용 경고.
    negation/attenuation 은 시나리오 공유절이 아니라 **행동구 diff**(neg-only 토큰)에서만 판정."""
    flags = []
    wp, wn = C.word_count(positive), C.word_count(negative)
    if wp > 18:
        flags.append(f"pos>18w({wp})")
    if wn > 18:
        flags.append(f"neg>18w({wn})")
    if abs(wp - wn) > 4:
        flags.append(f"len_diff>4({abs(wp - wn)})")
    contrast_neg = C.toks(negative) - C.toks(positive)
    if contrast_neg & NEGATION_TOKS:
        flags.append("negation")
    if contrast_neg & ATTENUATION_TOKS:
        flags.append("attenuation")
    pair_overlap = C.jaccard(C.toks(positive), C.toks(negative))
    if pair_overlap < 0.3:
        flags.append(f"low_overlap({pair_overlap:.2f})")
    return flags


# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description="모듈 1 · 1단계 — 대조 쌍 생성(시나리오당 1쌍)")
    ap.add_argument("--facets", default="assertiveness",
                    help="쉼표구분 facet 또는 'all' (기본: assertiveness)")
    ap.add_argument("--domains", default="all",
                    help="쉼표구분 도메인 또는 'all' (study,work,daily,leisure)")
    ap.add_argument("--limit", type=int, default=None,
                    help="facet×도메인당 시나리오 수 제한(빠른 시험용)")
    ap.add_argument("--scenarios-file", default=str(POOL_PATH))
    ap.add_argument("--out", default=str(C.OUT_DIR / "pairs_raw.jsonl"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--append", action="store_true", help="기존 출력에 이어붙임")
    return ap.parse_args()


def main():
    args = parse_args()
    api_key, model = C.load_env_and_model(args.model)
    temperature = (args.temperature if args.temperature is not None
                   else float(os.getenv("GEN_TEMPERATURE", "0.8")))
    client = OpenAI(api_key=api_key)

    facets_def = C.load_facets()
    pool = load_pool(Path(args.scenarios_file))

    target_facets = (C.EXTRAVERSION_FACETS if args.facets.strip().lower() == "all"
                     else [f.strip() for f in args.facets.split(",") if f.strip()])
    target_domains = (None if args.domains.strip().lower() == "all"
                      else {d.strip() for d in args.domains.split(",") if d.strip()})

    # 풀을 (facet,domain)별로 그룹화
    grouped: dict = {}
    for s in pool:
        grouped.setdefault((s["facet"], s["domain"]), []).append(s["scenario"])

    C.OUT_DIR.mkdir(exist_ok=True)
    out_path = Path(args.out)
    mode = "a" if args.append else "w"

    # 재개-안전: append 모드면 기존 출력의 (facet,domain,scenario)는 건너뜀(중복 방지)
    done = set()
    if args.append and out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["facet"], r["domain"], r["scenario"]))
            except Exception:
                pass
    print(f"모델={model} temp={C.temp_label(model, temperature)} "
          f"facets={target_facets} domains={args.domains} "
          f"limit={args.limit or '전체'} → {out_path}"
          + (f" | 기존 {len(done)}쌍 건너뜀(resume)" if done else ""))

    total = 0
    try:
      with out_path.open(mode, encoding="utf-8") as fh:
        for facet in target_facets:
            if facet not in facets_def:
                sys.exit(f"[입력 오류] facets_en.json 에 facet '{facet}' 없음.")
            # cue-rotation: 5개 high/low cue를 facet 전체에 고르게 분산(어휘 쏠림 방지)
            high_cues = C.split_cues(facets_def[facet]["cues"]["high"])
            low_cues = C.split_cues(facets_def[facet]["cues"]["low"])
            rot = 0  # facet 단위 카운터(도메인 경계 넘어 누적)
            domains = sorted({d for (f, d) in grouped if f == facet})
            if target_domains:
                domains = [d for d in domains if d in target_domains]
            if not domains:
                print(f"[{facet}] 풀에 해당 도메인 시나리오 없음 — 건너뜀.")
                continue
            for domain in domains:
                scens = grouped[(facet, domain)]
                if args.limit:
                    scens = scens[:args.limit]
                print(f"\n[{facet} × {domain}] {len(scens)}개 시나리오 → 쌍 생성")
                for i, scenario in enumerate(scens, 1):
                    a_high = high_cues[rot % len(high_cues)]
                    a_low = low_cues[rot % len(low_cues)]
                    rot += 1
                    if (facet, domain, scenario) in done:
                        continue  # 이미 생성됨(resume) — rot 은 정렬 위해 증가시킴
                    obj = C.chat_json(client, model,
                                      build_prompt(facet, facets_def[facet], domain,
                                                   scenario, a_high, a_low),
                                      temperature)
                    row = {
                        "facet": facet, "domain": domain, "scenario": scenario,
                        "positive": (obj.get("positive") or "").strip(),
                        "negative": (obj.get("negative") or "").strip(),
                    }
                    flags = quality_flags(row["positive"], row["negative"])
                    tag = f"  ⚠ {','.join(flags)}" if flags else ""
                    print(f"  [{i:>3}/{len(scens)}] ({a_high[:18]}) "
                          f"{row['positive'][:48]}{tag}", flush=True)
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
                    total += 1
    except C.QuotaExhausted as e:
        print(f"\n{e}\n진행분 {total}쌍은 {out_path} 에 저장됨(이어서 --append 가능).",
              flush=True)
        sys.exit(2)
    print(f"\n완료: 총 {total}쌍 → {out_path}")


if __name__ == "__main__":
    main()
