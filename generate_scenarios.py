#!/usr/bin/env python3
"""[모듈 1 · 0단계] 시나리오 풀 구축 (Scenario Pool Construction)

논문(arXiv:2602.19157) Section 3-2: cue를 4개 도메인에 매핑하고, facet을 자연스럽게
유발하는 **서로 다른 구체적 시나리오**를 설계한다.

논문은 시나리오당 1쌍이므로, facet당 250쌍을 채우려면 **facet당 ~250개의 고유
시나리오**(4 도메인 × ~62)가 필요하다. 사람이 다 쓰는 건 무리 → LLM으로 생성하고
(응답 내 중복 회피 + 누적 dedup), 사람이 검토·정제한다.

- 시나리오는 **상황만** 기술한다(외향/내향 반응·행동 금지). 반응은 1단계에서 부여.
- 기존 scenarios_en.json 의 사람 시드는 풀의 시작점이자 중복 회피 기준으로 재활용.

출력: data/scenarios_pool.json  (scenarios_en.json 과 동일 스키마)
  {"trait": "extraversion", "note": "...",
   "scenarios": [{"facet","domain","scenario"}, ...]}
기본 동작은 **병합(merge)** — 풀 파일을 읽어 새 항목만 dedup 후 추가(facet/도메인별로
나눠 돌려 누적). --fresh 로 덮어쓰기.

단계적 사용:
  # 0단계 시험: assertiveness × study 에서 ~62개
  python generate_scenarios.py --facets assertiveness --domains study --target 62
  # assertiveness 전 도메인
  python generate_scenarios.py --facets assertiveness --target 62
  # 전 facet
  python generate_scenarios.py --facets all --target 62
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from openai import OpenAI

import common as C

POOL_PATH = C.DATA_DIR / "scenarios_pool.json"

PROMPT_TEMPLATE = """\
List {m} DIFFERENT everyday situations in the {domain_desc} domain that would
naturally call for {facet_title} ({facet} — a facet of Extraversion): situations
where a person could lean toward the high pole ({cues_high}) OR the low pole
({cues_low}).

Rules:
- Each item is a one-line SITUATION description only. Do NOT write any response,
  reaction, or behavior; describe only the situation/context.
- All {m} must be clearly different from one another; no near-duplicates.
- Realistic and specific (e.g. "dividing roles in a group project",
  "a seminar discussion has stalled and no one is talking").
- Do NOT reuse or lightly reword any of these already-collected situations:
{avoid_block}

Output a JSON object only:
{{"scenarios": ["situation 1", "situation 2", ... exactly {m} strings ...]}}
"""


# --------------------------------------------------------------------------- #
def load_pool() -> dict:
    """풀 파일 → {(facet,domain): [scenario, ...]}."""
    if not POOL_PATH.exists():
        return {}
    raw = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    out: dict = {}
    for s in raw.get("scenarios", []):
        out.setdefault((s["facet"], s["domain"]), []).append(s["scenario"])
    return out


def write_pool(pool: dict) -> int:
    rows = []
    for (facet, domain) in sorted(pool):
        for sc in pool[(facet, domain)]:
            rows.append({"facet": facet, "domain": domain, "scenario": sc})
    POOL_PATH.write_text(json.dumps(
        {"trait": "extraversion",
         "note": "Stage-0 expanded scenario pool. Situations only (no responses); "
                 "1 minimal-edit pair per scenario in Stage 1.",
         "scenarios": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(rows)


def is_dup(cand: str, collected_tok: list[set], collected_norm: set,
           thresh: float = 0.8) -> bool:
    if C.norm(cand) in collected_norm:
        return True
    ct = C.toks(cand)
    return any(C.jaccard(ct, t) >= thresh for t in collected_tok)


def avoid_block(collected: list[str], k: int = 80) -> str:
    sample = collected[-k:] if len(collected) > k else collected
    return "\n".join(f"  - {s}" for s in sample) if sample else "  (none yet)"


def build_for_cell(client, model, temperature, facet, fdef, domain,
                   target, per_call, max_calls, seeds):
    """(facet, domain) 한 셀에 대해 고유 시나리오를 target 개까지 모은다."""
    collected = list(seeds)  # 시드로 시작(사람 검토본 재활용)
    collected_norm = {C.norm(s) for s in collected}
    collected_tok = [C.toks(s) for s in collected]
    calls = 0
    while len(collected) < target and calls < max_calls:
        m = min(per_call, target - len(collected) + 5)
        prompt = PROMPT_TEMPLATE.format(
            m=m, facet=facet, facet_title=facet.replace("_", " ").title(),
            domain_desc=C.DOMAINS[domain],
            cues_high=C.fmt_cues(fdef["cues"]["high"]),
            cues_low=C.fmt_cues(fdef["cues"]["low"]),
            avoid_block=avoid_block(collected),
        )
        obj = C.chat_json(client, model, prompt, temperature)
        cands = obj.get("scenarios", [])
        calls += 1
        added = 0
        for s in cands:
            s = (s or "").strip()
            if not s or is_dup(s, collected_tok, collected_norm):
                continue
            collected.append(s)
            collected_norm.add(C.norm(s))
            collected_tok.append(C.toks(s))
            added += 1
            if len(collected) >= target:
                break
        print(f"    call{calls}: 요청 {m}, 신규 {added} → 누적 {len(collected)}/{target}",
              flush=True)
    return collected


def pool_diversity_report(scenarios: list[str], thresh: float = 0.6) -> None:
    tk = [C.toks(s) for s in scenarios]
    near = []
    for i in range(len(scenarios)):
        for j in range(i + 1, len(scenarios)):
            sim = C.jaccard(tk[i], tk[j])
            if sim >= thresh:
                near.append((i, j, sim))
    print(f"    다양성: {len(scenarios)}개, 유사쌍(≥{thresh}) {len(near)}건")
    for i, j, sim in near[:5]:
        print(f"      ~{sim:.2f}: \"{scenarios[i][:40]}\" / \"{scenarios[j][:40]}\"")


# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description="모듈 1 · 0단계 — 시나리오 풀 구축")
    ap.add_argument("--facets", default="assertiveness",
                    help="쉼표구분 facet 또는 'all' (기본: assertiveness)")
    ap.add_argument("--domains", default="all",
                    help="쉼표구분 도메인 또는 'all' (study,work,daily,leisure)")
    ap.add_argument("--target", type=int, default=62, help="facet×도메인당 시나리오 수")
    ap.add_argument("--per-call", type=int, default=25, help="호출당 요청 개수 M")
    ap.add_argument("--max-calls", type=int, default=8, help="셀당 최대 호출 수(안전장치)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--fresh", action="store_true", help="풀 병합 대신 덮어쓰기")
    return ap.parse_args()


def main():
    args = parse_args()
    api_key, model = C.load_env_and_model(args.model)
    temperature = (args.temperature if args.temperature is not None
                   else float(os.getenv("GEN_TEMPERATURE", "0.8")))
    client = OpenAI(api_key=api_key)

    facets_def = C.load_facets()
    seeds = C.load_seed_scenarios()
    target_facets = (C.EXTRAVERSION_FACETS if args.facets.strip().lower() == "all"
                     else [f.strip() for f in args.facets.split(",") if f.strip()])
    target_domains = (list(C.DOMAINS) if args.domains.strip().lower() == "all"
                      else [d.strip() for d in args.domains.split(",") if d.strip()])
    for d in target_domains:
        if d not in C.DOMAINS:
            sys.exit(f"[인자 오류] 알 수 없는 도메인 '{d}'. 허용: {list(C.DOMAINS)}")

    pool = {} if args.fresh else load_pool()
    print(f"모델={model} temp={C.temp_label(model, temperature)} "
          f"facets={target_facets} domains={target_domains} target={args.target}/셀 "
          f"→ {POOL_PATH.name} ({'fresh' if args.fresh else 'merge'})")

    for facet in target_facets:
        if facet not in facets_def:
            sys.exit(f"[입력 오류] facets_en.json 에 facet '{facet}' 없음.")
        for domain in target_domains:
            existing = pool.get((facet, domain), [])
            seed_list = seeds.get((facet, domain), [])
            # 시드 중 풀에 없는 것 합치기(중복 제거)
            start = list(existing)
            snorm = {C.norm(s) for s in start}
            for s in seed_list:
                if C.norm(s) not in snorm:
                    start.append(s); snorm.add(C.norm(s))
            print(f"\n[{facet} × {domain}] 시작(기존 {len(existing)} + 시드 {len(seed_list)} "
                  f"→ {len(start)})", flush=True)
            collected = build_for_cell(client, model, temperature, facet,
                                       facets_def[facet], domain, args.target,
                                       args.per_call, args.max_calls, start)
            pool[(facet, domain)] = collected
            pool_diversity_report(collected)
            write_pool(pool)  # 셀마다 증분 저장(중간 실패 대비 + 진행 가시성)

    total = write_pool(pool)
    print(f"\n완료: 풀 총 {total}개 시나리오 → {POOL_PATH}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except C.QuotaExhausted as e:
        print(f"\n{e}\n(셀마다 증분 저장되므로 재실행 시 이어서 채움)", flush=True)
        sys.exit(2)
