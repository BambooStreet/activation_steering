#!/usr/bin/env python3
"""[모듈 1 · 1단계 · Batch] 대조 쌍 생성 — OpenAI Batch API 버전

generate_pairs.py 와 동일한 로직(cue-rotation + minimal-edit 프롬프트 + 시나리오당 1쌍)을
**Batch API**로 실행한다(약 50% 저렴, 비동기 24h 윈도). 저렴한 모델(gpt-5.4-mini 등)
품질 검증/대량 생성용.

흐름: 풀에서 작업 구성 → batch_input.jsonl 작성 → 업로드 → 배치 생성 → 완료까지 폴링
      → 결과 다운로드 → outputs 에 {facet,domain,scenario,positive,negative} 기록.

사용:
  python batch_pairs.py --facets excitement_seeking --model gpt-5.4-mini \
      --out outputs/pairs_es_mini.jsonl
폴링은 백그라운드 권장(완료까지 분~시간).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from openai import OpenAI

import common as C
from generate_pairs import build_prompt, quality_flags

SYSTEM = "You output only valid JSON matching the requested schema."


def build_tasks(facets_def, grouped, target_facets, target_domains, limit):
    """(custom_id -> task) 매핑 + cue-rotation 적용한 요청 리스트 생성."""
    tasks = {}
    for facet in target_facets:
        if facet not in facets_def:
            sys.exit(f"[입력 오류] facet '{facet}' 없음.")
        high = C.split_cues(facets_def[facet]["cues"]["high"])
        low = C.split_cues(facets_def[facet]["cues"]["low"])
        rot = 0
        domains = sorted({d for (f, d) in grouped if f == facet})
        if target_domains:
            domains = [d for d in domains if d in target_domains]
        for domain in domains:
            scens = grouped[(facet, domain)]
            if limit:
                scens = scens[:limit]
            for idx, scenario in enumerate(scens):
                a_high, a_low = high[rot % len(high)], low[rot % len(low)]
                rot += 1
                cid = f"{facet}__{domain}__{idx:04d}"
                tasks[cid] = {
                    "facet": facet, "domain": domain, "scenario": scenario,
                    "prompt": build_prompt(facet, facets_def[facet], domain,
                                           scenario, a_high, a_low),
                }
    return tasks


def write_batch_input(tasks, model, path):
    with path.open("w", encoding="utf-8") as fh:
        for cid, t in tasks.items():
            body = {"model": model,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": t["prompt"]}]}
            if C.supports_custom_temperature(model):
                body["temperature"] = 0.8
            fh.write(json.dumps({"custom_id": cid, "method": "POST",
                                 "url": "/v1/chat/completions", "body": body},
                                ensure_ascii=False) + "\n")


def poll_batch(client, batch_id, interval=20):
    while True:
        b = client.batches.retrieve(batch_id)
        rc = b.request_counts
        print(f"    상태={b.status} 완료={rc.completed}/{rc.total} 실패={rc.failed}",
              flush=True)
        if b.status in ("completed", "failed", "expired", "cancelled"):
            return b
        time.sleep(interval)


def parse_output(client, output_file_id, tasks):
    text = client.files.content(output_file_id).text
    rows, errors = [], 0
    for line in text.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        cid = rec.get("custom_id")
        t = tasks.get(cid)
        if rec.get("error") or not rec.get("response"):
            errors += 1
            continue
        try:
            content = rec["response"]["body"]["choices"][0]["message"]["content"]
            obj = json.loads(content)
        except Exception:
            errors += 1
            continue
        rows.append({"facet": t["facet"], "domain": t["domain"],
                     "scenario": t["scenario"],
                     "positive": (obj.get("positive") or "").strip(),
                     "negative": (obj.get("negative") or "").strip()})
    return rows, errors


def parse_args():
    ap = argparse.ArgumentParser(description="모듈 1 · 1단계 Batch 러너")
    ap.add_argument("--facets", default="excitement_seeking")
    ap.add_argument("--domains", default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=None, help="기본 .env GEN_MODEL (mini 가능)")
    ap.add_argument("--scenarios-file", default=str(C.DATA_DIR / "scenarios_pool.json"))
    ap.add_argument("--out", default=str(C.OUT_DIR / "pairs_batch.jsonl"))
    ap.add_argument("--poll", type=int, default=20, help="폴링 간격(초)")
    return ap.parse_args()


def main():
    args = parse_args()
    api_key, model = C.load_env_and_model(args.model)
    client = OpenAI(api_key=api_key)

    facets_def = C.load_facets()
    raw = json.loads(Path(args.scenarios_file).read_text(encoding="utf-8"))
    grouped = {}
    for s in raw["scenarios"]:
        grouped.setdefault((s["facet"], s["domain"]), []).append(s["scenario"])

    target_facets = (C.EXTRAVERSION_FACETS if args.facets.strip().lower() == "all"
                     else [f.strip() for f in args.facets.split(",") if f.strip()])
    target_domains = (None if args.domains.strip().lower() == "all"
                      else {d.strip() for d in args.domains.split(",") if d.strip()})

    tasks = build_tasks(facets_def, grouped, target_facets, target_domains, args.limit)
    print(f"모델={model} 작업 {len(tasks)}건 (facets={target_facets})", flush=True)

    C.OUT_DIR.mkdir(exist_ok=True)
    in_path = C.OUT_DIR / "batch_input.jsonl"
    write_batch_input(tasks, model, in_path)

    up = client.files.create(file=in_path.open("rb"), purpose="batch")
    batch = client.batches.create(input_file_id=up.id,
                                  endpoint="/v1/chat/completions",
                                  completion_window="24h")
    print(f"배치 생성: {batch.id}", flush=True)
    batch = poll_batch(client, batch.id, args.poll)
    if batch.status != "completed":
        sys.exit(f"[배치 실패] status={batch.status}")

    rows, errors = parse_output(client, batch.output_file_id, tasks)
    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    flagged = sum(1 for r in rows if quality_flags(r["positive"], r["negative"]))
    print(f"\n완료: {len(rows)}쌍 (오류 {errors}) → {out_path} | 플래그 {flagged}쌍",
          flush=True)


if __name__ == "__main__":
    try:
        main()
    except C.QuotaExhausted as e:
        print(f"\n{e}", flush=True)
        sys.exit(2)
