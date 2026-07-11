#!/usr/bin/env python3
"""[모듈 3] Gemma 2 2B 활성화 추출·캐시 (CAA/SAE 공용 입력).

대조쌍 각 문장을 base 모델에 통과시켜 **레이어별 pooled 활성화**만 디스크에 캐시한다.
per-token 이 아니라 pooled(mean+last)만 저장 → V1/V2 벡터 구성과 레이어 스윕(4a)을
**GPU 없이 numpy 재계산**으로 반복 가능(≈0.7GB).

입력:  outputs/pairs_raw.jsonl (또는 스텝2 재균등화 산출) — {facet,domain,positive,negative,...}
출력(artifacts/activations/):
  pooled_mean.npy  [2N, 27, 2304] fp16   (BOS/pad 제외 평균)
  pooled_last.npy  [2N, 27, 2304] fp16   (마지막 실토큰)
  index.jsonl      각 행 {row, pair_id, split(pos/neg), facet, domain, gen_model}
  layer_norms.npy  [27] fp32             (레이어별 평균 residual norm — 4b α 스케일)

클라우드 GPU 실행:
  pip install -r requirements-steering.txt
  huggingface-cli login   # 또는 .env HF_TOKEN
  python steering/extract_activations.py                 # 전체(1,488×2=2,976)
  python steering/extract_activations.py --limit 8       # 스모크(문장 8개)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import gemma_common as G


def load_pairs(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"[입력 누락] {path} 없음.")
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def flatten(pairs: list[dict]) -> list[dict]:
    """각 쌍 → 2 예시(pos/neg). 배열 행 순서 == 이 리스트 순서."""
    rows = []
    for i, r in enumerate(pairs):
        base = {"pair_id": i, "facet": r["facet"], "domain": r["domain"],
                "gen_model": r.get("gen_model", "")}
        rows.append({**base, "split": "pos", "text": (r.get("positive") or "").strip()})
        rows.append({**base, "split": "neg", "text": (r.get("negative") or "").strip()})
    return rows


def parse_args():
    ap = argparse.ArgumentParser(description="모듈 3 — Gemma 2 2B 활성화 추출·캐시")
    ap.add_argument("--in", dest="inp", default=str(G.C.OUT_DIR / "pairs_final.jsonl"))
    ap.add_argument("--out-dir", default=str(G.ACT_DIR))
    ap.add_argument("--model", default=G.MODEL_ID)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="예시 수 제한(스모크)")
    return ap.parse_args()


def main():
    args = parse_args()
    device = G.pick_device()
    if device == "cpu":
        print("[경고] CPU 실행 — 느림. 전체 추출은 GPU 권장(스모크는 가능).", flush=True)

    pairs = load_pairs(Path(args.inp))
    rows = flatten(pairs)
    if args.limit:
        rows = rows[:args.limit]
    n = len(rows)
    print(f"모델={args.model} device={device} | 예시 {n}개(쌍 {n // 2}) → {args.out_dir}",
          flush=True)

    model, tok = G.load_model(args.model, device=device)
    n_hs = model.config.num_hidden_layers + 1          # 모델-적응형 (2B=27, 9B=43)
    hidden = model.config.hidden_size
    print(f"  n_hidden_states={n_hs} hidden={hidden} mid_layer={G.mid_layer(n_hs)}", flush=True)

    mean_arr = np.zeros((n, n_hs, hidden), dtype=np.float16)
    last_arr = np.zeros((n, n_hs, hidden), dtype=np.float16)

    # 길이 정렬 배치(패딩 낭비↓) — 결과는 원래 행 인덱스로 scatter
    order = sorted(range(n), key=lambda k: len(rows[k]["text"]))
    bs = args.batch_size
    for s in tqdm(range(0, n, bs), desc="extract"):
        idx = order[s:s + bs]
        texts = [rows[k]["text"] for k in idx]
        mean_p, last_p = G.pooled_hidden(model, tok, texts, device, args.max_len)
        mp = mean_p.numpy().astype(np.float16)
        lp = last_p.numpy().astype(np.float16)
        for j, k in enumerate(idx):
            mean_arr[k] = mp[j]
            last_arr[k] = lp[j]

    # 레이어별 residual norm(4b α 스케일) — 표본으로 산정
    sample = [rows[k]["text"] for k in range(min(128, n))]
    norms = G.layer_residual_norms(model, tok, sample, device, args.max_len).numpy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "pooled_mean.npy", mean_arr)
    np.save(out_dir / "pooled_last.npy", last_arr)
    np.save(out_dir / "layer_norms.npy", norms.astype(np.float32))
    (out_dir / "meta.json").write_text(json.dumps(
        {"model": args.model, "n_hidden_states": n_hs, "hidden": hidden,
         "mid_layer": G.mid_layer(n_hs)}, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "index.jsonl").open("w", encoding="utf-8") as fh:
        for row_i, r in enumerate(rows):
            fh.write(json.dumps({"row": row_i, "pair_id": r["pair_id"], "split": r["split"],
                                 "facet": r["facet"], "domain": r["domain"],
                                 "gen_model": r["gen_model"]}, ensure_ascii=False) + "\n")

    print(f"완료: pooled_mean/last {mean_arr.shape} fp16, index {n}행, "
          f"layer_norms {norms.shape} → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
