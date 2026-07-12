#!/usr/bin/env python3
"""[모듈 4] CAA 스티어링 벡터 구성·저장 (V1 채택, layer 12).

4a 진단이 'V1, layer 12' 로 결론 → 캐시된 활성화에서 V1(예시별 L2→facet 균등)을 만들고
정제(mean-centering + L2)해 **단위 벡터**로 저장. 비교용 V2 도 함께 저장.

정제:
  - mean-centering: 전체 활성화 평균 방향 성분을 벡터에서 제거(공통 '템플릿' 오프셋 제거).
  - L2 정규화: 주입 방향을 단위벡터로(4b 에서 α=c·R 로 스케일).

입력:  artifacts/activations/pooled_mean.npy, index.jsonl, layer_norms.npy
출력:  artifacts/vectors/layer{L}_extraversion_v1.npy (단위벡터, 주입용)
       artifacts/vectors/layer{L}_extraversion_v2.npy (비교용)
       artifacts/vectors/layer{L}_meta.json

실행(추출 후, GPU 불필요):
  python steering/build_vectors.py            # layer 12
  python steering/build_vectors.py --layer 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import gemma_common as G
import diagnostics as D


def refine(v: np.ndarray, actL: np.ndarray) -> np.ndarray:
    """mean-centering(전체 평균 방향 성분 제거) + L2 단위화."""
    mu = actL.mean(0)
    mu_hat = mu / (np.linalg.norm(mu) + D.EPS)
    v = v - (v @ mu_hat) * mu_hat
    return v / (np.linalg.norm(v) + D.EPS)


def parse_args():
    ap = argparse.ArgumentParser(description="모듈 4 — CAA 벡터 구성·저장(V1)")
    ap.add_argument("--act-dir", default=str(G.ACT_DIR))
    ap.add_argument("--layer", type=int, default=None, help="미지정 시 모델 중간층 자동")
    ap.add_argument("--out-dir", default=str(G.VEC_DIR))
    ap.add_argument("--standardize", action="store_true",
                    help="차원별 z-score 후 CAA (거대활성 차원 다운웨이트; Qwen 등). 기본 off")
    return ap.parse_args()


def main():
    args = parse_args()
    acts, idx = D.load_cache(Path(args.act_dir))
    L = args.layer if args.layer is not None else G.mid_layer(acts.shape[1])
    actL = acts[:, L, :].astype(np.float64)
    if args.standardize:                       # 거대활성 차원 다운웨이트 → 트레이트 신호 보존
        actL = actL / (actL.std(0) + D.EPS)    # 이후 split/build/refine 전부 z-space 일관
    P, N, P_all, N_all = D.split_by_facet(actL, idx)
    V1, _sub = D.build_V1(P, N)
    V2 = D.build_V2(P_all, N_all)
    v1r = refine(V1, actL)
    v2r = refine(V2, actL)

    R = float(np.load(Path(args.act_dir) / "layer_norms.npy")[L])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"layer{L}_extraversion_v1.npy", v1r.astype(np.float32))
    np.save(out / f"layer{L}_extraversion_v2.npy", v2r.astype(np.float32))
    meta = {"layer": L, "residual_norm_R": round(R, 3), "standardized": bool(args.standardize),
            "cos_V1_V2": round(float(D._unit(V1) @ D._unit(V2)), 4),
            "cos_refinedV1_V2": round(float(v1r @ v2r), 4),
            "adopted": "v1", "hidden": int(v1r.shape[0]), "n_pairs": len(idx) // 2}
    (out / f"layer{L}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print(f"저장 → {out}/layer{L}_extraversion_v1.npy (단위벡터, dim={v1r.shape[0]})")
    print(f"  residual_norm R={R:.1f} (4b α=c·R 스케일) | cos(V1,V2)={meta['cos_V1_V2']}")


if __name__ == "__main__":
    main()
