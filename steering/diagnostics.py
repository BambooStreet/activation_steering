#!/usr/bin/env python3
"""[모듈 4a] CAA 벡터 진단 — V1(예시별 L2→facet 균등) vs V2(전체 풀링) 비교가 중심.

핵심 질문(사용자): 개수는 facet당 균등하나 **per-example norm 차이로 실질 기여는 불균등**할 수
있다 → 반드시 두 구성을 정량 비교해 근거로 채택한다.
  V1: 각 예시 활성화를 L2 정규화 → facet별 pos/neg 평균차 → 6 facet 균등가중 평균 → 단일 벡터.
  V2: 1,488쌍 전체 풀링 plain difference-of-means(norm 무보정).

입력:  artifacts/activations/pooled_mean.npy [2N,27,H], index.jsonl
출력:  artifacts/vectors/diagnostics.json (레이어별 지표) + 콘솔 요약

지표(레이어별):
  1. cos(V1,V2) — 헤드라인. >0.95 무관(V1 채택), <0.9 → 4b 둘 다.
  2. facet별 활성화 norm 분포 + 교차비(max/min). ES(gpt-5.4)·gregariousness 별도 관찰.
  3. facet별 V2 방향 기여도 c_f(합=1, 균등≈0.167; ≥0.35 지배 → V1 선호).
  4. facet 하위벡터 6×6 코사인(공통 외향성 축 존재 여부).
  5. SVD PC1 설명분산(≳0.7 → 단일 방향 강함).
  6. cue 의존성(캐시만으로): leave-one-facet-out cos, 도메인 일반화 cos.
     (token-level cue 투영은 Gemma 재-forward 필요 → 별도. 여기선 캐시 기반 2종.)

실행(추출 후, GPU 불필요):
  python steering/diagnostics.py
  python steering/diagnostics.py --layers 12
  python steering/diagnostics.py --self-test    # 합성 데이터로 로직 검증(캐시 불필요)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import gemma_common as G

FACETS = list(G.C.EXTRAVERSION_FACETS)
EPS = 1e-8


# --------------------------------------------------------------------------- #
# 벡터 구성 / 지표 (순수 numpy — self-test 로 검증)
# --------------------------------------------------------------------------- #
def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + EPS)


def _norm_rows(A: np.ndarray) -> np.ndarray:
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + EPS)


def build_V1(P: dict, N: dict):
    """예시별 L2 정규화 → facet별 diff → 균등가중. 반환 (V1, subvecs[6,H])."""
    ds = [(_norm_rows(P[f]).mean(0) - _norm_rows(N[f]).mean(0)) for f in FACETS]
    D = np.stack(ds)                       # [6,H] (정규화 하위벡터)
    return D.mean(0), D


def build_V2(P_all: np.ndarray, N_all: np.ndarray) -> np.ndarray:
    return P_all.mean(0) - N_all.mean(0)


def facet_contributions(P: dict, N: dict, V2: np.ndarray) -> dict:
    """개수 균등 전제: V2 = mean_f d_raw_f. c_f = (1/6)<d_raw_f,V2hat>/||V2||, 합=1."""
    v2n = np.linalg.norm(V2) + EPS
    v2hat = V2 / v2n
    w = 1.0 / len(FACETS)
    # 개수 균등: V2 = mean_f d_raw_f → c_f = w*<d_raw_f, v2hat> / ||V2||, 합=1.
    return {f: float(w * ((P[f].mean(0) - N[f].mean(0)) @ v2hat) / v2n) for f in FACETS}


def facet_cos_matrix(subvecs: np.ndarray) -> list[list[float]]:
    U = _norm_rows(subvecs)
    M = U @ U.T
    return [[round(float(M[i, j]), 3) for j in range(len(FACETS))] for i in range(len(FACETS))]


def svd_report(subvecs: np.ndarray) -> dict:
    # 공통 방향 강도(비-centering): 단일 외향성 방향이면 ↑ (plan '≳0.7' 기준 대응)
    s_r = np.linalg.svd(subvecs, compute_uv=False)
    var_r = s_r ** 2
    pc1_shared = float(var_r[0] / (var_r.sum() + EPS))
    # facet 간 잔차 spread(centering): 공통방향 제거 후 얼마나 흩어지나
    s_c = np.linalg.svd(subvecs - subvecs.mean(0, keepdims=True), compute_uv=False)
    var_c = s_c ** 2
    pc1_centered = float(var_c[0] / (var_c.sum() + EPS))
    part = float((var_c.sum() ** 2) / ((var_c ** 2).sum() + EPS))   # participation ratio
    return {"pc1_frac_shared": round(pc1_shared, 3),      # 공통 방향 비율(주 지표)
            "pc1_frac_centered": round(pc1_centered, 3),  # facet 간 잔차
            "participation_ratio": round(part, 2),
            "singular_values": [round(float(x), 4) for x in s_r]}


def leave_one_facet_out(P: dict, N: dict, V1: np.ndarray) -> dict:
    out = {}
    for drop in FACETS:
        ds = [(_norm_rows(P[f]).mean(0) - _norm_rows(N[f]).mean(0))
              for f in FACETS if f != drop]
        v = np.stack(ds).mean(0)
        out[drop] = round(float(_unit(v) @ _unit(V1)), 3)
    return out


def domain_generalization(P_rows, N_rows, facets, domains) -> dict:
    """3 도메인으로 만든 V 와 held-out 도메인 V 의 cos(트레이트 vs 템플릿 판별)."""
    doms = sorted(set(domains))
    facets = np.array(facets)
    domains = np.array(domains)
    out = {}
    for held in doms:
        tr = domains != held
        ho = domains == held

        def _v(selP, selN):
            ds = []
            for f in FACETS:
                mp = selP & (facets == f)
                mn = selN & (facets == f)
                if mp.sum() == 0 or mn.sum() == 0:
                    continue
                ds.append(_norm_rows(P_rows[mp]).mean(0) - _norm_rows(N_rows[mn]).mean(0))
            return np.stack(ds).mean(0) if ds else None
        # P_rows/N_rows 는 이미 pos/neg 로 분리된 행렬; sel 은 pair 단위 마스크
        vtr = _v(tr, tr)
        vho = _v(ho, ho)
        if vtr is not None and vho is not None:
            out[held] = round(float(_unit(vtr) @ _unit(vho)), 3)
    return out


# --------------------------------------------------------------------------- #
# 데이터 로딩 / 레이어 분석
# --------------------------------------------------------------------------- #
def load_cache(act_dir: Path):
    mp = act_dir / "pooled_mean.npy"
    ix = act_dir / "index.jsonl"
    if not mp.exists() or not ix.exists():
        sys.exit(f"[입력 누락] {act_dir} 에 pooled_mean.npy/index.jsonl 없음. "
                 f"먼저 extract_activations.py 실행.")
    acts = np.load(mp)                       # [2N,27,H]
    idx = [json.loads(l) for l in ix.open(encoding="utf-8") if l.strip()]
    return acts, idx


def split_by_facet(actL: np.ndarray, idx: list) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    """레이어 L 활성화[2N,H] → facet별 P/N dict + 전체 P/N 행렬(pair 정렬)."""
    facet = np.array([r["facet"] for r in idx])
    split = np.array([r["split"] for r in idx])
    P = {f: actL[(facet == f) & (split == "pos")] for f in FACETS}
    N = {f: actL[(facet == f) & (split == "neg")] for f in FACETS}
    P_all = actL[split == "pos"]
    N_all = actL[split == "neg"]
    return P, N, P_all, N_all


def analyze_layer(acts: np.ndarray, idx: list, L: int) -> dict:
    actL = acts[:, L, :].astype(np.float64)
    P, N, P_all, N_all = split_by_facet(actL, idx)
    V1, sub = build_V1(P, N)
    V2 = build_V2(P_all, N_all)
    # facet norm 분포
    norms = {}
    for f in FACETS:
        norms[f] = {"pos_mean": round(float(np.linalg.norm(P[f], axis=1).mean()), 2),
                    "neg_mean": round(float(np.linalg.norm(N[f], axis=1).mean()), 2)}
    nm = [norms[f]["pos_mean"] for f in FACETS]
    contrib = facet_contributions(P, N, V2)
    return {
        "layer": L,
        "cos_V1_V2": round(float(_unit(V1) @ _unit(V2)), 4),
        "facet_norm": norms,
        "facet_norm_ratio": round(max(nm) / (min(nm) + EPS), 3),
        "facet_contrib_to_V2": {k: round(v, 3) for k, v in contrib.items()},
        "max_contrib_facet": max(contrib, key=contrib.get),
        "facet_cos_matrix": {"keys": FACETS, "matrix": facet_cos_matrix(sub)},
        "svd": svd_report(sub),
        "leave_one_facet_out_cos": leave_one_facet_out(P, N, V1),
    }


# --------------------------------------------------------------------------- #
def self_test():
    """합성 데이터로 순수-numpy 로직 검증(캐시 불필요)."""
    rng = np.random.default_rng(0)
    H = 16
    trait = rng.standard_normal(H)
    idx, rows = [], []
    for f in FACETS:
        for s in ("pos", "neg"):
            for _ in range(20):
                base = rng.standard_normal(H) * 0.5
                v = base + (1 if s == "pos" else -1) * trait
                rows.append(v)
                idx.append({"facet": f, "split": s, "domain": "daily"})
    acts = np.stack(rows)[:, None, :]        # [2N,1,H]
    r = analyze_layer(acts, idx, 0)
    cs = sum(r["facet_contrib_to_V2"].values())
    print("self-test cos(V1,V2)=", r["cos_V1_V2"],
          "| contrib 합=", round(cs, 3), "(≈1 기대)",
          "| PC1_shared=", r["svd"]["pc1_frac_shared"], "(공통 trait → ↑ 기대)")
    assert abs(cs - 1.0) < 1e-3, "기여도 합 != 1"
    assert r["cos_V1_V2"] > 0.9, "공통 trait 인데 V1/V2 불일치"
    print("SELF-TEST OK")


def parse_args():
    ap = argparse.ArgumentParser(description="모듈 4a — CAA 벡터 진단(V1 vs V2)")
    ap.add_argument("--act-dir", default=str(G.ACT_DIR))
    ap.add_argument("--layers", default=None, help="쉼표구분 레이어 또는 미지정 시 SWEEP")
    ap.add_argument("--out", default=str(G.VEC_DIR / "diagnostics.json"))
    ap.add_argument("--self-test", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    acts, idx = load_cache(Path(args.act_dir))
    # 레이어는 캐시 배열 모양에서 모델-적응형으로 (2B=27→[5,9,12,15,19], 9B=43→중간 밴드)
    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else G.sweep_band(acts.shape[1]))
    report = {"n_pairs": len(idx) // 2, "layers": {}}
    for L in layers:
        r = analyze_layer(acts, idx, L)
        report["layers"][str(L)] = r
        print(f"[layer {L:>2}] cos(V1,V2)={r['cos_V1_V2']:.3f} "
              f"norm비={r['facet_norm_ratio']:.2f} "
              f"최대기여={r['max_contrib_facet']}({r['facet_contrib_to_V2'][r['max_contrib_facet']]:.2f}) "
              f"PC1_shared={r['svd']['pc1_frac_shared']:.2f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n완료 → {args.out}")
    # 권고
    best = min(report["layers"].values(), key=lambda r: abs(r["cos_V1_V2"] - 1))
    print(f"참고: cos(V1,V2) 최소={min(r['cos_V1_V2'] for r in report['layers'].values()):.3f} "
          f"— <0.9 인 레이어는 4b 에서 V1·V2 둘 다 평가.")


if __name__ == "__main__":
    main()
