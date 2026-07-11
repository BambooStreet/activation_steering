#!/usr/bin/env python3
"""[모듈 1 · 스텝 1] 시나리오 풀 임베딩 재검증 (코사인 기반, Jaccard 폐지)

0단계 dedup 은 토큰 Jaccard(어휘적)라 **말바꾸기 근사중복**을 놓친다. 여기서는
OpenAI 임베딩(text-embedding-3-small)으로 의미 수준 유사도를 재검증한다.

점검 항목
  1. 셀((facet,domain)) 내 코사인 근사중복(≥ threshold) — Jaccard 가 놓친 재탕 포착.
  2. 같은 facet · 다른 도메인 교차 근사중복(도메인만 바꾼 재탕).
  3. 전체 풀 근사중복(헤드라인 수치).
  4. 셀별 의미 다양성(평균 pairwise 코사인 — 높을수록 셀이 몰려 있음).
  5. 의미 편향/교락: 도메인 중심·facet 중심 코사인 행렬 + 최근접-중심 분류 정확도.
     (개수는 이미 셀당 62로 완전 균등 → 여기서 보는 건 *의미적* 편향.)

입력:  data/scenarios_pool.json
출력:  outputs/pool_revalidation.json (근사중복 쌍·셀 다양성·편향 지표)
캐시:  outputs/pool_embeddings.npz (임베딩 재사용 — 입력 변하면 자동 무효화)

제거/수정은 하지 않는다(사람 검토용 리포트). 실행:
  python revalidate_pool.py                     # 셀 내 근사중복(임계 0.85)
  python revalidate_pool.py --threshold 0.88 --scope facet
  python revalidate_pool.py --no-cache          # 임베딩 강제 재계산
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))  # module1/ → 루트(common.py)
import common as C

POOL_PATH = C.DATA_DIR / "scenarios_pool.json"
EMB_CACHE = C.OUT_DIR / "pool_embeddings.npz"
DEFAULT_OUT = C.OUT_DIR / "pool_revalidation.json"


# --------------------------------------------------------------------------- #
def load_pool(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"[입력 누락] {path} 없음. 0단계(generate_scenarios.py)를 먼저 실행.")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw.get("scenarios", [])


def pool_hash(scenarios: list[str], model: str) -> str:
    h = hashlib.sha256(model.encode("utf-8"))
    for s in scenarios:
        h.update(b"\x00")
        h.update(s.encode("utf-8"))
    return h.hexdigest()


def get_embeddings(client, scenarios: list[str], model: str,
                   use_cache: bool) -> np.ndarray:
    """임베딩 행렬 [N, d] (L2 정규화). 캐시 유효하면 재사용."""
    key = pool_hash(scenarios, model)
    if use_cache and EMB_CACHE.exists():
        cached = np.load(EMB_CACHE, allow_pickle=False)
        if str(cached["key"]) == key:
            print(f"[캐시] {EMB_CACHE.name} 재사용({len(scenarios)}개, {model})")
            return cached["emb"]
        print("[캐시] 입력/모델 변경 감지 → 재임베딩")
    print(f"[임베딩] {len(scenarios)}개 → {model} 호출 중…", flush=True)
    vecs = C.embed_texts(client, scenarios, model=model)
    emb = np.asarray(vecs, dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12  # L2 정규화 → 내적=코사인
    C.OUT_DIR.mkdir(exist_ok=True)
    np.savez(EMB_CACHE, emb=emb, key=key)
    print(f"[캐시] 저장 → {EMB_CACHE}")
    return emb


# --------------------------------------------------------------------------- #
def near_dups(idx: np.ndarray, emb: np.ndarray, thresh: float) -> list[tuple]:
    """idx 부분집합 내 코사인 ≥ thresh 인 (i,j,sim) 목록(전역 인덱스)."""
    if len(idx) < 2:
        return []
    sub = emb[idx]
    sim = sub @ sub.T
    iu, ju = np.triu_indices(len(idx), k=1)
    mask = sim[iu, ju] >= thresh
    return [(int(idx[iu[k]]), int(idx[ju[k]]), float(sim[iu[k], ju[k]]))
            for k in np.nonzero(mask)[0]]


def cell_diversity(idx: np.ndarray, emb: np.ndarray) -> float:
    """셀 내 평균 pairwise 코사인(높을수록 의미가 몰려 있음 = 저다양성)."""
    if len(idx) < 2:
        return 0.0
    sub = emb[idx]
    sim = sub @ sub.T
    iu, ju = np.triu_indices(len(idx), k=1)
    return float(sim[iu, ju].mean())


def centroids(emb: np.ndarray, labels: list[str], keys: list[str]) -> dict:
    """라벨별 정규화 중심 벡터."""
    out = {}
    lab = np.array(labels)
    for k in keys:
        m = emb[lab == k].mean(axis=0)
        out[k] = m / (np.linalg.norm(m) + 1e-12)
    return out


def cos_matrix(cents: dict, keys: list[str]) -> list[list[float]]:
    return [[round(float(cents[a] @ cents[b]), 3) for b in keys] for a in keys]


def nearest_centroid_acc(emb: np.ndarray, labels: list[str],
                         cents: dict, keys: list[str]) -> float:
    M = np.stack([cents[k] for k in keys])          # [K, d]
    pred = np.array(keys)[(emb @ M.T).argmax(axis=1)]
    return float((pred == np.array(labels)).mean())


# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description="모듈 1 · 스텝 1 — 시나리오 풀 임베딩 재검증")
    ap.add_argument("--threshold", type=float, default=0.85,
                    help="근사중복 코사인 임계(기본 0.85)")
    ap.add_argument("--scope", choices=["cell", "facet", "global"], default="cell",
                    help="근사중복 탐색 범위(기본 cell)")
    ap.add_argument("--top-dups", type=int, default=40, help="콘솔에 출력할 근사중복 수")
    ap.add_argument("--embed-model", default="text-embedding-3-small")
    ap.add_argument("--no-cache", action="store_true", help="임베딩 캐시 무시하고 재계산")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    return ap.parse_args()


def main():
    args = parse_args()
    api_key, _ = C.load_env_and_model()
    client = OpenAI(api_key=api_key)

    pool = load_pool(POOL_PATH)
    scenarios = [s["scenario"] for s in pool]
    facets = [s["facet"] for s in pool]
    domains = [s["domain"] for s in pool]
    print(f"풀 {len(pool)}개 · facets {sorted(set(facets))} · domains {sorted(set(domains))}")

    emb = get_embeddings(client, scenarios, args.embed_model, use_cache=not args.no_cache)

    # --- 근사중복(scope 별) ---------------------------------------------- #
    all_idx = np.arange(len(pool))
    groups: dict[str, np.ndarray] = {}
    if args.scope == "cell":
        for f in sorted(set(facets)):
            for d in sorted(set(domains)):
                sel = all_idx[(np.array(facets) == f) & (np.array(domains) == d)]
                groups[f"{f}×{d}"] = sel
    elif args.scope == "facet":
        for f in sorted(set(facets)):
            groups[f] = all_idx[np.array(facets) == f]
    else:  # global
        groups["__all__"] = all_idx

    dup_records = []
    for gname, gidx in groups.items():
        for i, j, sim in near_dups(gidx, emb, args.threshold):
            dup_records.append({
                "group": gname, "sim": round(sim, 4),
                "i": i, "j": j,
                "facet_i": facets[i], "domain_i": domains[i], "scenario_i": scenarios[i],
                "facet_j": facets[j], "domain_j": domains[j], "scenario_j": scenarios[j],
            })
    dup_records.sort(key=lambda r: -r["sim"])

    # --- 셀 다양성(항상 셀 단위로 계산) ---------------------------------- #
    cell_div = []
    for f in sorted(set(facets)):
        for d in sorted(set(domains)):
            sel = all_idx[(np.array(facets) == f) & (np.array(domains) == d)]
            cell_div.append({"cell": f"{f}×{d}", "n": int(len(sel)),
                             "mean_cos": round(cell_diversity(sel, emb), 4)})
    cell_div.sort(key=lambda r: -r["mean_cos"])   # 저다양성(몰림) 먼저

    # --- 의미 편향/교락 ------------------------------------------------- #
    dom_keys = sorted(set(domains))
    fac_keys = sorted(set(facets))
    dom_cent = centroids(emb, domains, dom_keys)
    fac_cent = centroids(emb, facets, fac_keys)
    bias = {
        "domain_centroid_cos": {"keys": dom_keys, "matrix": cos_matrix(dom_cent, dom_keys)},
        "facet_centroid_cos": {"keys": fac_keys, "matrix": cos_matrix(fac_cent, fac_keys)},
        "nearest_centroid_domain_acc": round(nearest_centroid_acc(emb, domains, dom_cent, dom_keys), 3),
        "nearest_centroid_facet_acc": round(nearest_centroid_acc(emb, facets, fac_cent, fac_keys), 3),
    }
    # facet 중심 최고 교락쌍(대각 제외)
    fm = np.array(bias["facet_centroid_cos"]["matrix"])
    np.fill_diagonal(fm, -1)
    fi, fj = np.unravel_index(fm.argmax(), fm.shape)
    top_facet_entangle = {"a": fac_keys[fi], "b": fac_keys[fj], "cos": round(float(fm[fi, fj]), 3)}

    # --- 리포트 저장 ---------------------------------------------------- #
    report = {
        "n_scenarios": len(pool), "embed_model": args.embed_model,
        "threshold": args.threshold, "scope": args.scope,
        "n_near_dups": len(dup_records),
        "near_dups": dup_records,
        "cell_diversity": cell_div,
        "semantic_bias": bias,
        "top_facet_entanglement": top_facet_entangle,
    }
    out_path = Path(args.out)
    C.OUT_DIR.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- 콘솔 요약 ------------------------------------------------------ #
    print(f"\n=== 근사중복(scope={args.scope}, ≥{args.threshold}): {len(dup_records)}쌍 ===")
    for r in dup_records[:args.top_dups]:
        print(f"  {r['sim']:.3f} [{r['group']}]")
        print(f"      i: {r['scenario_i']}")
        print(f"      j: {r['scenario_j']}")
    if len(dup_records) > args.top_dups:
        print(f"  … 외 {len(dup_records) - args.top_dups}쌍 (전체는 {out_path.name})")

    print(f"\n=== 저다양성 셀 TOP5(평균 코사인↑) ===")
    for r in cell_div[:5]:
        print(f"  {r['mean_cos']:.3f}  {r['cell']} (n={r['n']})")

    print(f"\n=== 의미 편향 ===")
    print(f"  최근접-중심 도메인 분류 정확도: {bias['nearest_centroid_domain_acc']} "
          f"(높을수록 도메인 의미 구분 뚜렷)")
    print(f"  최근접-중심 facet 분류 정확도: {bias['nearest_centroid_facet_acc']} "
          f"(낮으면 facet 의미 얽힘 → 누수 위험)")
    print(f"  최고 facet 얽힘쌍: {top_facet_entangle['a']} ↔ {top_facet_entangle['b']} "
          f"cos={top_facet_entangle['cos']}")
    print(f"\n완료 → {out_path}")


if __name__ == "__main__":
    try:
        main()
    except C.QuotaExhausted as e:
        print(f"\n{e}", flush=True)
        sys.exit(2)
