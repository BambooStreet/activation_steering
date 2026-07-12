"""[모듈 3+ 공통] Gemma 2 2B 로딩·활성화 풀링·스티어링 주입 유틸.

extract_activations(추출) / diagnostics(4a) / steer_eval(4b) / sae_gemmascope(5) 가 공유.
모듈 1~2 의 common.py 스타일 계승(상수·경로 재사용). 무거운 deps(torch/transformers)는
클라우드 GPU 전용 — requirements-steering.txt 참고.

검증된 Gemma 2 2B 사실 (arXiv:2408.00118 / HF google/gemma-2-2b):
  - 26 decoder layers, hidden_size=2304, vocab 256000.
  - output_hidden_states → 길이 27 튜플: idx0=임베딩 출력, idx L+1 = 0-indexed layer L 이후 residual.
    (그래서 layer L 추출 == layer L 주입 정합: model.model.layers[L] 훅이 hidden_states[L+1]을 수정)
  - soft-capping 보존 위해 attn_implementation="eager", bf16 권장.
  - 게이트 모델 → HF 라이선스 수락 + HF_TOKEN 필요.
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv

# torch/transformers 는 GPU 런타임에만 필요 → 함수 내부 lazy import.
# (diagnostics.py 등 순수 numpy 분석은 torch 없이 이 모듈의 상수만 재사용)

# 저장소 루트를 path 에 넣어 모듈 1 common.py 의 상수 재사용
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import common as C  # noqa: E402  (EXTRAVERSION_FACETS, DOMAINS, OUT_DIR)

ARTIFACTS_DIR = ROOT / "artifacts"
ACT_DIR = ARTIFACTS_DIR / "activations"
VEC_DIR = ARTIFACTS_DIR / "vectors"

# 모델/레이어 기본값 (env 로 오버라이드 가능)
# -it(instruct): 설문 자기보고(게이트 A)를 실제로 따르고 멀티턴 대화에 자연스러움.
# base 로 벡터 방향은 더 깨끗하나 2B base 는 Likert 자기보고가 평평(측정 실패) → -it 채택.
MODEL_ID = os.getenv("STEER_MODEL", "google/gemma-2-2b-it")


def is_chat_model(model_id: str) -> bool:
    """instruct/chat 모델이면 True → 프롬프트에 chat 템플릿 적용."""
    return "-it" in model_id.lower() or "instruct" in model_id.lower()
# 2B 기본값(폴백). 실제 차원/레이어는 모델 config·캐시 배열에서 자동 결정(아래 헬퍼).
N_LAYERS = 26
HIDDEN = 2304
N_HIDDEN_STATES = N_LAYERS + 1                              # =27 (임베딩 포함)
PRIMARY_LAYER = 12                                         # middle; Gemma Scope 폭 스윕 존재
SWEEP_LAYERS = [5, 8, 12, 16, 19]


def mid_layer(n_hidden_states: int) -> int:
    """모델 크기에 비례한 중간 decoder layer (2B: 27→12 검증치 기준). env STEER_LAYER 우선."""
    e = os.getenv("STEER_LAYER")
    if e:
        return int(e)
    return max(1, min(n_hidden_states - 1, round((n_hidden_states - 1) * 12 / 26)))


def sweep_band(n_hidden_states: int) -> list:
    """중간층 주변 스윕 밴드 (2B→[5,9,12,15,19] 유사)."""
    m = mid_layer(n_hidden_states)
    span = max(1, round((n_hidden_states - 1) * 7 / 26))
    return sorted({max(1, min(n_hidden_states - 1, m + d))
                   for d in (-span, -span // 2, 0, span // 2, span)})


# --------------------------------------------------------------------------- #
# 환경 / 모델 로딩
# --------------------------------------------------------------------------- #
def hf_token() -> str | None:
    load_dotenv(ROOT / ".env")
    return os.getenv("HF_TOKEN")


def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_id: str = MODEL_ID, device: str | None = None, dtype=None,
               load_4bit: bool | None = None):
    """(model, tokenizer) 로드. eval + no_grad 는 호출측 책임.

    right-padding(BOS=col0 고정 → 풀링에서 BOS 제외 단순화). 게이트 모델이면 HF_TOKEN 사용.
    load_4bit(또는 env STEER_4BIT=1): 큰 모델을 무료 T4(16GB)에 4-bit 양자화로 적재.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = dtype or torch.bfloat16
    device = device or pick_device()
    if load_4bit is None:
        load_4bit = os.getenv("STEER_4BIT", "").lower() in ("1", "true", "yes")
    tok = AutoTokenizer.from_pretrained(model_id, token=hf_token())
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    kwargs = dict(token=hf_token(), attn_implementation="eager")  # Gemma2 soft-capping 보존
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
        kwargs["device_map"] = "auto"
    else:
        kwargs["dtype"] = dtype                 # torch_dtype 은 최신 transformers 에서 deprecated
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if not load_4bit:
        model = model.to(device)
    model.eval()
    return model, tok


# --------------------------------------------------------------------------- #
# 활성화 풀링 (추출: 읽기)
# --------------------------------------------------------------------------- #
def pooled_hidden(model, tok, texts: list[str], device: str, max_len: int = 64):
    """texts 배치 → (mean_pooled, last_pooled), 각 [B, 27, HIDDEN] (fp32, CPU).

    mean: BOS(있으면)·pad 제외 토큰 평균. last: 마지막 실토큰. right-padding 전제.
    no-BOS 토크나이저(Qwen2.5 등)는 col0=첫 실토큰이므로 제외하지 않음.
    """
    import torch
    with torch.no_grad():
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states                       # 27 x [B, S, H]
        mask = enc["attention_mask"]                 # [B, S] (1=real)
        fmask = mask.to(hs[0].dtype)
        m2 = fmask.clone()
        if tok.bos_token_id is not None:             # BOS 방출 모델(Gemma/Mistral)만 col0 제외.
            m2[enc["input_ids"][:, 0] == tok.bos_token_id, 0] = 0   # Qwen2.5(no-BOS)는 첫 실토큰 보존
        denom = m2.sum(1).clamp(min=1.0)             # [B]
        last_idx = (mask.sum(1).long() - 1).clamp(min=0)   # [B]
        ar = torch.arange(hs[0].size(0), device=device)
        means, lasts = [], []
        for h in hs:                                 # h: [B, S, H]
            means.append((h * m2.unsqueeze(-1)).sum(1) / denom.unsqueeze(-1))
            lasts.append(h[ar, last_idx])
        mean_p = torch.stack(means, dim=1).float().cpu()    # [B, 27, H]
        last_p = torch.stack(lasts, dim=1).float().cpu()
    return mean_p, last_p


# --------------------------------------------------------------------------- #
# 스티어링 주입 (4b: 쓰기) — forward hook
# --------------------------------------------------------------------------- #
@contextmanager
def steering_hook(model, layer_idx: int, vector, alpha: float):
    """model.model.layers[layer_idx] 출력 residual 에 alpha*vector 더함(모든 위치).

    layer_idx(0-indexed)는 hidden_states[layer_idx+1] 과 정합. 양방향: alpha 부호로.
    """
    import torch
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    v = torch.as_tensor(vector, dtype=dt, device=dev).reshape(-1)

    def hook(_module, _inp, out):
        # Gemma2 decoder layer 는 튜플 반환; out[0] 이 hidden [B,S,H]
        h = out[0] if isinstance(out, tuple) else out
        h.add_(alpha * v)
        return out

    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def layer_residual_norms(model, tok, texts: list[str], device: str, max_len: int = 64):
    """레이어별 평균 토큰 residual L2 norm [27] — 4b 의 α 스케일(α=c·R) 산정용."""
    mean_p, _ = pooled_hidden(model, tok, texts, device, max_len)  # [B,27,H]
    return mean_p.norm(dim=-1).mean(dim=0)                          # [27]
