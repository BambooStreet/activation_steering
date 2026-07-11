"""모듈 1 공통 유틸 — 환경/모델 로딩, JSON 호출, 텍스트 유사도.

generate_scenarios.py(0단계)와 generate_pairs.py(1단계)가 공유한다.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import openai
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

# 외향성 6 facet (NEO-PI). 정의/cues 는 facets_en.json 이 채운다(여기서 발명 안 함).
EXTRAVERSION_FACETS = [
    "warmth",
    "gregariousness",
    "assertiveness",
    "activity",
    "excitement_seeking",
    "positive_emotions",
]

# 캐논 도메인 키(기존 scenarios_en.json 과 동일) → 프롬프트용 설명.
DOMAINS = {
    "study": "STUDY (school, classes, coursework, study groups, exams)",
    "work": "WORK (workplace, job tasks, meetings, colleagues, clients)",
    "daily": "DAILY LIFE (home, errands, family, neighbors, routine day-to-day)",
    "leisure": "SOCIAL & LEISURE (free time, hobbies, parties, friends, outings)",
}


# --------------------------------------------------------------------------- #
# 환경 / 모델
# --------------------------------------------------------------------------- #
def load_env_and_model(model_override: str | None = None) -> tuple[str, str]:
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("[설정 누락] OPENAI_API_KEY 가 .env 에 없습니다. (.env.example 참고)")
    model = model_override or os.getenv("GEN_MODEL", "gpt-4o")
    if "mini" in model.lower():
        # 스펙 기본은 mini 금지(minimal-edit 미준수 우려). 단 경험적 검증을 위해 허용은 함.
        print(f"[경고] mini 모델 사용: {model} — minimal-edit/극성 품질을 반드시 검증할 것.",
              flush=True)
    return api_key, model


def supports_custom_temperature(model: str) -> bool:
    """gpt-5 계열은 temperature 커스텀값을 거부(기본 1만 허용)."""
    return not model.lower().startswith("gpt-5")


class QuotaExhausted(RuntimeError):
    """OpenAI 크레딧/쿼터 소진(insufficient_quota) — 재시도 불가, 결제 확인 필요."""


_BACKOFFS = [5, 15, 45, 90]


def chat_json(client: OpenAI, model: str, prompt: str, temperature: float,
              system: str = "You output only valid JSON matching the requested schema.",
              max_parse_retries: int = 2, max_api_retries: int = 4) -> dict:
    """JSON 객체 요청+파싱. gpt-5 계열엔 temperature 미전송.

    - 일시적 API 오류(rate limit / timeout / 연결 / 5xx)는 지수 백오프 재시도.
    - insufficient_quota(크레딧 소진)는 QuotaExhausted 로 즉시 중단(재시도 무의미).
    - JSON 파싱 실패는 별도 재시도.
    """
    kwargs = {}
    if supports_custom_temperature(model):
        kwargs["temperature"] = temperature
    api_attempt = 0
    parse_attempt = 0
    while True:
        try:
            resp = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": prompt}],
                **kwargs,
            )
        except openai.RateLimitError as e:
            msg = str(e)
            if "insufficient_quota" in msg or "exceeded your current quota" in msg:
                raise QuotaExhausted(
                    "[OpenAI 쿼터 소진] insufficient_quota — 결제/크레딧을 확인하거나 "
                    "다른 키로 교체하세요. (.env OPENAI_API_KEY)") from None
            if api_attempt >= max_api_retries:
                raise
            wait = _BACKOFFS[min(api_attempt, len(_BACKOFFS) - 1)]
            print(f"    ⏳ rate limit — {wait}s 대기 후 재시도({api_attempt + 1})", flush=True)
            time.sleep(wait)
            api_attempt += 1
            continue
        except (openai.APITimeoutError, openai.APIConnectionError,
                openai.InternalServerError) as e:
            if api_attempt >= max_api_retries:
                raise
            wait = _BACKOFFS[min(api_attempt, len(_BACKOFFS) - 1)]
            print(f"    ⏳ {type(e).__name__} — {wait}s 대기 후 재시도({api_attempt + 1})",
                  flush=True)
            time.sleep(wait)
            api_attempt += 1
            continue

        try:
            return json.loads(resp.choices[0].message.content)
        except json.JSONDecodeError as e:
            parse_attempt += 1
            if parse_attempt > max_parse_retries:
                raise RuntimeError(f"JSON 파싱 실패(재시도 {max_parse_retries}회 후): {e}")


def temp_label(model: str, temperature: float) -> str:
    return f"{temperature}" if supports_custom_temperature(model) else "기본(1, 고정)"


def embed_texts(client: OpenAI, texts: list[str],
                model: str = "text-embedding-3-small", batch_size: int = 256,
                max_api_retries: int = 4) -> list[list[float]]:
    """텍스트 리스트 → 임베딩 벡터 리스트(입력 순서 유지).

    chat_json 과 동일 견고성: rate limit/timeout/5xx 지수 백오프 재시도,
    insufficient_quota 는 QuotaExhausted 로 즉시 중단. numpy 비의존(순수 list 반환).
    """
    out: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = [t if (t and t.strip()) else " " for t in texts[start:start + batch_size]]
        attempt = 0
        while True:
            try:
                resp = client.embeddings.create(model=model, input=batch)
                break
            except openai.RateLimitError as e:
                msg = str(e)
                if "insufficient_quota" in msg or "exceeded your current quota" in msg:
                    raise QuotaExhausted(
                        "[OpenAI 쿼터 소진] insufficient_quota — 결제/크레딧 확인.") from None
                if attempt >= max_api_retries:
                    raise
                wait = _BACKOFFS[min(attempt, len(_BACKOFFS) - 1)]
                print(f"    ⏳ rate limit(embed) — {wait}s 후 재시도({attempt + 1})", flush=True)
                time.sleep(wait)
                attempt += 1
            except (openai.APITimeoutError, openai.APIConnectionError,
                    openai.InternalServerError) as e:
                if attempt >= max_api_retries:
                    raise
                wait = _BACKOFFS[min(attempt, len(_BACKOFFS) - 1)]
                print(f"    ⏳ {type(e).__name__}(embed) — {wait}s 후 재시도({attempt + 1})",
                      flush=True)
                time.sleep(wait)
                attempt += 1
        out.extend(d.embedding for d in resp.data)
    return out


# --------------------------------------------------------------------------- #
# Anthropic (Claude) 판정 — 교차 검증용(생성=GPT ↔ 판정=Claude)
# --------------------------------------------------------------------------- #
def anthropic_json(client, model: str, prompt: str, schema: dict,
                   system: str = "You output only valid JSON matching the requested schema.",
                   max_api_retries: int = 4) -> dict:
    """Claude로 JSON 판정. output_config.format(json_schema)로 유효 JSON 강제.

    - Opus 4.8/4.7·Sonnet 5 계열: temperature 미전송(400 방지), thinking 생략(미사용 실행).
    - rate limit/timeout/5xx 지수 백오프 재시도, 크레딧 소진은 QuotaExhausted.
    - anthropic SDK는 함수 내부에서 lazy import(모듈1 OpenAI 전용 환경 불변).
    """
    import anthropic  # lazy: 미설치 환경에서도 common import 가능
    kwargs = {
        "model": model, "max_tokens": 1024, "system": system,
        "messages": [{"role": "user", "content": prompt}],
        "output_config": {"format": {"type": "json_schema", "schema": schema}},
    }
    attempt = 0
    while True:
        try:
            resp = client.messages.create(**kwargs)
        except anthropic.RateLimitError as e:
            msg = str(e).lower()
            if "credit" in msg or "balance" in msg or "quota" in msg:
                raise QuotaExhausted(
                    "[Anthropic 크레딧 소진] 결제/크레딧 확인. (.env ANTHROPIC_API_KEY)") from None

            if attempt >= max_api_retries:
                raise
            wait = _BACKOFFS[min(attempt, len(_BACKOFFS) - 1)]
            print(f"    ⏳ rate limit(claude) — {wait}s 후 재시도({attempt + 1})", flush=True)
            time.sleep(wait)
            attempt += 1
            continue
        except (anthropic.APIConnectionError, anthropic.APITimeoutError,
                anthropic.InternalServerError) as e:
            if attempt >= max_api_retries:
                raise
            wait = _BACKOFFS[min(attempt, len(_BACKOFFS) - 1)]
            print(f"    ⏳ {type(e).__name__}(claude) — {wait}s 후 재시도({attempt + 1})",
                  flush=True)
            time.sleep(wait)
            attempt += 1
            continue
        except anthropic.APIStatusError as e:
            code = getattr(e, "status_code", None)
            msg = str(e).lower()
            if "credit" in msg or "balance" in msg:
                raise QuotaExhausted(
                    "[Anthropic 크레딧 소진] 결제/크레딧 확인. (.env ANTHROPIC_API_KEY)") from None
            if code and code >= 500 and attempt < max_api_retries:
                wait = _BACKOFFS[min(attempt, len(_BACKOFFS) - 1)]
                time.sleep(wait)
                attempt += 1
                continue
            raise
        # output_config.format → 첫 text 블록이 유효 JSON
        text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), "")
        return json.loads(text)


# --------------------------------------------------------------------------- #
# 입력 로딩
# --------------------------------------------------------------------------- #
def load_facets() -> dict:
    """facets_en.json → {facet: {definition, cues:{high,low}}}."""
    p = DATA_DIR / "facets_en.json"
    if not p.exists():
        sys.exit(f"[입력 누락] {p} 없음.")
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {f["facet"]: f for f in raw["facets"]}


def load_seed_scenarios() -> dict:
    """scenarios_en.json(사람 시드) → {(facet,domain): [scenario, ...]}."""
    p = DATA_DIR / "scenarios_en.json"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    out: dict = {}
    for s in raw["scenarios"]:
        out.setdefault((s["facet"], s["domain"]), []).append(s["scenario"])
    return out


# --------------------------------------------------------------------------- #
# 텍스트 유사도 / 정규화
# --------------------------------------------------------------------------- #
def toks(s: str) -> set:
    return set(re.findall(r"[a-z0-9']+", (s or "").lower()))


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 0.0


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", (s or "").lower())).strip()


def word_count(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s or ""))


def fmt_cues(cues_val) -> str:
    if isinstance(cues_val, (list, tuple)):
        return "; ".join(str(c) for c in cues_val)
    return str(cues_val)


def split_cues(cues_val) -> list[str]:
    """cues 문자열("a, b, c.")을 개별 행동 단서 리스트로 분해."""
    if isinstance(cues_val, (list, tuple)):
        items = list(cues_val)
    else:
        items = re.split(r"[;,]", str(cues_val))
    return [c.strip().rstrip(".").strip() for c in items if c.strip()]
