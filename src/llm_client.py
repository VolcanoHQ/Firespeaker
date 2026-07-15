#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Firespeaker LLM Client
Single chokepoint for all LLM backend access: Gemini Flash (free tier) ->
Groq Llama-3.1-8B-Instant (free tier) -> local Ollama -> None.

Tier 1 ingestion (src/tier_1_parser.py) never imports this module unless a
caller explicitly opts in to LLM enrichment. When FIRESPEAKER_LLM_ENRICHMENT=off
(or no keys/backends are reachable), query_llm_json() returns (None, None) and
callers fall back to their existing non-LLM heuristics.
"""

import os
import re
import json
import time
import logging
import urllib.request
import urllib.error
from datetime import date
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel

logger = logging.getLogger("LLMClient")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV_PATH = os.path.join(REPO_ROOT, ".env")
USAGE_STATE_PATH = os.path.join(REPO_ROOT, "data", "llm_usage_state.json")
AUDIT_LOG_PATH = os.path.join(REPO_ROOT, "data", "llm_call_audit.jsonl")

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=ENV_PATH)
except ImportError:
    logger.warning("python-dotenv not installed; relying on process environment only.")

# Free-tier limits as of implementation time; re-verify against provider
# dashboards periodically, these change without notice.
# GEMINI_RPM confirmed live via a real 429 response body on 2026-07-03:
# "limit: 5, model: gemini-2.5-flash" (quotaId GenerateRequestsPerMinutePerProjectPerModel-FreeTier)
GEMINI_RPM = 5
GEMINI_RPD = 250
GROQ_RPM = 30
GROQ_RPD = 14400

OLLAMA_MODEL_PREFERENCE_PATTERNS = [
    r"^llama3\.3",
    r"^llama3\.2",
    r"^llama3\.1",
    r"^llama3",
    r"^mistral",
    r"^qwen2\.5.*instruct",
    r"^qwen2\.5",
    r"^qwen2",
]


def _list_local_ollama_models(tags_url: str = "http://localhost:11434/api/tags") -> List[str]:
    try:
        res = urllib.request.urlopen(tags_url, timeout=1.0)
        if res.status != 200:
            return []
        payload = json.loads(res.read().decode("utf-8"))
        return [model.get("name", "") for model in payload.get("models", []) if model.get("name")]
    except Exception:
        return []


def _select_preferred_ollama_model(available_models: List[str], requested_model: Optional[str] = None) -> Optional[str]:
    if requested_model:
        for model in available_models:
            if model == requested_model:
                return model
        logger.warning(f"Requested Ollama model '{requested_model}' is not installed. Available models: {available_models}")

    env_override = os.getenv("FIRESPEAKER_OLLAMA_MODEL")
    if env_override:
        for model in available_models:
            if model == env_override:
                return model
        logger.warning(f"FIRESPEAKER_OLLAMA_MODEL='{env_override}' is not installed. Available models: {available_models}")

    lowered_models = [(model, model.lower()) for model in available_models]
    for pattern in OLLAMA_MODEL_PREFERENCE_PATTERNS:
        for original_model, lowered_model in lowered_models:
            if re.search(pattern, lowered_model):
                return original_model

    return available_models[0] if available_models else None


def _is_enrichment_enabled() -> bool:
    flag = os.getenv("FIRESPEAKER_LLM_ENRICHMENT", "on").strip().lower()
    return flag not in ("off", "0", "false", "no")


def _today() -> str:
    return date.today().isoformat()


def _load_usage_state() -> Dict[str, Any]:
    try:
        with open(USAGE_STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_usage_state(state: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(USAGE_STATE_PATH), exist_ok=True)
        with open(USAGE_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not persist LLM usage state: {e}")


def _append_audit_log(record: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as e:
        logger.warning(f"Could not append LLM audit log entry: {e}")


def _extract_json_object(text: str) -> Optional[dict]:
    """Robustly extract a JSON object/array from a possibly markdown-fenced LLM response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace:last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None
    return None


class _QuotaTracker:
    """Tracks per-provider daily request counts and minimum-interval RPM pacing.

    Pacing timestamps use wall-clock time.time(), NOT time.monotonic() --
    monotonic values are not comparable across processes, and this state file
    persists between runs.
    """

    # Pace at 20% under the advertised RPM cap: riding a per-minute limit exactly
    # at its boundary (e.g. 12s spacing against a 5 RPM cap) still trips 429s due
    # to provider-side window drift.
    RPM_SAFETY_MARGIN = 1.2

    def __init__(self):
        self.state = _load_usage_state()

    def _provider_entry(self, provider: str) -> Dict[str, Any]:
        entry = self.state.get(provider, {})
        if entry.get("date") != _today():
            entry = {"date": _today(), "requests_today": 0, "last_call_ts": 0.0}
        self.state[provider] = entry
        return entry

    def is_exhausted(self, provider: str, rpd: int) -> bool:
        entry = self._provider_entry(provider)
        return entry["requests_today"] >= rpd

    def wait_for_rpm(self, provider: str, rpm: int) -> None:
        entry = self._provider_entry(provider)
        min_interval = (60.0 / max(rpm, 1)) * self.RPM_SAFETY_MARGIN
        elapsed = time.time() - entry.get("last_call_ts", 0.0)
        if 0 <= elapsed < min_interval:
            time.sleep(min_interval - elapsed)

    def record_call(self, provider: str) -> None:
        entry = self._provider_entry(provider)
        entry["requests_today"] += 1
        entry["last_call_ts"] = time.time()
        self.state[provider] = entry
        _save_usage_state(self.state)


class LLMClient:
    """Provider-abstracted JSON-mode LLM client with a Gemini -> Groq -> Ollama -> None fallback chain."""

    GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash"]
    GROQ_MODEL = "llama-3.1-8b-instant"

    def __init__(self, enabled: Optional[bool] = None, ollama_url: str = "http://localhost:11434/api/generate", ollama_model: Optional[str] = None):
        self.enabled = _is_enrichment_enabled() if enabled is None else enabled
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.quota = _QuotaTracker()
        self._gemini_client = None
        self._groq_client = None
        # In-memory cooldowns set on hard 429s so subsequent calls in the same run
        # fall straight through to the next provider instead of re-pacing into a
        # guaranteed rate-limit failure for every scene.
        self._cooldown_until: Dict[str, float] = {}

        self.gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")

    def query_json(self, prompt: str, schema: Optional[Type[BaseModel]] = None, timeout: float = 60.0, task_name: str = "generic", allowed_providers: Optional[Tuple[str, ...]] = None) -> Tuple[Optional[dict], Optional[str]]:
        """Returns (parsed_json_or_None, provider_label_or_None). Never raises.

        allowed_providers restricts the fallback chain for quality-sensitive tasks
        (e.g. clean-check is only trustworthy on Gemini; the smaller Groq model
        false-flags real story text as boilerplate).
        """
        if not self.enabled:
            return None, None

        chain = (
            ("gemini", self._call_gemini),
            ("groq", self._call_groq),
            ("ollama", self._call_ollama),
        )
        active = [(n, f) for n, f in chain if not allowed_providers or n in allowed_providers]
        # Single-provider quality-gated tasks (clean-check, alias merge, book
        # analysis) get bounded wait-and-retry: one wait wasn't enough when the
        # provider's RPM window was still hot from preceding calls (measured:
        # the alias merge waited once, hit a fresh 429, and silently no-oped).
        attempts = 3 if len(active) == 1 else 1
        for _ in range(attempts):
            for provider_name, attempt_fn in active:
                cooldown_remaining = self._cooldown_until.get(provider_name, 0.0) - time.time()
                if cooldown_remaining > 0:
                    if len(active) == 1 and cooldown_remaining <= 90.0:
                        logger.info(f"Waiting {cooldown_remaining:.0f}s for {provider_name} cooldown ({task_name})...")
                        time.sleep(cooldown_remaining)
                    else:
                        continue
                payload, label = attempt_fn(prompt, schema, timeout, task_name)
                if payload is not None:
                    return payload, label

        return None, None

    # ------------------------------------------------------------------
    # Provider: Gemini (Google AI Studio free tier, NOT Vertex)
    # ------------------------------------------------------------------
    def _call_gemini(self, prompt: str, schema: Optional[Type[BaseModel]], timeout: float, task_name: str) -> Tuple[Optional[dict], Optional[str]]:
        if not self.gemini_api_key:
            return None, None
        if self.quota.is_exhausted("gemini", GEMINI_RPD):
            logger.info("Gemini daily quota exhausted; skipping to next provider.")
            return None, None

        try:
            from google import genai
            from google.genai import types, errors
        except ImportError:
            logger.warning("google-genai package not installed; skipping Gemini.")
            return None, None

        if self._gemini_client is None:
            try:
                self._gemini_client = genai.Client(api_key=self.gemini_api_key)
            except Exception as e:
                logger.warning(f"Failed to construct Gemini client: {e}")
                return None, None

        for model_name in self.GEMINI_MODELS:
            self.quota.wait_for_rpm("gemini", GEMINI_RPM)
            start = time.monotonic()
            success = False
            error_str = None
            try:
                config = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1)
                response = self._gemini_client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
                self.quota.record_call("gemini")
                raw_text = response.text or ""
                parsed = _extract_json_object(raw_text)
                success = parsed is not None
                if success:
                    label = f"gemini:{model_name}"
                    self._audit("gemini", model_name, task_name, start, True, None)
                    return parsed, label
                error_str = "JSON parse failure"
            except errors.ClientError as e:
                self.quota.record_call("gemini")
                error_str = str(e)
                if getattr(e, "code", None) == 429:
                    self._cooldown_until["gemini"] = time.time() + 65.0
                    logger.warning(f"Gemini rate limited (429) on {model_name}; cooling down 65s and falling through.")
                    self._audit("gemini", model_name, task_name, start, False, error_str)
                    break  # do not try smaller gemini models after a hard rate limit
                logger.warning(f"Gemini client error on {model_name}: {e}")
            except Exception as e:
                error_str = str(e)
                logger.warning(f"Gemini call failed on {model_name}: {e}")

            self._audit("gemini", model_name, task_name, start, success, error_str)

        return None, None

    # ------------------------------------------------------------------
    # Provider: Groq (free tier)
    # ------------------------------------------------------------------
    def _call_groq(self, prompt: str, schema: Optional[Type[BaseModel]], timeout: float, task_name: str) -> Tuple[Optional[dict], Optional[str]]:
        if not self.groq_api_key:
            return None, None
        if self.quota.is_exhausted("groq", GROQ_RPD):
            logger.info("Groq daily quota exhausted; skipping to next provider.")
            return None, None

        try:
            import groq
        except ImportError:
            logger.warning("groq package not installed; skipping Groq.")
            return None, None

        if self._groq_client is None:
            try:
                self._groq_client = groq.Groq(api_key=self.groq_api_key, timeout=timeout)
            except Exception as e:
                logger.warning(f"Failed to construct Groq client: {e}")
                return None, None

        self.quota.wait_for_rpm("groq", GROQ_RPM)
        start = time.monotonic()
        success = False
        error_str = None
        try:
            completion = self._groq_client.chat.completions.create(
                model=self.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            self.quota.record_call("groq")
            raw_text = completion.choices[0].message.content or ""
            parsed = _extract_json_object(raw_text)
            success = parsed is not None
            if success:
                self._audit("groq", self.GROQ_MODEL, task_name, start, True, None)
                return parsed, f"groq:{self.GROQ_MODEL}"
            error_str = "JSON parse failure"
        except groq.RateLimitError as e:
            self.quota.record_call("groq")
            self._cooldown_until["groq"] = time.time() + 65.0
            error_str = str(e)
            logger.warning(f"Groq rate limited: {e}; cooling down 65s.")
        except Exception as e:
            error_str = str(e)
            logger.warning(f"Groq call failed: {e}")

        self._audit("groq", self.GROQ_MODEL, task_name, start, success, error_str)
        return None, None

    # ------------------------------------------------------------------
    # Provider: local Ollama (relocated from looped_analyzer.py, unchanged behavior)
    # ------------------------------------------------------------------
    def _call_ollama(self, prompt: str, schema: Optional[Type[BaseModel]], timeout: float, task_name: str) -> Tuple[Optional[dict], Optional[str]]:
        available_models = _list_local_ollama_models()
        model_name = _select_preferred_ollama_model(available_models, self.ollama_model)
        if not model_name:
            return None, None

        payload = {
            "model": model_name,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.1},
        }
        start = time.monotonic()
        success = False
        error_str = None
        for attempt in range(1, 3):
            try:
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    self.ollama_url,
                    data=data,
                    headers={"Content-Type": "application/json"},
                )
                res = urllib.request.urlopen(req, timeout=timeout)
                if res.status == 200:
                    response_obj = json.loads(res.read().decode("utf-8"))
                    parsed = json.loads(response_obj.get("response", "{}"))
                    success = True
                    self._audit("ollama", model_name, task_name, start, True, None)
                    return parsed, f"ollama:{model_name}"
            except Exception as e:
                error_str = str(e)
                logger.warning(f"Ollama JSON query attempt {attempt}/2 failed: {e}")
                if attempt == 1:
                    time.sleep(2)

        self._audit("ollama", model_name, task_name, start, success, error_str)
        return None, None

    def _audit(self, provider: str, model: str, task_name: str, start_monotonic: float, success: bool, error: Optional[str]) -> None:
        _append_audit_log({
            "timestamp": time.time(),
            "provider": provider,
            "model": model,
            "task_name": task_name,
            "latency_ms": round((time.monotonic() - start_monotonic) * 1000, 1),
            "success": success,
            "error": error,
        })


_default_client: Optional[LLMClient] = None


def _get_default_client() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


def query_llm_json(prompt: str, schema: Optional[Type[BaseModel]] = None, timeout: float = 60.0, task_name: str = "generic", allowed_providers: Optional[Tuple[str, ...]] = None) -> Tuple[Optional[dict], Optional[str]]:
    """Module-level convenience wrapper around a lazily-constructed default LLMClient."""
    return _get_default_client().query_json(prompt, schema=schema, timeout=timeout, task_name=task_name, allowed_providers=allowed_providers)
