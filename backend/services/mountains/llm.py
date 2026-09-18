# -*- coding: utf-8 -*-
"""Єдиний LLM-шар для агентів (Гори, Світи): Gemini → Claude → None (правила).

Ключі з env: GOOGLE_AI_API_KEY (Gemini, REST без SDK) або ANTHROPIC_API_KEY (пакет anthropic).
Повертає ТЕКСТ відповіді (очікуємо JSON); валідація/клампи — у викликача. Ключі ніколи не логуються.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import requests

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def provider() -> Optional[str]:
    if os.getenv("GOOGLE_AI_API_KEY"):
        return "gemini"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "claude"
    return None


def _gemini(system: str, user: str, max_tokens: int) -> Optional[str]:
    """Безкоштовний тариф Gemini має ліміт запитів за хвилину/добу → на 429 пробуємо наступну модель
    (flash → flash-lite), далі викликач іде на правила. Ключ ніколи не логуємо."""
    key = os.getenv("GOOGLE_AI_API_KEY", "")
    models = [m.strip() for m in os.getenv("GEMINI_MODELS", "gemini-2.5-flash,gemini-2.5-flash-lite").split(",") if m.strip()]
    body = {"systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            # gemini-2.5: «думання» зʼїдає бюджет токенів → з 300 приходила порожня відповідь; вимикаємо thinking і даємо запас
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": max(max_tokens, 1500), "responseMimeType": "application/json",
                                 "thinkingConfig": {"thinkingBudget": 0}}}
    for model in models:
        try:
            r = requests.post(GEMINI_URL.format(model=model), params={"key": key}, json=body, timeout=40)
        except requests.RequestException as exc:
            print(f"[LLM] gemini {model}: {exc}", flush=True); continue
        if r.status_code == 429:
            print(f"[LLM] gemini {model}: 429 quota → next model", flush=True); continue
        if r.status_code != 200:
            print(f"[LLM] gemini {model} HTTP {r.status_code}: {r.text[:160]}", flush=True); continue
        d = r.json()
        try:
            return "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError):
            print(f"[LLM] gemini: unexpected response {json.dumps(d)[:160]}", flush=True)
    return None


def _claude(system: str, user: str, max_tokens: int) -> Optional[str]:
    try:
        import anthropic  # type: ignore
    except Exception:
        return None
    client = anthropic.Anthropic()
    msg = client.messages.create(model=os.getenv("MOUNTAINS_LLM_MODEL", "claude-sonnet-5"), max_tokens=max_tokens, system=system,
                                 messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def complete_json(system: str, user: str, max_tokens: int = 700) -> tuple[Optional[dict], Optional[str]]:
    """→ (dict | None, назва провайдера | None). Будь-яка помилка → (None, None), викликач іде на правила."""
    prov = provider()
    if prov is None:
        return None, None
    try:
        txt = (_gemini if prov == "gemini" else _claude)(system, user[:2000], max_tokens)
        if not txt:
            return None, None
        a, b = txt.find("{"), txt.rfind("}")
        if a < 0 or b < 0:
            return None, None
        return json.loads(txt[a:b + 1]), prov
    except Exception as exc:  # noqa: BLE001
        print(f"[LLM] {prov} failed → rules: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
        return None, None
