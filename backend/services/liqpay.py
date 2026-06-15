"""LiqPay (Приватбанк) online-payment integration — checkout generation + callback
verification. DORMANT until LIQPAY_PUBLIC_KEY / LIQPAY_PRIVATE_KEY are set in env.

LiqPay docs: data = base64(json(params)); signature = base64(sha1(private+data+private)).
Checkout = HTML form POST to https://www.liqpay.ua/api/3/checkout with data+signature.
We return {data, signature, action_url} to the frontend, which auto-submits the form.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from typing import Any, Dict, Optional, Tuple

CHECKOUT_URL = "https://www.liqpay.ua/api/3/checkout"
# LiqPay приймає лише ці валюти
_ALLOWED_CCY = {"UAH", "USD", "EUR"}


def _keys() -> Tuple[Optional[str], Optional[str]]:
    pub = (os.getenv("LIQPAY_PUBLIC_KEY") or "").strip()
    priv = (os.getenv("LIQPAY_PRIVATE_KEY") or "").strip()
    return (pub, priv) if (pub and priv) else (None, None)


def is_configured() -> bool:
    pub, priv = _keys()
    return bool(pub and priv)


def _signature(data_b64: str, private: str) -> str:
    raw = (private + data_b64 + private).encode("utf-8")
    return base64.b64encode(hashlib.sha1(raw).digest()).decode("ascii")


def parse_amount(est_price: str, product_type: str, pricing: Dict[str, Any]) -> Tuple[float, str]:
    """Сума для оплати = ціна, ПОКАЗАНА клієнту (est_price). Fallback — з pricing.json."""
    digits = re.sub(r"[^\d]", "", est_price or "")
    amount = float(digits) if digits else 0.0
    currency = "EUR" if "€" in (est_price or "") else (pricing.get("currency") or "UAH")
    if amount <= 0:
        if product_type == "keychain":
            amount = float(pricing.get("keychain", {}).get("base", 120))
        else:
            amount = float(pricing.get("map", {}).get("from", 250))
        currency = pricing.get("currency") or "UAH"
    if currency not in _ALLOWED_CCY:
        currency = "UAH"
    return round(amount, 2), currency


def build_checkout(
    *,
    amount: float,
    currency: str,
    description: str,
    order_id: str,
    result_url: str = "",
    server_url: str = "",
    language: str = "uk",
) -> Optional[Dict[str, str]]:
    """Повертає {provider, action_url, data, signature} для форми LiqPay, або None
    якщо ключі не налаштовані / сума некоректна."""
    pub, priv = _keys()
    if not pub or not priv or amount <= 0:
        return None
    params: Dict[str, Any] = {
        "public_key": pub,
        "version": "3",
        "action": "pay",
        "amount": f"{amount:.2f}",
        "currency": currency if currency in _ALLOWED_CCY else "UAH",
        "description": (description or "Monadruk")[:280],
        "order_id": str(order_id),
        "language": "uk" if language not in ("uk", "en") else language,
    }
    if result_url:
        params["result_url"] = result_url
    if server_url:
        params["server_url"] = server_url
    data = base64.b64encode(json.dumps(params, ensure_ascii=False).encode("utf-8")).decode("ascii")
    return {
        "provider": "liqpay",
        "action_url": CHECKOUT_URL,
        "data": data,
        "signature": _signature(data, priv),
    }


def verify_callback(data_b64: str, signature: str) -> Optional[Dict[str, Any]]:
    """Перевіряє підпис server-callback від LiqPay. Повертає decoded payload або None."""
    _, priv = _keys()
    if not priv or not data_b64 or not signature:
        return None
    if _signature(data_b64, priv) != signature:
        return None
    try:
        return json.loads(base64.b64decode(data_b64).decode("utf-8"))
    except Exception:
        return None
