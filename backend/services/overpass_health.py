# -*- coding: utf-8 -*-
"""Запобіжник (circuit breaker) для Overpass.

⭐Навіщо (заміряно на проді, реальна генерація користувача, Мадрид, 08.09.2026):
`overpass-api.de` почав відмовляти у зʼєднанні (`[Errno 111] Connection refused`).
Кожен шар даних чесно вичікував свою пару таймаутів:

    будівлі        13:35:09 → 13:38:17   (126 с)
    частини будівель 13:38:17 → 13:40:49 (152 с)
    вода/дороги/мости 13:40:49 → 13:42:58 (129 с)
    ────────────────────────────────────────────
    [TIMING] fetch_source: 471.24s

Майже вісім хвилин очікування — і в кінці користувач отримав ПОРОЖНЮ пластину,
бо пайплайн вважав зону «розрідженою» (`sparse-zone mode`) і мовчки продовжив.

Ідея проста: якщо хост відмовив у зʼєднанні двічі поспіль, він лежить — решта
шарів У ЦІЙ САМІЙ генерації чекатимуть намарно. Памʼятаємо про збій коротку
паузу й відмовляємо миттєво.

ВАЖЛИВО, чому рахуємо не всі помилки: порожня відповідь або довгий запит — це
НЕ аварія джерела. У полі за містом справді може не бути будівель, а важкий
bbox справді може лічитись хвилину. Запобіжник реагує ЛИШЕ на помилки рівня
зʼєднання (хост недосяжний), бо тільки вони гарантовано повторяться для
наступного шару.
"""
from __future__ import annotations

import os
import threading
import time

__all__ = [
    "OverpassUnavailableError",
    "is_connection_error",
    "note_failure",
    "note_success",
    "outage_active",
    "outage_reason",
    "reset",
]


class OverpassUnavailableError(RuntimeError):
    """Джерело карт недосяжне — чекати немає сенсу.

    Повідомлення українською: воно доходить до користувача через статус задачі.
    """

    #: Готовий текст для покупця. Технічна причина живе окремо в `detail` і йде
    #: лише в лог — у фронт віддаємо тільки цей рядок.
    user_message = "Джерело карт (OpenStreetMap) тимчасово недоступне. Спробуйте за кілька хвилин."

    def __init__(self, detail: str = "") -> None:
        super().__init__(f"{self.user_message} [{detail}]" if detail else self.user_message)
        self.detail = detail


def _threshold() -> int:
    try:
        return max(1, int(os.getenv("OVERPASS_BREAKER_FAILS", "2")))
    except Exception:
        return 2


def _cooldown_s() -> float:
    try:
        return max(5.0, float(os.getenv("OVERPASS_BREAKER_COOLDOWN_S", "120")))
    except Exception:
        return 120.0


# Маркери помилок РІВНЯ ЗʼЄДНАННЯ. Перевіряємо і клас, і текст: requests/urllib3
# загортають першопричину так, що клас на верхньому рівні буває загальним, а
# суть лишається в тексті («Connection refused», «Name or service not known»).
_CONN_MARKERS = (
    "connection refused",
    "failed to establish a new connection",
    "newconnectionerror",
    "connectionerror",
    "connecttimeout",
    "connection reset",
    "connection aborted",
    "name or service not known",
    "temporary failure in name resolution",
    "nodename nor servname",
    "no route to host",
    "network is unreachable",
    "max retries exceeded",
)

_lock = threading.Lock()
_fails = 0
_open_until = 0.0
_reason = ""


def is_connection_error(exc: BaseException | None) -> bool:
    """Чи це помилка рівня зʼєднання (хост недосяжний), а не порожня відповідь."""
    if exc is None:
        return False
    seen: set[int] = set()
    node: BaseException | None = exc
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        blob = f"{type(node).__name__} {node}".lower()
        if any(marker in blob for marker in _CONN_MARKERS):
            return True
        node = node.__cause__ or node.__context__
    return False


def note_failure(exc: BaseException | None) -> bool:
    """Реєструє невдачу. Повертає True, якщо вона зарахована як аварія зʼєднання."""
    global _fails, _open_until, _reason
    if not is_connection_error(exc):
        return False
    with _lock:
        _fails += 1
        if _fails >= _threshold():
            _open_until = time.time() + _cooldown_s()
            _reason = str(exc)[:200]
            print(
                f"[OVERPASS] запобіжник УВІМКНЕНО на {int(_cooldown_s())} с після "
                f"{_fails} помилок зʼєднання поспіль: {_reason}",
                flush=True,
            )
    return True


def note_success() -> None:
    """Успішний запит гасить лічильник і закриває запобіжник."""
    global _fails, _open_until, _reason
    with _lock:
        if _fails or _open_until:
            print("[OVERPASS] джерело відповіло — запобіжник вимкнено", flush=True)
        _fails = 0
        _open_until = 0.0
        _reason = ""


def outage_active() -> bool:
    with _lock:
        if _open_until and time.time() < _open_until:
            return True
        if _open_until:
            # пауза минула — даємо джерелу ще один чесний шанс
            _expire_locked()
        return False


def outage_reason() -> str:
    with _lock:
        return _reason


def reset() -> None:
    """Для тестів і ручного скидання."""
    global _fails, _open_until, _reason
    with _lock:
        _fails = 0
        _open_until = 0.0
        _reason = ""


def _expire_locked() -> None:
    global _fails, _open_until, _reason
    _fails = 0
    _open_until = 0.0
    _reason = ""
