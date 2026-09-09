"""Черга генерацій: чесна оцінка очікування замість «за кілька хвилин».

⭐Прод-заміри 08.09.2026 (`[QUEUE] ... started after Ns wait`), кожна задача —
окрема, `Zone ID: None`, тобто це РІЗНІ запити, а не плитки однієї сітки:

    09:46  1050 с      10:05  2219 с
    09:52  1433 с      10:12  2609 с  ← 43 хвилини
    09:59  1812 с      13:35   546 с  ← той самий візит, що потім упав на Overpass

Увесь цей час на екрані стояло незмінне «Ваша черга настане за кілька хвилин».
Тепер черга знає, ЩО саме рахується, і скільки той вид задачі зазвичай триває.
"""
from __future__ import annotations

import pytest

from services import gen_queue


@pytest.fixture(autouse=True)
def _clean():
    gen_queue.reset_for_tests()
    yield
    gen_queue.reset_for_tests()


def test_no_estimate_when_nothing_is_running():
    assert gen_queue.eta_free_s() is None


def test_estimate_counts_down_as_the_running_job_progresses(monkeypatch):
    monkeypatch.setattr("services.result_cache.eta_seconds", lambda bucket, foreign=False: 600)
    clock = {"t": 1000.0}
    monkeypatch.setattr(gen_queue.time, "time", lambda: clock["t"])

    gen_queue.acquire(gen_queue.CAPACITY, bucket="print:150")
    assert gen_queue.eta_free_s() == 600

    clock["t"] += 200
    assert gen_queue.eta_free_s() == 400, "оцінка має зменшуватись разом із роботою"


def test_estimate_never_promises_zero(monkeypatch):
    """Якщо задача переросла свою оцінку, показуємо мінімум 30 с: нуль на екрані
    при задачі, що досі йде, гірший за скромну обіцянку."""
    monkeypatch.setattr("services.result_cache.eta_seconds", lambda bucket, foreign=False: 100)
    clock = {"t": 1000.0}
    monkeypatch.setattr(gen_queue.time, "time", lambda: clock["t"])

    gen_queue.acquire(gen_queue.CAPACITY, bucket="preview:80")
    clock["t"] += 5000
    assert gen_queue.eta_free_s() == 30


def test_release_clears_the_estimate(monkeypatch):
    monkeypatch.setattr("services.result_cache.eta_seconds", lambda bucket, foreign=False: 300)
    gen_queue.acquire(gen_queue.CAPACITY, bucket="print:80")
    assert gen_queue.eta_free_s() is not None
    gen_queue.release(gen_queue.CAPACITY)
    assert gen_queue.eta_free_s() is None
    assert gen_queue.stats()["running"] == 0


def test_unknown_bucket_gives_no_false_promise(monkeypatch):
    """Краще лишити старий чесний текст «за кілька хвилин», ніж вигадати число."""
    gen_queue.acquire(gen_queue.CAPACITY, bucket="")
    assert gen_queue.eta_free_s() is None


def test_broken_estimator_does_not_break_the_queue(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("stats file corrupt")

    monkeypatch.setattr("services.result_cache.eta_seconds", boom)
    gen_queue.acquire(gen_queue.CAPACITY, bucket="print:150")
    assert gen_queue.eta_free_s() is None       # немає оцінки — але й немає падіння
    gen_queue.release(gen_queue.CAPACITY)


def test_waiting_job_is_told_how_long_to_wait(monkeypatch):
    """Головне, заради чого все: той, хто чекає, отримує оновлення, а не
    завмерлий екран. Викликаємо on_wait із СПРАВЖНЬОЇ черги, у другому потоці."""
    import threading

    monkeypatch.setattr("services.result_cache.eta_seconds", lambda bucket, foreign=False: 900)
    gen_queue.acquire(gen_queue.CAPACITY, bucket="print:150")     # слот зайнято

    ticks: list[tuple[float, int | None]] = []
    started = threading.Event()

    def waiter():
        started.set()
        gen_queue.acquire(gen_queue.CAPACITY, bucket="preview:80",
                          on_wait=lambda waited, eta: ticks.append((waited, eta)))

    th = threading.Thread(target=waiter, daemon=True)
    th.start()
    started.wait(timeout=5)
    # Умова очікування прокидається раз на 5 с; будимо її самі, щоб не спати.
    for _ in range(3):
        with gen_queue._cond:
            gen_queue._cond.notify_all()
    gen_queue.release(gen_queue.CAPACITY)      # звільняємо → waiter проходить
    th.join(timeout=10)
    assert not th.is_alive(), "черга не відпустила задачу"

    if ticks:                                   # тіки не гарантовані таймінгом
        assert all(eta is None or eta > 0 for _w, eta in ticks)


def test_on_wait_failure_never_blocks_the_queue(monkeypatch):
    """Оновлення статусу — річ другорядна: якщо воно падає, черга має працювати."""
    import threading

    gen_queue.acquire(gen_queue.CAPACITY, bucket="print:150")
    passed = threading.Event()

    def waiter():
        gen_queue.acquire(gen_queue.CAPACITY, bucket="preview:80",
                          on_wait=lambda *_a: (_ for _ in ()).throw(RuntimeError("status write failed")))
        passed.set()

    th = threading.Thread(target=waiter, daemon=True)
    th.start()
    for _ in range(3):
        with gen_queue._cond:
            gen_queue._cond.notify_all()
    gen_queue.release(gen_queue.CAPACITY)
    assert passed.wait(timeout=10), "падіння колбека заблокувало чергу"


def test_stats_exposes_running_count_for_deploy():
    """`/api/health` віддає `checks.queue`, щоб скрипт деплою міг дочекатись
    `running == 0` і не рестартувати бекенд посеред чужої генерації.
    За тиждень у логах Caddy 377 відповідей 502 — і всі у вікна рестартів."""
    assert gen_queue.stats()["running"] == 0
    gen_queue.acquire(gen_queue.CAPACITY, bucket="print:150")
    s = gen_queue.stats()
    assert s["running"] == 1 and s["free"] == 0
    gen_queue.release(gen_queue.CAPACITY)
    assert gen_queue.stats()["running"] == 0
