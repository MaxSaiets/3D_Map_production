"""Кожне ім'я налаштування osmnx, яке згадує наш код, має існувати насправді.

⭐Навіщо саме такий тест. За дві доби (08–09.09.2026) ОДИН І ТОЙ САМИЙ баг
вкусив ТРИЧІ:

    ox.settings.overpass_endpoint = mirror   # у встановленій osmnx це `overpass_url`
    ox.settings.timeout = 25                 # у встановленій osmnx це `requests_timeout`

У Python присвоєння неіснуючого атрибута нічого не ламає — воно мовчки створює
НОВИЙ атрибут. Тому код виглядав робочим, логи писали «пробую дзеркало X», а
запит ішов на старий хост; короткий таймаут превʼю не діяв взагалі. Наслідок:
генерації поза Україною падали через ~8 хвилин, і людина отримувала порожню
пластину.

Тест обходить наш код по AST (а не грепом, щоб не ловити згадки в коментарях),
збирає всі літеральні `ox.settings.<name>` та `setattr(ox.settings, "<name>")`
і звіряє їх із ВСТАНОВЛЕНОЮ osmnx. Оновлення osmnx, яке перейменує налаштування,
тепер впаде тут — гучно й одразу, а не мовчки в проді через місяць.

Динамічні звертання (`getattr(ox.settings, _EP_ATTR)`) навмисно поза увагою: у
`_run_overpass_with_retries` ім'я обирається через `hasattr` саме тому, що
сумісність там уже перевіряється в рантаймі.
"""
from __future__ import annotations

import ast
from pathlib import Path

import osmnx as ox
import pytest

BACKEND = Path(__file__).resolve().parents[1]


def _is_ox_settings(node: ast.AST) -> bool:
    """Чи це вираз `ox.settings` (або `osmnx.settings`)."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "settings"
        and isinstance(node.value, ast.Name)
        and node.value.id in ("ox", "osmnx")
    )


def _collect(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return set()

    names: set[str] = set()
    for node in ast.walk(tree):
        # ox.settings.<name>  — і читання, і запис
        if isinstance(node, ast.Attribute) and _is_ox_settings(node.value):
            names.add(node.attr)
        # setattr/getattr/hasattr(ox.settings, "<name>")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("setattr", "getattr", "hasattr")
            and len(node.args) >= 2
            and _is_ox_settings(node.args[0])
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            names.add(node.args[1].value)
    return names


def _sources() -> list[Path]:
    files = [BACKEND / "main.py"]
    files += sorted((BACKEND / "services").rglob("*.py"))
    return [f for f in files if f.is_file()]


# Імена, які код навмисно згадує ЯК СТАРІ (fallback для давніх версій osmnx).
# Вони перевіряються через hasattr у рантаймі, тож існувати не зобов'язані.
KNOWN_LEGACY = {"overpass_endpoint", "timeout"}


def test_every_osmnx_setting_we_touch_exists():
    used: dict[str, list[str]] = {}
    for path in _sources():
        for name in _collect(path):
            used.setdefault(name, []).append(str(path.relative_to(BACKEND)))

    assert used, "не знайшов жодного звертання до ox.settings — тест втратив сенс"

    missing = {
        name: where
        for name, where in sorted(used.items())
        if name not in KNOWN_LEGACY and not hasattr(ox.settings, name)
    }
    assert not missing, (
        f"osmnx {ox.__version__} не має таких налаштувань, а код у них пише "
        f"(присвоєння мовчки створить мертвий атрибут): {missing}"
    )


@pytest.mark.parametrize("legacy", sorted(KNOWN_LEGACY))
def test_legacy_names_are_only_used_behind_a_runtime_check(legacy):
    """Старе ім'я допустиме лише як запасний варіант поруч із hasattr-перевіркою.
    Якщо воно колись знову з'явиться як пряме присвоєння — тест впаде."""
    for path in _sources():
        src = path.read_text(encoding="utf-8", errors="replace")
        bad = f"ox.settings.{legacy} ="
        assert bad not in src, (
            f"{path.relative_to(BACKEND)}: пряме присвоєння `{bad}` — "
            f"у osmnx {ox.__version__} такого налаштування немає, воно нічого не зробить"
        )


def test_the_two_settings_the_overpass_code_depends_on():
    """Іменний щит на випадок оновлення osmnx: якщо зникне і нове, і старе ім'я,
    фолбек у `_run_overpass_with_retries` тихо перестане перемикати дзеркала."""
    assert hasattr(ox.settings, "overpass_url") or hasattr(ox.settings, "overpass_endpoint")
    assert hasattr(ox.settings, "requests_timeout") or hasattr(ox.settings, "timeout")
