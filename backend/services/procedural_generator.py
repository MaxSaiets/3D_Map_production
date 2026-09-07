"""
procedural_generator.py — генерує ДРУКОВАНУ 3D-модель «світу» зі специфікації
(spec), яку дає llm_orchestrator (Claude) або rule-based парсер промту.

Режим «опиши світ → отримай 3D». БЕЗ зовнішнього text-to-3D API: будуємо
heightfield (шум + форма + ерозія) → watertight-solid (верх+стіни+дно), готовий
до друку (manifold by construction).

⭐ПЕРЕПИСАНО 2026-09-08 (власник: «зараз просто херня створюється»). Що було не так
у першій версії і що виправлено — НЕ повертати назад:
  1. Форми були РАДІАЛЬНО СИМЕТРИЧНІ маски (r від центру) → будь-який промт давав
     ту саму «купу в центрі». Тепер кожна форма має власну геометричну ідею:
     гори = хребет із кількох піків, каньйон = ВИРІЗАНЕ звивисте русло, дюни =
     анізотропні хвилі з вітром, кратер = вал + центральна гірка + викиди.
  2. Меш будувався ДВОМА вкладеними Python-циклами (78k граней ≈ 7–13 с на слабкій
     VM). Тепер повністю векторизовано numpy → ~0.05 с. Профіль генерації світу
     впав з ~13 с до ~1 с.
  3. Не було ерозії: fbm-шум виглядає як «вата», а не рельєф. Додано дешеву
     термальну ерозію (обмеження нахилу) + вологу маску долин → схили читаються.
  4. Не було друкарських обмежень: максимальний нахил не обмежувався (друк
     звисань), тонкі шпилі ламались. Тепер slope-clamp у мм/клітинку.

spec = {
  "shape": mountain|island|valley|plateau|ridges|crater|rolling|volcano|archipelago,
  "width_mm": float,            # розмір моделі по X/Y (мм)
  "max_height_mm": float,       # макс. висота рельєфу над базою (мм)
  "base_thickness_mm": float,   # товщина суцільної бази (мм)
  "roughness": float,           # 0..1 деталізація/шум
  "erosion": float,             # 0..1 сила ерозії (0 = сирий шум)
  "seed": int,                  # для відтворюваності
}
generate_world_mesh(spec) -> trimesh.Trimesh (у МІЛІМЕТРАХ, дно на z=0).
"""
from __future__ import annotations

from typing import Any
import numpy as np
import trimesh

try:
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover
    gaussian_filter = None

# Форми, які реально вміє генератор. Кожна — окрема геометрична ідея, не маска
# від центру (див. шапку файлу). Порядок = порядок у UI.
SHAPES = (
    "mountain",      # хребет із кількох піків
    "island",        # острів із нерівним берегом і мілиною
    "valley",        # каньйон: русло, вирізане у плато
    "plateau",       # столова гора з ерозованим краєм
    "ridges",        # дюни/хвилі з напрямком вітру
    "crater",        # ударний кратер: вал, центральна гірка, викиди
    "rolling",       # м'які пагорби
    "volcano",       # конус із кратером і потоками лави
    "archipelago",   # кілька островів у морі
)
_SHAPES = set(SHAPES)
_RES = 160  # роздільність сітки heightfield (N×N вершин)


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def normalize_spec(spec: dict | None) -> dict:
    """Приводить spec у безпечні, друковані межі."""
    spec = dict(spec or {})
    shape = str(spec.get("shape", "mountain")).lower().strip()
    if shape not in _SHAPES:
        shape = "mountain"
    out = {
        "shape": shape,
        "width_mm": float(_clamp(float(spec.get("width_mm", 120) or 120), 40.0, 220.0)),
        "max_height_mm": float(_clamp(float(spec.get("max_height_mm", 18) or 18), 2.0, 40.0)),
        "base_thickness_mm": float(_clamp(float(spec.get("base_thickness_mm", 3) or 3), 1.0, 8.0)),
        "roughness": float(_clamp(float(spec.get("roughness", 0.5) or 0.5), 0.0, 1.0)),
        "erosion": float(_clamp(float(spec.get("erosion", 0.6) or 0.0), 0.0, 1.0)),
        "seed": int(spec.get("seed", 0) or 0) & 0x7FFFFFFF,
        "label": str(spec.get("label", "") or "")[:40],
    }
    return out


# ── Шум ────────────────────────────────────────────────────────────────────────
def _smooth(a: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return a
    if gaussian_filter is not None:
        return gaussian_filter(a, sigma=sigma, mode="nearest")
    # Дешевий сепарабельний box-blur ×2 ≈ гаусс (fallback без scipy).
    k = max(1, int(sigma * 2))
    out = a.astype(np.float64)
    for _ in range(2):
        pad = np.pad(out, k, mode="edge")
        c = np.cumsum(np.cumsum(pad, axis=0), axis=1)
        c = np.pad(c, ((1, 0), (1, 0)))
        n = 2 * k + 1
        out = (c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]) / (n * n)
    return out


def _norm01(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / max(hi - lo, 1e-9)


def _value_noise(n: int, cells: int, rng: np.random.Generator) -> np.ndarray:
    """Гладкий value-noise: випадкова сітка cells×cells → білінійний апсемпл n×n
    зі згладжуванням (smoothstep), щоб не було «квадратів»."""
    g = rng.random((cells + 1, cells + 1))
    t = np.linspace(0, cells, n)
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, cells)
    f = t - i0
    f = f * f * (3.0 - 2.0 * f)  # smoothstep
    top = g[np.ix_(i0, i0)] * (1 - f)[None, :] + g[np.ix_(i0, i1)] * f[None, :]
    bot = g[np.ix_(i1, i0)] * (1 - f)[None, :] + g[np.ix_(i1, i1)] * f[None, :]
    return top * (1 - f)[:, None] + bot * f[:, None]


def _fbm(n: int, rng: np.random.Generator, roughness: float, octaves: int = 6, base_cells: int = 3) -> np.ndarray:
    """Фрактальний шум (fBm) у [0,1]. roughness керує вагою дрібних октав."""
    field = np.zeros((n, n), dtype=np.float64)
    amp, total, cells = 1.0, 0.0, base_cells
    gain = 0.35 + 0.35 * roughness
    for _ in range(octaves):
        field += amp * _value_noise(n, cells, rng)
        total += amp
        amp *= gain
        cells = min(cells * 2, n - 1)
    return _norm01(field / max(total, 1e-9))


def _ridged(n: int, rng: np.random.Generator, roughness: float) -> np.ndarray:
    """Ridged-multifractal: |1-2*noise| дає гострі гребені замість «вати» —
    саме це відрізняє гірський хребет від купи шуму."""
    f = _fbm(n, rng, roughness, octaves=6, base_cells=3)
    r = 1.0 - np.abs(2.0 * f - 1.0)
    return _norm01(r ** 1.4)


def _warp(field_fn, n: int, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Domain warping: зміщуємо координати шумом → «течія» рельєфу, а не сітка."""
    base = field_fn()
    if strength <= 0:
        return base
    wx = (_fbm(n, rng, 0.4, octaves=3, base_cells=2) - 0.5) * 2.0
    wy = (_fbm(n, rng, 0.4, octaves=3, base_cells=2) - 0.5) * 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    amp = strength * n * 0.12
    sx = np.clip(xx + wx * amp, 0, n - 1.001)
    sy = np.clip(yy + wy * amp, 0, n - 1.001)
    # ПАСТКА 08.09: `.astype(int)` = nearest-neighbour → по краях моделі йшли
    # вертикальні смуги-сходинки (видно на hillshade і в друці). Білінійно.
    x0 = sx.astype(int); y0 = sy.astype(int)
    x1 = np.minimum(x0 + 1, n - 1); y1 = np.minimum(y0 + 1, n - 1)
    fx = sx - x0; fy = sy - y0
    top = base[y0, x0] * (1 - fx) + base[y0, x1] * fx
    bot = base[y1, x0] * (1 - fx) + base[y1, x1] * fx
    return top * (1 - fy) + bot * fy


# ── Ерозія ─────────────────────────────────────────────────────────────────────
def _thermal_erosion(h: np.ndarray, iterations: int, talus: float) -> np.ndarray:
    """Термальна ерозія: матеріал зсипається зі схилів, крутіших за talus.
    Векторно, 4-сусідська. Дає «осип» біля підніжжя і читабельні гребені —
    рельєф перестає виглядати як розмитий шум."""
    if iterations <= 0:
        return h
    out = h.astype(np.float64).copy()
    for _ in range(iterations):
        d = np.zeros_like(out)
        for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
            diff = out - np.roll(out, sh, axis=ax)
            # межі не «загортаємо»: обнуляємо перший/останній зріз
            if ax == 0:
                (diff[0, :] if sh == 1 else diff[-1, :]).fill(0.0)
            else:
                (diff[:, 0] if sh == 1 else diff[:, -1]).fill(0.0)
            move = np.where(diff > talus, (diff - talus) * 0.25, 0.0)
            d -= move
            d += np.roll(move, -sh, axis=ax)
        out += d
    return out


def _valley_moisture(h: np.ndarray) -> np.ndarray:
    """Груба «вологість»: наскільки клітинка нижча за околицю. Використовуємо,
    щоб поглибити долини — дешева заміна гідравлічної ерозії."""
    return np.clip(_smooth(h, 3.5) - h, 0.0, None)


# ── Форми ──────────────────────────────────────────────────────────────────────
def _falloff(n: int, margin: float = 0.16) -> np.ndarray:
    """М'який спад до країв плитки (щоб рельєф не обрізався стіною).

    ПАСТКА 08.09: `max(|x|,|y|)` дає ЗЛАМ по діагоналі — на hillshade видно
    рівний діагональний шов через усю модель. Тут добуток двох 1D-smoothstep:
    гладко (C1) по всій площині, кут теж без шва."""
    t = np.linspace(0.0, 1.0, n)
    e = np.clip((np.minimum(t, 1.0 - t)) / max(margin, 1e-6), 0.0, 1.0)
    e = e * e * (3.0 - 2.0 * e)  # smoothstep
    return e[:, None] * e[None, :]


def _peak_ridge(n: int, rng: np.random.Generator, peaks: int) -> np.ndarray:
    """Хребет: ламана лінія через полотно + піки на ній. Це те, що робить
    «гори» горами: кілька вершин уздовж напрямку, а не одна купа в центрі."""
    ang = rng.uniform(0, np.pi)
    cx, cy = n / 2.0, n / 2.0
    dx, dy = np.cos(ang), np.sin(ang)
    ts = np.linspace(-0.42, 0.42, peaks)
    field = np.zeros((n, n), dtype=np.float64)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    for i, t in enumerate(ts):
        jitter = (rng.random(2) - 0.5) * n * 0.12
        px = cx + dx * t * n + jitter[0] - dy * (rng.random() - 0.5) * n * 0.10
        py = cy + dy * t * n + jitter[1] + dx * (rng.random() - 0.5) * n * 0.10
        sigma = n * rng.uniform(0.10, 0.19)
        amp = rng.uniform(0.55, 1.0) if i else 1.0
        field += amp * np.exp(-(((xx - px) ** 2 + (yy - py) ** 2) / (2 * sigma ** 2)))
    return _norm01(field)


def _shape_field(shape: str, n: int, rng: np.random.Generator, roughness: float) -> np.ndarray:
    """Форма → heightfield [0,1]. Кожна гілка — власна геометрія (див. шапку)."""
    fall = _falloff(n)
    if shape == "mountain":
        # Хребет має ДОМІНУВАТИ: у першій версії деталь (0.38) з'їдала силует і
        # виходив «просто шорсткий квадрат». Тепер деталь лише модулює схили.
        ridge = _peak_ridge(n, rng, peaks=int(rng.integers(3, 6)))
        ridge = _norm01(ridge ** 0.75)                       # ширші підошви, гострі верхи
        detail = _warp(lambda: _ridged(n, rng, roughness), n, rng, 0.5)
        h = ridge * (0.70 + 0.30 * detail) + 0.10 * detail * ridge
        h = h * fall
    elif shape == "island":
        # Берег = поріг по шуму, але з ПЛАВНИМ пляжем (smoothstep), інакше виходив
        # прямовисний обрив по всьому контуру — «шматок пінопласту», а не острів.
        land = _warp(lambda: _fbm(n, rng, 0.5, octaves=5, base_cells=2), n, rng, 0.6) * fall
        c = np.clip((land - 0.38) / 0.46, 0.0, 1.0)
        coast = c * c * (3.0 - 2.0 * c)                      # пляж, а не стіна
        hills = _ridged(n, rng, roughness)
        h = coast * (0.35 + 0.65 * hills) * (0.35 + 0.65 * coast)
    elif shape == "archipelago":
        land = _warp(lambda: _fbm(n, rng, 0.6, octaves=6, base_cells=4), n, rng, 0.7) * fall
        c = np.clip((land - 0.49) / 0.34, 0.0, 1.0)
        isl = c * c * (3.0 - 2.0 * c)
        h = isl * (0.45 + 0.55 * _ridged(n, rng, roughness)) * (0.4 + 0.6 * isl)
    elif shape == "valley":
        # Плато, у якому ВИРІЗАНЕ звивисте русло (річка/каньйон) — саме різ, а не
        # радіальна чаша. Русло ширше й з терасами, борти читаються.
        table = 0.70 + 0.30 * _fbm(n, rng, roughness * 0.7, octaves=5, base_cells=3)
        mean = _fbm(n, rng, 0.3, octaves=3, base_cells=2)[:, 0]
        path = 0.5 + (mean - mean.mean()) * 1.5             # звивина русла по X
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
        dist = np.abs(xx / (n - 1) - path[:, None])
        # ширина русла міняється вздовж течії (вектор по рядках, не зріз стовпця)
        width = 0.09 + 0.06 * _fbm(n, rng, 0.3, octaves=2, base_cells=2)[:, 0][:, None]
        u = np.clip(dist / width, 0.0, 1.0)
        cut = 1.0 - u * u * (3.0 - 2.0 * u)                 # 1 у руслі → 0 на бортах
        h = (table * (1.0 - 0.85 * cut)) * fall
    elif shape == "plateau":
        # Меза: ПЛОСКИЙ верх на сталій висоті + круті борти + осип унизу.
        top = _warp(lambda: _fbm(n, rng, 0.35, octaves=4, base_cells=2), n, rng, 0.3)
        m = np.clip((top - 0.48) / 0.12, 0.0, 1.0)
        mesa = m * m * (3.0 - 2.0 * m)
        skirt = 0.18 * _ridged(n, rng, roughness) * (1.0 - mesa)
        h = (0.20 + 0.80 * mesa + skirt) * fall
    elif shape == "ridges":
        # Дюни: анізотропні хвилі + вітровий зсув гребенів.
        ang = rng.uniform(0, np.pi)
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
        proj = (xx * np.cos(ang) + yy * np.sin(ang)) / n
        drift = _fbm(n, rng, 0.4, octaves=3, base_cells=2)
        wave = 0.5 + 0.5 * np.sin(proj * np.pi * rng.uniform(5.0, 9.0) + drift * 5.0)
        h = (wave ** 1.6) * (0.55 + 0.45 * drift) * fall
    elif shape == "crater":
        cx = cy = (n - 1) / 2.0
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (n / 2.0)
        r = r * (0.85 + 0.3 * _fbm(n, rng, 0.3, octaves=3, base_cells=3))  # нерівний вал
        rim = np.exp(-((r - 0.45) ** 2) / (2 * 0.10 ** 2))                  # кільцевий вал
        bowl = -np.exp(-(r ** 2) / (2 * 0.26 ** 2)) * 0.55                  # западина
        peak = np.exp(-(r ** 2) / (2 * 0.06 ** 2)) * 0.30                   # центральна гірка
        ejecta = np.clip(0.35 - r * 0.3, 0.0, None) * _fbm(n, rng, roughness, octaves=4, base_cells=6)
        h = _norm01(0.45 + rim + bowl + peak + ejecta) * fall
    elif shape == "volcano":
        cx = cy = (n - 1) / 2.0
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (n / 2.0)
        r = r * (0.9 + 0.2 * _fbm(n, rng, 0.3, octaves=3, base_cells=4))
        # Крутіший і вужчий конус (був майже плоский млинець) + помітна кальдера.
        cone = np.clip(1.0 - r / 0.62, 0.0, 1.0) ** 0.85
        caldera = np.clip(1.0 - (r / 0.19) ** 2, 0.0, 1.0) * 0.55           # кратер зверху
        flows = _ridged(n, rng, roughness) * np.clip(1.0 - r / 0.7, 0.0, 1.0) * 0.22
        h = _norm01(np.clip(cone - caldera + flows, 0.0, None)) * fall
    else:  # rolling
        base = _warp(lambda: _fbm(n, rng, roughness * 0.5, octaves=4, base_cells=2), n, rng, 0.35)
        h = _smooth(base, 1.2) * fall  # м'які пагорби, а не «шум по всьому полю»
    return _norm01(h)


# ── Побудова меша ──────────────────────────────────────────────────────────────
def _grid_solid(top_z: np.ndarray, w: float) -> trimesh.Trimesh:
    """Heightfield (n×n, мм) → герметичний solid: верх + дно z=0 + 4 стіни.
    ПОВНІСТЮ ВЕКТОРНО (було два Python-цикли на ~78k граней ≈ 7–13 с)."""
    n = top_z.shape[0]
    xs = np.linspace(0.0, w, n)
    gx, gy = np.meshgrid(xs, xs)
    top_v = np.column_stack([gx.ravel(), gy.ravel(), top_z.ravel()])
    bot_v = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(n * n)])
    verts = np.vstack([top_v, bot_v])
    NB = n * n

    idx = np.arange(n * n).reshape(n, n)
    a = idx[:-1, :-1].ravel(); b = idx[:-1, 1:].ravel()
    c = idx[1:, 1:].ravel();   d = idx[1:, :-1].ravel()
    top_f = np.concatenate([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
    bot_f = np.concatenate([np.column_stack([a + NB, c + NB, b + NB]),
                            np.column_stack([a + NB, d + NB, c + NB])])

    walls = []
    # ПАСТКА 08.09 (двічі наступив): намотку стін НЕ вгадувати прапорцями.
    # Грані у площинах x=0 і y=0 дають НУЛЬОВИЙ внесок у знаковий об'єм, тож
    # вивернуті нормалі там не видно ні по mesh.volume, ні по is_watertight —
    # лише слайсер потім бачить дірку. Тому кожну групу орієнтуємо явно за
    # відомим зовнішнім напрямком (детерміновано, без networkx-обходу).
    wall_dirs = []
    for j, outward in ((0, (-1.0, 0.0, 0.0)), (n - 1, (1.0, 0.0, 0.0))):
        t0 = idx[:-1, j]; t1 = idx[1:, j]; b0 = t0 + NB; b1 = t1 + NB
        walls += [np.column_stack([t0, b0, b1]), np.column_stack([t0, b1, t1])]
        wall_dirs += [outward, outward]
    for i, outward in ((0, (0.0, -1.0, 0.0)), (n - 1, (0.0, 1.0, 0.0))):
        t0 = idx[i, :-1]; t1 = idx[i, 1:]; b0 = t0 + NB; b1 = t1 + NB
        walls += [np.column_stack([t0, b0, b1]), np.column_stack([t0, b1, t1])]
        wall_dirs += [outward, outward]

    def _orient(block: np.ndarray, outward) -> np.ndarray:
        """Розвертає ті грані блоку, чия нормаль дивиться всередину."""
        v0 = verts[block[:, 0]]; v1 = verts[block[:, 1]]; v2 = verts[block[:, 2]]
        nrm = np.cross(v1 - v0, v2 - v0)
        bad = nrm @ np.asarray(outward, dtype=float) < 0.0
        out = block.copy()
        out[bad] = out[bad][:, [0, 2, 1]]
        return out

    top_f = _orient(top_f, (0.0, 0.0, 1.0))
    bot_f = _orient(bot_f, (0.0, 0.0, -1.0))
    walls = [_orient(blk, d) for blk, d in zip(walls, wall_dirs)]

    faces = np.vstack([top_f, bot_f] + walls)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    try:
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
    except Exception:  # noqa: BLE001
        pass
    # ⛔НЕ кликати mesh.fix_normals(): trimesh.repair.fix_winding обходить граф
    # суміжності через networkx — 13.4 с із 13.5 с усієї генерації на 102k граней.
    # Обхід намотки тут ПРАВИЛЬНИЙ за побудовою (верх CCW, дно реверснуте, стіни
    # орієнтовані назовні); якщо об'єм від'ємний — просто інвертуємо всі грані.
    try:
        if mesh.volume < 0:
            mesh.invert()
    except Exception:  # noqa: BLE001
        pass
    return mesh


def generate_world_mesh(spec: dict) -> trimesh.Trimesh:
    """spec → watertight-solid trimesh у мм (дно z=0). Manifold by construction."""
    s = normalize_spec(spec)
    n = _RES
    rng = np.random.default_rng(s["seed"] or 12345)

    field = _shape_field(s["shape"], n, rng, s["roughness"])

    # Ерозія: спершу термальна (осип), потім поглиблення долин за «вологістю».
    er = s["erosion"]
    if er > 0.01:
        # ПАСТКА: багато ітерацій із малим talus дають «сходинки-тераси» на крутих
        # схилах (берег острова виглядав як зіккурат). Менше ітерацій, вищий поріг.
        iters = int(4 + 12 * er)
        talus = 0.075 * (1.0 - 0.35 * er)
        field = _thermal_erosion(field, iters, talus)
        field = np.clip(field - _valley_moisture(field) * 0.35 * er, 0.0, None)
        field = _norm01(field)

    # Друкарське обмеження нахилу: без нього шпилі виходять із звисанням >70°
    # (друкується як «спагеті»). Обмежуємо перепад між сусідніми клітинками.
    cell_mm = s["width_mm"] / (n - 1)
    top_z = s["base_thickness_mm"] + field * s["max_height_mm"]
    max_step = cell_mm * 2.6  # ≈69° максимальний схил
    for _ in range(3):
        for ax in (0, 1):
            for sh in (1, -1):
                nb = np.roll(top_z, sh, axis=ax)
                if ax == 0:
                    (nb[0, :] if sh == 1 else nb[-1, :])[...] = top_z[0, :] if sh == 1 else top_z[-1, :]
                else:
                    (nb[:, 0] if sh == 1 else nb[:, -1])[...] = top_z[:, 0] if sh == 1 else top_z[:, -1]
                top_z = np.minimum(top_z, nb + max_step)

    # Легке фінальне згладження — прибирає сходинки після clamp, лишає форму.
    top_z = _smooth(top_z, 0.6)
    top_z = np.maximum(top_z, s["base_thickness_mm"] * 0.6)

    mesh = _grid_solid(top_z, s["width_mm"])
    mesh.visual.face_colors = np.tile([242, 242, 242, 255], (len(mesh.faces), 1))
    return mesh
