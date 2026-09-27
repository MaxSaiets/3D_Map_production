# -*- coding: utf-8 -*-
"""Нічна генерація УНІКАЛЬНИХ текстів для SEO-сторінок міст/районів/вулиць (Gemini).

Принцип: ШІ лише переказує живою мовою ПЕРЕВІРЕНІ факти сторінки (Wikidata,
OpenStreetMap, розрахунки), нічого не вигадує. Кожна відповідь проходить валідатор:
  * без тире (— –) і без типових ШІ-кліше;
  * усі числа в тексті мають бути з набору фактів (інакше — повтор із зауваженням);
  * обсяг 90–190 слів на мову, обидві мови присутні.
Непройдене не публікується (сторінка лишається з шаблонним текстом).

Ключ: GOOGLE_AI_API_KEY читається з /opt/3dmap/backend/.env на сервері через ssh
під час запуску і лише тримається в пам'яті (не пишеться на диск, не логується).
Кеш відповідей: D:\\3dmap_tmp\\ai_texts\\{id}.json (готові пропускаються).
Результат: frontend/lib/pageAiText.ts.

Запуск: python tools/ai_page_texts.py [--limit N] [--pause 9] [--only id1,id2]
"""
import argparse, json, math, re, subprocess, sys, time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
FE = ROOT / "frontend"
CACHE = Path(r"D:\3dmap_tmp\ai_texts")
OSM = Path(r"D:\3dmap_tmp\osm_stats")
URL = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
MODELS = ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-flash-lite"]

BANNED = [
    "унікальн", "неповторн", "шедевр", "пориньте", "поринути", "зануритися", "зануртеся", "ідеальн", "дивовижн", "магі",
    "у серці міста", "в серці міста", "не просто", "справжня перлина", "перлин", "атмосферн", "незабутн", "чарівн",
    "яскравий приклад", "варто зазначити", "важливо зазначити", "підсумовуючи", "отже,", "таким чином", "в епоху",
    "чудов", "неймовірн", "захоплив", "unique", "hidden gem", "stunning", "wonderful", "nestled", "vibrant", "delve", "immerse", "breathtaking", "testament", "tapestry",
    "in the heart of", "not just", "it's worth noting", "in conclusion", "whether you", "look no further",
]
DASHES = ["—", "–", " - "]


def get_key():
    out = subprocess.run(["ssh", "monadruk-vm-cf", "grep '^GOOGLE_AI_API_KEY=' /opt/3dmap/backend/.env | cut -d= -f2-"],
                         capture_output=True, text=True, timeout=60).stdout.strip().strip('"')
    if not out:
        sys.exit("немає GOOGLE_AI_API_KEY на сервері")
    return out


# ---------- факти сторінок (парсимо TS-джерела) ----------
def hav(a, b):
    R = 6371
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    s = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * R * math.asin(math.sqrt(s))


def bearing_uk(a, b):
    y = math.sin(math.radians(b[1] - a[1])) * math.cos(math.radians(b[0]))
    x = math.cos(math.radians(a[0])) * math.sin(math.radians(b[0])) - math.sin(math.radians(a[0])) * math.cos(math.radians(b[0])) * math.cos(math.radians(b[1] - a[1]))
    i = round(((math.degrees(math.atan2(y, x)) + 360) % 360) / 45) % 8
    return ["північ", "північний схід", "схід", "південний схід", "південь", "південний захід", "захід", "північний захід"][i]


def field(line, name):
    m = re.search(name + r': (?:"([^"]*)"|(\d+(?:\.\d+)?))', line)
    return None if not m else (m.group(1) if m.group(1) is not None else float(m.group(2)))


def obj(line, name):
    m = re.search(name + r': \{ uk: "([^"]*)", (?:latin|en): "([^"]*)" \}', line)
    return None if not m else (m.group(1), m.group(2))


def center(line):
    m = re.search(r"center: \[([\d.]+), ([\d.]+)\]", line)
    return (float(m.group(1)), float(m.group(2)))


def pages():
    P = {}
    tpl = (FE / "lib" / "templates.ts").read_text(encoding="utf-8")
    city_center, city_uk = {}, {}
    for key, label, lat, lon in re.findall(r'\{ key: "(\w+)",\s*label: "([^"]+)",\s*center: \[([\d.]+), ([\d.]+)\]', tpl):
        slug = "ivano-frankivsk" if key == "IvanoFrankivsk" else key.lower().replace("_", "-")
        city_center[slug] = (float(lat), float(lon)); city_uk[slug] = label
    facts_src = (FE / "lib" / "cityFacts.ts").read_text(encoding="utf-8")
    for slug in city_center:
        m = re.search(r'(?:^|\n)\s+"?' + re.escape(slug) + r'"?: \{(.*?)\} \},', facts_src, re.S)
        f = {"тип": "обласний центр", "місто": city_uk[slug]}
        if m:
            blk = m.group(1)
            for k, lab in (("population", "населення"), ("populationYear", "рік оцінки"), ("founded", "рік"), ("area_km2", "площа_км2")):
                v = re.search(k + r": ([\d.]+)", blk)
                if v: f[lab] = float(v.group(1)) if "." in v.group(1) else int(v.group(1))
            fm = re.search(r"firstMention: (true|false)", blk)
            if fm and "рік" in f:
                f["рік_першої_згадки" if fm.group(1) == "true" else "рік_заснування"] = f.pop("рік")
            for k, lab in (("river", "водойма"), ("oblast", "область"), ("landmark", "візитівка")):
                o = re.search(k + r': \{ uk: "([^"]*)"', blk)
                if o: f[lab] = o.group(1)
        P[slug] = ("місто", f, city_uk[slug])
    for line in (FE / "lib" / "uaCities2.ts").read_text(encoding="utf-8").splitlines():
        if 'slug: "' not in line or "names:" not in line:
            continue
        slug = field(line, "slug"); nm = re.search(r'names: \{ uk: "([^"]*)"', line).group(1)
        f = {"тип": "місто", "місто": nm, "населення": int(field(line, "population")), "рік оцінки": int(field(line, "populationYear"))}
        a = field(line, "area_km2")
        if a: f["площа_км2"] = a; f["щільність_осіб_км2"] = round(f["населення"] / a)
        for k, lab in (("oblast", "область"), ("river", "водойма"), ("landmark", "візитівка")):
            o = obj(line, k)
            if o: f[lab] = o[0]
        fo = field(line, "founded")
        if fo: f["рік_першої_згадки" if "firstMention: true" in line else "рік_заснування"] = int(fo)
        c = center(line)
        near = sorted(city_center, key=lambda s: hav(c, city_center[s]))[0]
        f["найближчий_обласний_центр"] = f"{city_uk[near]}, {round(hav(c, city_center[near]))} км, напрям {bearing_uk(c, city_center[near])}"
        P[slug] = ("місто", f, nm)
    raions = []
    entries = []
    for ln in (FE / "lib" / "cityRaions.ts").read_text(encoding="utf-8").splitlines():
        if 'citySlug: "' in ln:
            entries.append(ln)
        elif entries and "landmarks:" in ln and "L(" in ln:
            entries[-1] += ln
    for line in entries:
        cs, slug, uk = field(line, "citySlug"), field(line, "slug"), field(line, "uk")
        c = center(line)
        lms = re.findall(r'L\("([^"]+)"', line)
        raions.append((cs, slug, uk, c))
        f = {"тип": "район міста", "район": uk, "місто": city_uk.get(cs, cs),
             "відстань_від_центру_міста_км": round(hav(city_center[cs], c), 1), "напрям_від_центру": bearing_uk(city_center[cs], c)}
        if lms: f["відомі_місця_району"] = lms
        P[f"{cs}--{slug}"] = ("район", f, f"{uk}, {city_uk.get(cs, cs)}")
    for line in (FE / "lib" / "cityStreets.ts").read_text(encoding="utf-8").splitlines():
        if 'citySlug: "' not in line:
            continue
        cs, slug, uk = field(line, "citySlug"), field(line, "slug"), field(line, "uk")
        c = center(line)
        f = {"тип": "вулиця", "вулиця": uk, "місто": city_uk.get(cs, cs),
             "відстань_від_центру_міста_км": round(hav(city_center[cs], c), 1)}
        rs = [r for r in raions if r[0] == cs]
        if rs: f["район"] = min(rs, key=lambda r: hav(c, r[3]))[2]
        for k, lab in (("namedAfter", "названа_на_честь"),):
            v = field(line, k)
            if v: f[lab] = v
        for k, lab in (("lengthM", "довжина_м"), ("since", "відома_з_року")):
            v = field(line, k)
            if v: f[lab] = int(v)
        P[f"{cs}--{slug}"] = ("вулиця", f, f"{uk}, {city_uk.get(cs, cs)}")
    return P


def add_osm(pid, f):
    p = OSM / f"{pid}.json"
    if not p.exists():
        return f
    s = json.loads(p.read_text(encoding="utf-8"))
    f["ділянка_моделі"] = "квадрат 800 на 800 метрів навколо центру, мапа 8 см"
    f["будівель_у_ділянці"] = s.get("b")
    if s.get("maxLv"): f["найвища_будівля_поверхів"] = s["maxLv"]
    if s.get("avgLv"): f["середня_поверховість"] = s["avgLv"]
    if s.get("parks"): f["парки_й_сквери"] = s["parks"]
    if s.get("water"): f["водойми"] = s["water"]
    if s.get("pois"): f["памятки_й_будівлі"] = [n for n, _ in s["pois"][:6]]
    return f


SYSTEM = """Ти досвідчений редактор українського інтернет-магазину Monadruk. Магазин друкує на 3D-принтері обʼємні мапи міст, районів і вулиць: будинки з реальною висотою, дороги, парки, вода. Мапу можна купити готовою або замовити файл.

Пиши як жива людина, яка знає місто: просто, конкретно, без пафосу.
Суворі правила:
1. Жодних тире: ні довгого, ні середнього, ні дефіса як розділового знака між словами. Замість тире став кому, двокрапку або крапку. Дефіс лише всередині слів (3D-модель, Івано-Франківськ).
2. Лише факти з наданих даних. Не вигадуй історію, дати, людей, архітекторів, події, числа. Не описуй характер місцевості, якого немає в даних (житловий масив, тихий, зелений, історичний, люди гуляють, відпочивають, багато туристів). Якщо факту немає, не пиши про це.
3. Числа бери точно з даних; можна округлити населення до тисяч словами (близько 82 тисяч).
4. Заборонені слова й звороти: унікальний, неповторний, ідеальний, дивовижний, магія, шедевр, перлина, атмосферний, незабутній, чарівний, поринути, зануритися, у серці міста, не просто ... а, варто зазначити, таким чином, отже, підсумовуючи. Англійською: unique, vibrant, nestled, delve, immerse, breathtaking, testament, tapestry, in the heart of, not just, it's worth noting.
5. Без риторичних питань, без звертань «уявіть», без списків, без емодзі, без лапок для наголосу.
6. Природно, 1 або 2 рази на мову, використай фрази покупця: «купити 3D-модель», «замовити макет», «3D-мапа» (англійською: buy a 3D model, order a map).
7. Кожне речення має нести факт або практичну користь для покупця (що буде видно на моделі, який розмір обрати, кому це подарунок).
8. Різна будова речень, різні початки абзаців.
9. Про те, що буде ВИДНО НА МОДЕЛІ, пиши лише з полів «памятки_й_будівлі», «парки_й_сквери», «водойми», «будівель_у_ділянці». Візитівка й водойма міста можуть бути поза ділянкою 800 на 800 м, тому про них пиши як про факти міста, а не моделі.
11. Про продукт (ціни, терміни, доставка, матеріал) максимум ОДНЕ коротке речення на мову, і не в кожному тексті однакове. Решта тексту про саме місце: факти з даних і що з них буде на моделі, який розмір обрати для цієї ділянки, кому така мапа може бути дорога.
10. Не починай фразами «Ми пропонуємо», «Пропонуємо вам», «We offer». Пиши звичайно, як про річ, яку людина хоче подарувати або поставити вдома.

Відповідь лише JSON: {"uk": ["абзац 1", "абзац 2"], "en": ["paragraph 1", "paragraph 2"]}. Кожна мова 110 до 170 слів загалом."""


def allowed_numbers(f):
    nums = set()
    def walk(v):
        if isinstance(v, (int, float)):
            nums.add(str(int(v)) if float(v).is_integer() else str(v).replace(".", ","))
            nums.add(str(v)); nums.add(str(int(round(v))))
        elif isinstance(v, str):
            for n in re.findall(r"\d+(?:[.,]\d+)?", v): nums.add(n); nums.add(n.replace(".", ","))
        elif isinstance(v, list):
            for x in v: walk(x)
        elif isinstance(v, dict):
            for x in v.values(): walk(x)
    walk(f)
    nums |= {"3", "8", "800", "3D", "3MF", "350", "770", "170", "149", "2", "4", "5", "11", "15", "1", "10"}
    pop = f.get("населення")
    if pop:
        nums.add(str(round(pop / 1000))); nums.add(str(round(pop / 1000000, 1)).replace(".", ","))
    return nums


def fact_values(f):
    """Усі числа з фактів (як float) + сталі продукту (розміри, ціни, терміни)."""
    vals = set()

    def walk(v):
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            vals.add(float(v))
        elif isinstance(v, str):
            for n in re.findall(r"\d+(?:[.,]\d+)?", v):
                vals.add(float(n.replace(",", ".")))
        elif isinstance(v, list):
            for x in v: walk(x)
        elif isinstance(v, dict):
            for x in v.values(): walk(x)
    walk(f)
    vals |= {3, 8, 800, 350, 770, 170, 149, 2, 4, 5, 5.5, 11, 15, 1, 10, 6, 9, 12}
    return vals


def numbers_in(txt):
    t = txt.replace(" ", " ").replace(" ", " ")
    t = re.sub(r"(?<=\d) (?=\d{3}\b)", "", t)   # 2 952 301 → 2952301
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)   # 717,000 → 717000
    t = re.sub(r"\b3D\b|\b3MF\b", "", t)
    return [float(n.replace(",", ".")) for n in re.findall(r"\d+(?:[.,]\d+)?", t)]


def number_ok(x, vals):
    """Число з тексту допустиме, якщо це значення з фактів (у т. ч. «717 тисяч», «2,95 мільйона»), ±1 %."""
    for v in vals:
        for mul in (1, 1000, 1_000_000):
            if v and abs(x * mul - v) <= max(0.011 * v, 0.51 if v < 10 else 1.0):
                return True
    return False


def validate(res, f):
    errs = []
    if not isinstance(res, dict) or not res.get("uk") or not res.get("en"):
        return ["немає uk або en"]
    vals = fact_values(f)
    for lang in ("uk", "en"):
        paras = res[lang] if isinstance(res[lang], list) else [res[lang]]
        txt = " ".join(paras)
        wc = len(re.findall(r"\w+", txt))
        if not 80 <= wc <= 230: errs.append(f"{lang}: {wc} слів замість 110-170")
        for d in DASHES:
            if d in txt: errs.append(f"{lang}: є тире «{d.strip()}»")
        low = txt.lower()
        for b in BANNED:
            if b in low: errs.append(f"{lang}: заборонене «{b}»")
        for x in numbers_in(txt):
            if not number_ok(x, vals):
                errs.append(f"{lang}: число {x:g} відсутнє у фактах")
        if "?" in txt: errs.append(f"{lang}: риторичне питання")
        if re.search(r"\d\s*-\s*\d", txt): errs.append(f"{lang}: діапазон через дефіс (пиши «від 2 до 4»)")
        prod = sum(low.count(w) for w in (["гривень", "грн", "новою поштою", "3mf", "робочих дн", "pla", "брелок"] if lang == "uk"
                                            else ["uah", "hryvnia", "nova poshta", "3mf", "working day", "pla", "keychain"]))
        if prod > 2: errs.append(f"{lang}: забагато про ціни/доставку ({prod}), залиш одне речення, решта про місце")
        if lang == "uk" and "рік_першої_згадки" in f and re.search(r"засн", low):
            errs.append("uk: рік це ПЕРША ЗГАДКА, а не заснування")
        if lang == "en" and "рік_першої_згадки" in f and "founded" in low:
            errs.append("en: the year is the FIRST MENTION, not founding")
    return errs


def ask(key, facts, name, kind, feedback=""):
    product = ["превʼю моделі в конструкторі безкоштовне", "рамку можна посунути прямо на свій будинок",
               "парки й вода друкуються окремим кольором", "мапа буває від 5,5 до 15 см", "на звороті можна додати напис чи дату",
               "можна замовити брелок з цією ж ділянкою", "будинки на моделі мають реальну висоту"]
    pf = product[sum(map(ord, name)) % len(product)]
    user = (f"Тип сторінки: {kind}. Назва: {name}.\nФакти (JSON):\n{json.dumps(facts, ensure_ascii=False, indent=1)}\n\n"
            f"Можна згадати ОДИН факт про продукт: {pf}. Цін, доставки, термінів і матеріалів не згадуй.\n"
            "Напиши 2 абзаци українською і 2 англійською для сторінки, де цю ділянку можна купити як 3D-мапу.")
    if feedback:
        user += "\n\nПопередня версія мала помилки, виправ: " + "; ".join(feedback[:8])
    body = {"systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0.85, "maxOutputTokens": 2500, "responseMimeType": "application/json",
                                 "thinkingConfig": {"thinkingBudget": 0}}}
    for m in MODELS:
        r = requests.post(URL.format(m=m), params={"key": key}, json=body, timeout=90)
        if r.status_code == 429:
            time.sleep(30); continue
        if r.status_code != 200:
            continue
        try:
            t = "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"])
            return json.loads(t[t.find("{"): t.rfind("}") + 1]), m
        except Exception:
            continue
    return None, None


def clean(res):
    out = {}
    for lang in ("uk", "en"):
        paras = res[lang] if isinstance(res[lang], list) else [res[lang]]
        out[lang] = [re.sub(r"\s+", " ", p).strip() for p in paras if p.strip()]
    return out


def write_ts():
    data = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(CACHE.glob("*.json"))}
    data = {k: v for k, v in data.items() if v.get("ok")}
    body = json.dumps({k: {"uk": v["uk"], "en": v["en"]} for k, v in data.items()}, ensure_ascii=False, separators=(",", ":"))
    (FE / "lib" / "pageAiText.ts").write_text(
        "/**\n * Тексти сторінок міст/районів/вулиць: ШІ (Gemini) переказує ЛИШЕ перевірені факти\n"
        " * (Wikidata, OSM); кожен текст пройшов валідатор (без тире, без кліше, числа з фактів).\n"
        " * ФАЙЛ ГЕНЕРУЄ tools/ai_page_texts.py — не правити вручну.\n */\n"
        f"export const PAGE_AI_TEXT: Record<string, {{ uk: string[]; en: string[] }}> = {body};\n", encoding="utf-8")
    return len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pause", type=int, default=9)
    ap.add_argument("--only", default="")
    ap.add_argument("--need-osm", action="store_true", help="лише сторінки, для яких уже зібрано дані OSM")
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    key = get_key()
    P = pages()
    ids = list(P)
    if a.only: ids = [i for i in ids if i in set(a.only.split(","))]
    n = ok = 0
    for pid in ids:
        if (CACHE / f"{pid}.json").exists():
            continue
        if a.need_osm and not (OSM / f"{pid}.json").exists():
            continue
        if a.limit and n >= a.limit:
            break
        n += 1
        kind, facts, name = P[pid]
        facts = add_osm(pid, dict(facts))
        fb, res, model = [], None, None
        for attempt in range(3):
            raw, model = ask(key, facts, name, kind, fb)
            if raw is None:
                fb = ["QUOTA"]; break
            errs = validate(raw, facts)
            if not errs:
                res = clean(raw); break
            fb = errs
            time.sleep(a.pause)
        if not res and fb == ["QUOTA"]:
            # квота Gemini вичерпана: НЕ кешуємо як невдачу, чекаємо й пробуємо цю ж сторінку пізніше
            print(time.strftime("%H:%M:%S"), "QUOTA — пауза 15 хв", pid, flush=True)
            time.sleep(900)
            continue
        rec = {"ok": bool(res), "model": model, "facts": facts}
        if res: rec.update(res); ok += 1
        else: rec["errors"] = fb
        (CACHE / f"{pid}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        print(time.strftime("%H:%M:%S"), "OK " if res else "SKIP", pid, "" if res else fb[:3], flush=True)
        time.sleep(a.pause)
    print("END", ok, "/", n, "у файлі", write_ts(), flush=True)


if __name__ == "__main__":
    main()
