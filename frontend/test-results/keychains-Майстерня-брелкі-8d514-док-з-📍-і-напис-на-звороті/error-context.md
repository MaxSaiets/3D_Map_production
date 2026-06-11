# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: keychains.spec.ts >> Майстерня брелків /keychains >> нові текстові поля: другий рядок з 📍 і напис на звороті
- Location: e2e\keychains.spec.ts:31:7

# Error details

```
Error: expect(locator).toBeVisible() failed

Locator:  getByPlaceholder('12.06.2026')
Expected: visible
Received: hidden
Timeout:  15000ms

Call log:
  - Expect "toBeVisible" with timeout 15000ms
  - waiting for getByPlaceholder('12.06.2026')
    32 × locator resolved to <input value="" placeholder="12.06.2026" class="min-w-0 flex-1 rounded-[20px] border border-[var(--surface-border)] bg-white/90 px-4 py-3 text-sm font-semibold uppercase tracking-[0.08em] text-[var(--text-primary)] outline-none transition focus:border-[var(--accent)]"/>
       - unexpected value "hidden"

```

```yaml
- dialog "Підказка":
  - img
  - text: Підказка · 1/3
  - button "Закрити":
    - img
  - heading "Оберіть місто та район" [level=4]
  - paragraph: Виберіть місто й точку на карті — це буде мапа на вашому брелку.
  - button "Не показувати знову"
  - button "Далі":
    - text: Далі
    - img
- banner:
  - paragraph: Конструктор брелків
  - heading "Майстерня брелків з мапою" [level=1]
  - paragraph: Пласка багатоколірна пластина з посиленою петлею, чистою смугою під напис і контрольованою висотою будинків.
  - link "Мапи":
    - /url: /
    - img
    - text: Мапи
  - link "Кабінет":
    - /url: /account
    - img
    - text: Кабінет
  - text: Місто
  - combobox:
    - option "Київ" [selected]
    - option "Хмельницький"
    - option "Львів"
    - option "Одеса"
    - option "Дніпро"
    - option "Харків"
    - option "Вінниця"
    - option "Тернопіль"
    - option "Івано-Франківськ"
    - option "Чернігів"
    - option "Запоріжжя"
    - option "Кривий Ріг"
    - option "Миколаїв"
    - option "Полтава"
    - option "Черкаси"
    - option "Житомир"
    - option "Суми"
    - option "Рівне"
    - option "Луцьк"
    - option "Ужгород"
    - option "Чернівці"
    - option "Херсон"
    - option "Кропивницький"
    - option "Інше / вручну"
  - text: Стан Ділянка вибрана
- navigation "Кроки створення":
  - button "Крок 1 Місце Київ · Виділено":
    - img
    - text: Крок 1 Місце Київ · Виділено
  - button "Крок 2 Дизайн Текст, петля, розмір":
    - img
    - text: Крок 2 Дизайн Текст, петля, розмір
  - button "Крок 3 Готово Згенеруйте модель":
    - img
    - text: Крок 3 Готово Згенеруйте модель
- text: 1 Оберіть форму брелка Натисніть приклад — форма застосується
- button "35 x 55 35×55 Стандартний компактний вертикальний брелок." [pressed]:
  - img: KYIV
  - text: 35 x 55 35×55 Стандартний компактний вертикальний брелок.
- button "Token 55 x 30 55×30 Стандартний жетон 55×30 з лівим отвором Ø3 мм і капсульною основою.":
  - img: KYIV
  - text: Token 55 x 30 55×30 Стандартний жетон 55×30 з лівим отвором Ø3 мм і капсульною основою.
- button "Серце 46 × 42 46×42 Мапа місця, що в серці — подарунок для двох.":
  - img: KYIV
  - text: Серце 46 × 42 46×42 Мапа місця, що в серці — подарунок для двох.
- button "Будиночок 44 × 48 44×48 Дім — там, де твоя вулиця. Дах з вушком зверху.":
  - img: KYIV
  - text: Будиночок 44 × 48 44×48 Дім — там, де твоя вулиця. Дах з вушком зверху.
- button "Side Loop 55×35 Петля справа, зручно для широкої карти.":
  - img: KYIV
  - text: Side Loop 55×35 Петля справа, зручно для широкої карти.
- button "Vertical 35×55 Вертикальний брелок з повернутим написом.":
  - img: KYIV
  - text: Vertical 35×55 Вертикальний брелок з повернутим написом.
- button "Capsule 55×35 М'яка капсульна форма з slot-вушком.":
  - img: KYIV
  - text: Capsule 55×35 М'яка капсульна форма з slot-вушком.
- paragraph: "Далі: перетягуйте карту, напис і вушко прямо в прев'ю. Карту й напис можна обертати — тягніть кутову ручку ⟳ на карті або зелену ручку ↻ над написом."
- heading "Постав форму на карту" [level=2]:
  - img
  - text: Постав форму на карту
- paragraph: Перетягни рамку; ручка ⟳ на карті — обертання. Бірюзова рамка тримає пропорції з превʼю.
- img
- button
- button "⟳"
- button "клік = поставити · ⟳ = крутити"
- button "Zoom in"
- button "Zoom out"
- link "Leaflet":
  - /url: https://leafletjs.com
- text: ©
- link "OpenStreetMap":
  - /url: https://www.openstreetmap.org/copyright
- text: contributors
- button "Карта"
- button "Супутник"
- button "⤢ На весь екран"
- button "−15°": ↺
- text: 0°
- button "+15°": ↻
- text: Область друку Клік ставить рамку. Бірюзовий квадрат змінює розмір, кругла ручка ⟳ крутить форму. 123 x 193 м · 0.4 мм = ~1.4 м
- complementary:
  - paragraph: Майстер створення
  - heading "Спочатку форма, потім карта" [level=2]
  - paragraph: "Мінімальний потік: шаблон, ділянка карти, підпис, генерація."
  - img
  - text: Готово до друку Можна створювати 3MF.
  - group: Деталі друку та швидкі дії ▾
  - button "Показати додаткові налаштування":
    - img
    - text: Показати додаткові налаштування
  - navigation:
    - button "1. Виріб"
    - button "2. Карта"
    - button "3. Текст"
    - button "4. Друк"
  - img
  - heading "Перевірте основу брелка" [level=3]
  - paragraph: Готові шаблони знаходяться прямо під превю. Тут залишені тільки швидкі дії, які клієнту реально потрібні після вибору шаблону.
  - text: Основа 35 x 55 мм Вушко кругле
  - button "35 x 55"
  - button "Макс. карта"
  - button "Жетон 55 x 30"
  - button "Центр. вушко"
  - button "Side loop"
  - text: Поворот макета
  - button "0°"
  - button "90°"
  - button "180°"
  - button "270°"
  - button "Створити брелок":
    - img
    - text: Створити брелок
  - button "Завантажити 3MF" [disabled]:
    - img
    - text: Завантажити 3MF
  - button "Замовити друк":
    - img
    - text: Замовити друк
- paragraph:
  - img
  - text: Product Layout
- heading "Розмір, зона карти, вушко і підпис" [level=2]
- paragraph: Підбери форму брелка локально, потім встав обрану ділянку карти в пунктирну область.
- img
- text: 3MF Тягни карту, текст, вушко або нижній правий маркер
- button "Лице"
- button "Зворот"
- img: center KYIV ↻ 35 mm 55 mm map 35 x 55 O 4.0 / hole 2.0
- heading "3D-перегляд готового брелка" [level=2]:
  - img
  - text: 3D-перегляд готового брелка
- button "3D"
- button "Шари"
- img
- text: 3D модель з'явиться після створення Натисніть «Створити 3MF» — і тут зʼявиться реальний 3D-перегляд з усіма шарами.
- button "Звʼязатися з нами":
  - img
- alert
- paragraph:
  - text: Ми використовуємо файли cookie для аналітики та покращення сайту.
  - link "Детальніше":
    - /url: /privacy
- button "Відхилити"
- button "Прийняти"
- img
- text: 1 error
- button "Hide Errors":
  - img
```

# Test source

```ts
  1  | import { test, expect } from "@playwright/test";
  2  | 
  3  | test.describe("Майстерня брелків /keychains", () => {
  4  |   test.beforeEach(async ({ page }) => {
  5  |     await page.goto("/uk/keychains");
  6  |     await page.evaluate(() => localStorage.removeItem("monadruk:draft:keychain"));
  7  |   });
  8  | 
  9  |   test("степер 3 кроки + нові шаблони Серце/Будиночок у списку", async ({ page }) => {
  10 |     await expect(page.locator('nav[aria-label] > button')).toHaveCount(3);
  11 |     await expect(page.getByRole("button", { name: /Серце 46 × 42/ })).toBeVisible();
  12 |     await expect(page.getByRole("button", { name: /Будиночок 44 × 48/ })).toBeVisible();
  13 |   });
  14 | 
  15 |   test("клік «Серце» застосовує параметричний контур у дизайнері", async ({ page }) => {
  16 |     await page.getByRole("button", { name: /Серце 46 × 42/ }).click();
  17 |     // Контур серця — полілінія з 90+ сегментів у SVG превʼю
  18 |     const heartPath = page.locator("svg path").filter({
  19 |       has: page.locator(":scope"),
  20 |     });
  21 |     await expect
  22 |       .poll(async () => {
  23 |         const ds = await page.locator("svg path").evaluateAll((els) =>
  24 |           els.map((e) => e.getAttribute("d") || ""),
  25 |         );
  26 |         return ds.some((d) => (d.match(/L /g) || []).length > 60);
  27 |       }, { timeout: 10_000 })
  28 |       .toBe(true);
  29 |   });
  30 | 
  31 |   test("нові текстові поля: другий рядок з 📍 і напис на звороті", async ({ page }) => {
  32 |     // Панель секційна — поля тексту живуть у табі «3. Текст»
  33 |     await page.getByRole("button", { name: "3. Текст" }).click();
> 34 |     await expect(page.getByPlaceholder("12.06.2026")).toBeVisible();
     |                                                       ^ Error: expect(locator).toBeVisible() failed
  35 |     await expect(page.getByRole("button", { name: "📍" })).toBeVisible();
  36 |     await expect(page.getByPlaceholder("ІМʼЯ · ДАТА")).toBeVisible();
  37 |     await expect(page.getByText(/Гравіюється у нижню грань/)).toBeVisible();
  38 |   });
  39 | 
  40 |   test("топо-режим: перемикач «Рельєф висот» у табі Карта + слайдер висоти", async ({ page }) => {
  41 |     await page.getByRole("button", { name: "2. Карта" }).click();
  42 |     const toggle = page.getByText(/Рельєф висот \(топо\)/);
  43 |     await expect(toggle).toBeVisible();
  44 |     // Вмикаємо — зʼявляється слайдер висоти рельєфу
  45 |     await page.locator("label", { hasText: "Рельєф висот (топо)" }).locator('input[type="checkbox"]').check();
  46 |     await expect(page.getByText("Висота рельєфу")).toBeVisible();
  47 |     await expect(page.getByText(/Гори замість вулиць/)).toBeVisible();
  48 |   });
  49 | 
  50 |   test("чипи форм містять Серце ♥ і Будиночок (додаткові налаштування)", async ({ page }) => {
  51 |     await page.getByRole("button", { name: /Показати додаткові налаштування/ }).click();
  52 |     await expect(page.getByRole("button", { name: "Серце ♥" })).toBeVisible();
  53 |     await expect(page.getByRole("button", { name: "Будиночок", exact: true })).toBeVisible();
  54 |   });
  55 | });
  56 | 
```