import { test, expect } from "@playwright/test";

test.describe("Майстерня брелків /keychains", () => {
  test.beforeEach(async ({ page }) => {
    // GUIDED-режим (сценарний вхід) увімкнено ЗА ЗАМОВЧУВАННЯМ і ховає повний
    // конструктор — ця сьют тестує саме ПОВНИЙ UI, тож вимикаємо guided до
    // завантаження сторінки (addInitScript діє на всі наступні goto).
    await page.addInitScript(() => {
      localStorage.setItem("3dmap_kc_guided_v1", "0");
      // Інтро-оверлей перекриває панель до першої взаємодії — гасимо.
      localStorage.setItem("intro_keychain_v1", "1");
    });
    await page.goto("/uk/keychains");
    await page.evaluate(() => localStorage.removeItem("monadruk:draft:keychain"));
  });

  test("REGRESSION: зміна міста ПЕРЕЛІТАЄ карту брелка (скидає стару зону)", async ({ page }) => {
    const tileLon = () => page.evaluate(() => {
      const tiles = Array.from(document.querySelectorAll("img.leaflet-tile")) as HTMLImageElement[];
      for (const t of tiles) {
        const m = t.src.match(/\/(\d+)\/(\d+)\/(\d+)\.png/);
        if (m) return (+m[2]) / Math.pow(2, +m[1]) * 360 - 180;
      }
      return null;
    });
    // Київ ~30.5°E
    await expect.poll(tileLon, { timeout: 12000 }).toBeGreaterThan(28);
    // місто → Львів (~24°E) дає чіткий зсув на захід
    await page.locator("select").first().selectOption({ label: "Львів" });
    await expect.poll(tileLon, { timeout: 12000 }).toBeLessThan(26);
  });

  test("вибір міста і форми зверху + нові шаблони Серце/Будиночок у списку", async ({ page }) => {
    // Степер «Крок 1/2/3» свідомо прибрано (власник) — замість нього над сценою
    // компактний вибір міста (перенесено зі шапки) і ряд шаблонів форм.
    await expect(page.getByRole("combobox", { name: "Місто" })).toBeVisible();
    await expect(page.getByRole("button", { name: /Серце 46 × 42/ })).toBeVisible();
    await expect(page.getByRole("button", { name: /Будиночок 44 × 48/ })).toBeVisible();
  });

  test("Ф1b: мобільна навігація уніфікована (sticky БЕЗ ціни, один бар)", async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 });
    await page.goto("/uk/keychains");
    await page.waitForTimeout(1000);
    // Sticky-бар: дія «Створити брелок» + order-now «Замовити друк», БЕЗ ціни
    // (власник: ціна лише при замовленні; продукт-лейбл показується лише разом
    // із ціною — тож у барі його теж нема).
    const sticky = page.locator("div.fixed").filter({
      has: page.locator("button", { hasText: /Створити брелок|Замовити/ }),
    });
    // Рівно ОДИН закріплений бар (раніше конкурували два портали)
    await expect(sticky).toHaveCount(1);
    await expect(sticky.first()).toBeVisible();
    await expect(sticky.first()).not.toContainText(/₴/);
    await expect(sticky.first()).toContainText(/Створити брелок/);
    // Степер «Крок 1/2/3» свідомо прибрано (власник: зайвий chrome) — навігація
    // без дублів: секційні таби панелі лишаються єдиною навігацією налаштувань.
    await expect(page.getByRole("tablist").filter({ has: page.getByRole("tab", { name: "Карта" }) })).toHaveCount(1);
  });

  test("клік «Серце» застосовує параметричний контур у дизайнері", async ({ page }) => {
    await page.getByRole("button", { name: /Серце 46 × 42/ }).click();
    // Контур серця — полілінія з 90+ сегментів у SVG превʼю
    const heartPath = page.locator("svg path").filter({
      has: page.locator(":scope"),
    });
    await expect
      .poll(async () => {
        const ds = await page.locator("svg path").evaluateAll((els) =>
          els.map((e) => e.getAttribute("d") || ""),
        );
        return ds.some((d) => (d.match(/L /g) || []).length > 60);
      }, { timeout: 10_000 })
      .toBe(true);
  });

  test("нові текстові поля: другий рядок з 📍 і напис на звороті", async ({ page }) => {
    // Таби панелі перейменовано (без нумерації) і стали role=tab — секція «Текст»
    await page.getByRole("tab", { name: "Текст" }).click();
    await expect(page.getByPlaceholder("12.06.2026")).toBeVisible();
    // Кнопка «📍» тепер має aria-label (a11y) — шукаємо за ним, не за emoji
    await expect(page.getByRole("button", { name: "Вставити координати центру зони" })).toBeVisible();
    await expect(page.getByPlaceholder("ІМʼЯ · ДАТА")).toBeVisible();
    await expect(page.getByText(/Гравіюється у нижню грань/)).toBeVisible();
  });

  test("топо-режим: перемикач «Рельєф висот» у табі Карта + слайдер висоти", async ({ page }) => {
    // Таб перейменовано «2. Карта» → «Карта» (role=tab)
    await page.getByRole("tab", { name: "Карта" }).click();
    const toggle = page.getByText(/Рельєф висот \(топо\)/);
    await expect(toggle).toBeVisible();
    // Вмикаємо — зʼявляється слайдер висоти рельєфу
    await page.locator("label", { hasText: "Рельєф висот (топо)" }).locator('input[type="checkbox"]').check();
    await expect(page.getByText("Висота рельєфу")).toBeVisible();
    await expect(page.getByText(/Гори замість вулиць/)).toBeVisible();
  });

  test("GPX-трек у брелках: завантаження показує назву + кількість точок", async ({ page }) => {
    // Таб перейменовано «2. Карта» → «Карта» (role=tab)
    await page.getByRole("tab", { name: "Карта" }).click();
    await expect(page.getByText(/Маршрут \(GPX\) на брелку/)).toBeVisible();
    const gpx = `<?xml version="1.0"?><gpx><trk><name>Біг містом</name><trkseg>${Array.from(
      { length: 24 },
      (_, i) => `<trkpt lat="${49.84 + i * 0.0002}" lon="${24.03 + i * 0.0002}"/>`,
    ).join("")}</trkseg></trk></gpx>`;
    await page.locator('[data-testid="kc-gpx-input"]').setInputFiles({
      name: "run.gpx", mimeType: "application/gpx+xml", buffer: Buffer.from(gpx, "utf-8"),
    });
    await expect(page.getByText(/Біг містом · 24 точок/)).toBeVisible();
    await page.getByRole("button", { name: "Прибрати" }).first().click();
    await expect(page.getByText(/Біг містом/)).toHaveCount(0);
  });

  test("чипи форм містять Серце ♥ і Будиночок (додаткові налаштування)", async ({ page }) => {
    await page.getByRole("button", { name: /Показати додаткові налаштування/ }).click();
    await expect(page.getByRole("button", { name: "Серце ♥" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Будиночок", exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: "Пазл L 🧩" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Пазл R 🧩" })).toBeVisible();
  });

  test("пазл-пара: шаблони L/R застосовують контур із виступом/пазом", async ({ page }) => {
    await page.getByRole("button", { name: /Пазл L · 40 × 42/ }).click();
    // Контур L містить дугу головки (sweep=1) за межами тіла
    await expect
      .poll(async () => {
        const ds = await page.locator("svg path").evaluateAll((els) => els.map((e) => e.getAttribute("d") || ""));
        return ds.some((d) => / A [\d.]+ [\d.]+ 0 1 1 /.test(d));
      }, { timeout: 10_000 })
      .toBe(true);
    await page.getByRole("button", { name: /Пазл R · 40 × 42/ }).click();
    // Контур R містить дугу паза (sweep=0)
    await expect
      .poll(async () => {
        const ds = await page.locator("svg path").evaluateAll((els) => els.map((e) => e.getAttribute("d") || ""));
        return ds.some((d) => / A [\d.]+ [\d.]+ 0 1 0 /.test(d));
      }, { timeout: 10_000 })
      .toBe(true);
  });

  test("серце-пара для закоханих: шаблони L/R дають половинку з замком", async ({ page }) => {
    await page.getByRole("button", { name: /Серце пари · L · 30 × 44/ }).click();
    // Тіло L ширше за 30мм (замок стирчить праворуч за грань розрізу)
    await expect
      .poll(async () => {
        const d = await page
          .locator('[data-testid="keychain-body-path"]')
          .evaluate((el) => el.getAttribute("d") || "");
        const xs = (d.match(/-?[\d.]+/g) || []).map(Number).filter((_, i) => i % 2 === 0);
        return Math.max(...xs, 0);
      }, { timeout: 10_000 })
      .toBeGreaterThan(31);
    await page.getByRole("button", { name: /Серце пари · R · 30 × 44/ }).click();
    // Тіло R вписане у 30мм (паз всередину, нічого не стирчить)
    await expect
      .poll(async () => {
        const d = await page
          .locator('[data-testid="keychain-body-path"]')
          .evaluate((el) => el.getAttribute("d") || "");
        const xs = (d.match(/-?[\d.]+/g) || []).map(Number).filter((_, i) => i % 2 === 0);
        return xs.length ? Math.max(...xs) : 999;
      }, { timeout: 10_000 })
      .toBeLessThan(30.5);
    // Чипи нових форм доступні і в додаткових налаштуваннях
    await page.getByRole("button", { name: /Показати додаткові налаштування/ }).click();
    await expect(page.getByRole("button", { name: "Серце пари L 💕" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Серце пари R 💕" })).toBeVisible();
  });
});
