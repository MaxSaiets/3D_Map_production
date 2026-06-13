import { test, expect } from "@playwright/test";

test.describe("Майстерня брелків /keychains", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/uk/keychains");
    await page.evaluate(() => localStorage.removeItem("monadruk:draft:keychain"));
  });

  test("степер 3 кроки + нові шаблони Серце/Будиночок у списку", async ({ page }) => {
    await expect(page.locator('nav[aria-label] > button')).toHaveCount(3);
    await expect(page.getByRole("button", { name: /Серце 46 × 42/ })).toBeVisible();
    await expect(page.getByRole("button", { name: /Будиночок 44 × 48/ })).toBeVisible();
  });

  test("Ф1b: мобільна навігація уніфікована (sticky з ціною + єдиний степер)", async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 });
    await page.goto("/uk/keychains");
    await page.waitForTimeout(1000);
    // Sticky-бар показує ціну одразу (не «—») + дію «Створити брелок»
    const sticky = page.locator("div.fixed").filter({ hasText: /Орієнтовна вартість/i }).first();
    await expect(sticky).toBeVisible();
    await expect(sticky).toContainText(/≈\s*120\s*₴/);
    await expect(sticky).toContainText(/Створити брелок/);
    // Єдина навігація — 3-кроковий степер
    await expect(page.locator('nav[aria-label] > button')).toHaveCount(3);
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
    // Панель секційна — поля тексту живуть у табі «3. Текст»
    await page.getByRole("button", { name: "3. Текст" }).click();
    await expect(page.getByPlaceholder("12.06.2026")).toBeVisible();
    await expect(page.getByRole("button", { name: "📍" })).toBeVisible();
    await expect(page.getByPlaceholder("ІМʼЯ · ДАТА")).toBeVisible();
    await expect(page.getByText(/Гравіюється у нижню грань/)).toBeVisible();
  });

  test("топо-режим: перемикач «Рельєф висот» у табі Карта + слайдер висоти", async ({ page }) => {
    await page.getByRole("button", { name: "2. Карта" }).click();
    const toggle = page.getByText(/Рельєф висот \(топо\)/);
    await expect(toggle).toBeVisible();
    // Вмикаємо — зʼявляється слайдер висоти рельєфу
    await page.locator("label", { hasText: "Рельєф висот (топо)" }).locator('input[type="checkbox"]').check();
    await expect(page.getByText("Висота рельєфу")).toBeVisible();
    await expect(page.getByText(/Гори замість вулиць/)).toBeVisible();
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
          .locator("#keychainMapClip")
          .evaluate((el) => el.closest("svg")?.querySelector('path[fill="#a6926b"]')?.getAttribute("d") || "");
        const xs = (d.match(/-?[\d.]+/g) || []).map(Number).filter((_, i) => i % 2 === 0);
        return Math.max(...xs, 0);
      }, { timeout: 10_000 })
      .toBeGreaterThan(31);
    await page.getByRole("button", { name: /Серце пари · R · 30 × 44/ }).click();
    // Тіло R вписане у 30мм (паз всередину, нічого не стирчить)
    await expect
      .poll(async () => {
        const d = await page
          .locator("#keychainMapClip")
          .evaluate((el) => el.closest("svg")?.querySelector('path[fill="#a6926b"]')?.getAttribute("d") || "");
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
