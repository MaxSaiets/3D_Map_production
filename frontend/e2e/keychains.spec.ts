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
  });
});
