import { test, expect } from "@playwright/test";

/**
 * ⭐09.09.2026. Ці сторінки — посадкові для платної реклами: заміряно Google
 * autosuggest, і реальний попит в Україні є саме на НАГОДУ («що подарувати на
 * новосілля» — 10 підказок, «подарунок на річницю весілля» — 10), тоді як на
 * назву товару попиту немає взагалі («подарунок з картою», «брелок з картою»,
 * «магніт з картою» — 0 підказок).
 *
 * Людина з реклами вирішує за секунди, а ціну доводилось шукати прокруткою.
 * Тест тримає контракт: ціна, строк і доставка видні одразу, без скролу.
 */
test.describe("Подарункова посадкова сторінка", () => {
  test("ціна, строк і доставка — видні без прокрутки", async ({ page }) => {
    await page.goto("/uk/podarunok/na-novosillya");
    const facts = page.getByTestId("gift-facts");
    await expect(facts).toBeVisible();
    await expect(facts).toContainText("170 ₴");
    await expect(facts).toContainText("210 ₴");
    await expect(facts).toContainText("350 ₴");
    await expect(facts).toContainText("Нова Пошта");
    await expect(facts).toContainText("149 ₴");   // файл платний з 10.09

    // саме «без прокрутки»: блок має бути в межах першого екрана
    const box = await facts.boundingBox();
    const vh = page.viewportSize()?.height ?? 720;
    expect(box).not.toBeNull();
    expect(box!.y).toBeLessThan(vh);
  });

  test("англійська версія каже те саме і чесно про доставку лише по Україні", async ({ page }) => {
    await page.goto("/en/podarunok/na-novosillya");
    const facts = page.getByTestId("gift-facts");
    await expect(facts).toBeVisible();
    await expect(facts).toContainText("within Ukraine");
    await expect(facts).toContainText("149 ₴");
  });

  test("кнопка веде в конструктор", async ({ page }) => {
    await page.goto("/uk/podarunok/na-novosillya");
    await page.getByRole("link", { name: "Створити 3D-мапу" }).first().click();
    await expect(page).toHaveURL(/\/create/);
  });
});
