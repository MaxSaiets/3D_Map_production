import { test, expect } from "@playwright/test";

/**
 * ⭐09.09.2026, заміряно на 30 днях аналітики: з 23 людей, які СТВОРИЛИ модель,
 * лише 8 з України. Решта — FR, ES, DE, AT, IT, MX, CH; десятеро користувались
 * англійським інтерфейсом. Друк і доставка — тільки по Україні, але дізнавались
 * вони про це аж на екрані «готово», змарнувавши 1–4 хвилини генерації
 * (закордонні найповільніші — ідуть через Overpass) і місце в спільній черзі.
 *
 * Тепер правда стоїть ДО кнопки: що саме людина отримає напевно — файл.
 */
test.describe("Доставка лише по Україні — сказано до генерації", () => {
  test("англійський конструктор мап попереджає ще до кнопки", async ({ page }) => {
    await page.goto("/en/create?product=map3d");
    const note = page.getByTestId("ua-only-early");
    await expect(note).toBeVisible({ timeout: 15_000 });
    await expect(note).toContainText(/Ukraine/i);
    // саме ДО генерації: кнопка створення ще на екрані
    await expect(page.getByTestId("scenario-create")).toBeVisible();
  });

  test("український конструктор цього рядка НЕ показує", async ({ page }) => {
    await page.goto("/uk/create?product=map3d");
    await expect(page.getByTestId("scenario-create")).toBeVisible({ timeout: 15_000 });
    await expect(page.getByTestId("ua-only-early")).toHaveCount(0);
  });

  test("брелки: переклад є, сирий ключ не показується", async ({ page }) => {
    // У брелках кнопка створення зʼявляється після вибору шаблону — рядок
    // стоїть поруч із нею, тобто теж ДО генерації.
    await page.goto("/en/keychains");
    const flow = page.getByTestId("kc-scenario-flow");
    await expect(flow).toBeVisible({ timeout: 15_000 });
    await flow.getByTestId("kc-scenario-heart-46").click();
    await expect(page.getByTestId("kc-scenario-create")).toBeEnabled({ timeout: 20_000 });

    const note = page.getByTestId("ua-only-early");
    await expect(note).toBeVisible();
    await expect(note).not.toHaveText("uaOnly");   // ключ без перекладу
    await expect(note).toContainText(/Ukraine/i);
  });
});
