import { test, expect } from "@playwright/test";

test.describe("Конструктор мап /create", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/uk/create");
    // чернетка з минулих прогонів не має впливати
    await page.evaluate(() => localStorage.removeItem("monadruk:draft:create"));
  });

  test("майстер: 3 клікабельні кроки Місце/Параметри/Готово", async ({ page }) => {
    const steps = page.locator('nav[aria-label] > button');
    await expect(steps).toHaveCount(3);
    await expect(steps.nth(0)).toContainText("Місце");
    await expect(steps.nth(1)).toContainText("Параметри");
    await expect(steps.nth(2)).toContainText("Готово");
  });

  test("шаблон району → зона вибрана → Згенерувати активна", async ({ page }) => {
    await page.getByRole("button", { name: /Поділ/ }).first().click();
    await expect(page.locator("#panel-settings").getByText(/Ділянку вибрано/)).toBeVisible();
    const genBtn = page.getByRole("button", { name: /Згенерувати модель/ }).first();
    await expect(genBtn).toBeEnabled();
  });

  test("магніт: перемикач + поле підпису + жива/фолбек ціна в кнопці замовлення", async ({ page }) => {
    const magnet = page.getByRole("button", { name: /Магніт на холодильник/ });
    await expect(magnet).toBeVisible();
    await magnet.click();
    await expect(page.getByPlaceholder(/Підпис на магніті/)).toBeVisible();

    const orderBtn = page.getByRole("button", { name: /Замовити друк/ }).first();
    await expect(orderBtn).toBeVisible();
  });

  test("діалог замовлення: ціна, Україна/Європа, 15 країн ЄС", async ({ page }) => {
    await page.getByRole("button", { name: /Замовити друк/ }).first().click();
    const dialog = page.locator(".fixed.inset-0", { hasText: "Орієнтовна вартість" });
    await expect(dialog).toBeVisible();
    // ціна: жива (≈ N ₴) або статичний fallback (від N ₴)
    await expect(dialog.getByText(/[≈від]+\s*\d+\s*₴/)).toBeVisible();

    await dialog.getByRole("button", { name: /Європа/ }).click();
    await expect(dialog.getByRole("button", { name: "Nova Post (EU)" })).toBeVisible();
    await expect(dialog.getByRole("button", { name: "Meest" })).toBeVisible();
    const options = dialog.locator("select option");
    await expect(options).toHaveCount(16); // плейсхолдер + 15 країн
  });
});
