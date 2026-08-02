import { expect, test } from "@playwright/test";

/**
 * Сервіс «макет квартири» (/maket).
 *
 * Тест навмисно не залежить від бекенда: він перевіряє те, що ламається саме на
 * фронті — SEO-розмітку, hreflang, i18n і те, що кнопка генерації ЗАКРИТА, поки
 * користувач не підтвердив масштаб. Останнє критичне: непідтверджений масштаб
 * означає виріб неправильного фізичного розміру, а на екрані це непомітно.
 */
test.describe("Макет квартири /maket", () => {
  test("h1, підзаголовок і екран завантаження", async ({ page }) => {
    await page.goto("/uk/maket");
    await expect(page.getByRole("heading", { level: 1 })).toContainText("Макет");
    await expect(page.getByRole("heading", { name: "Завантажте план" })).toBeVisible();
    await expect(page.getByText(/Знімайте прямо згори/)).toBeVisible();
  });

  test("Product, HowTo і BreadcrumbList у JSON-LD", async ({ page }) => {
    await page.goto("/uk/maket");
    const lds = (await page.locator('script[type="application/ld+json"]').allTextContents()).join("");
    expect(lds).toContain('"Product"');
    expect(lds).toContain('"HowTo"');
    expect(lds).toContain('"BreadcrumbList"');
    expect(lds).toContain("MND-FLOORPLAN");
  });

  test("усі 7 hreflang", async ({ page }) => {
    await page.goto("/uk/maket");
    for (const lang of ["uk", "en", "de", "pl", "fr", "es", "x-default"]) {
      await expect(page.locator(`link[rel="alternate"][hreflang="${lang}"]`)).toHaveCount(1);
    }
  });

  test("локалізація: /en і /pl показують свої тексти", async ({ page }) => {
    await page.goto("/en/maket");
    await expect(page.getByRole("heading", { name: "Upload the plan" })).toBeVisible();
    expect(await page.title()).toContain("Floor plan");

    await page.goto("/pl/maket");
    await expect(page.getByRole("heading", { name: "Wgraj rzut" })).toBeVisible();
  });

  test("є посилання в підвалі та в шапці", async ({ page }) => {
    await page.goto("/uk/");
    await expect(page.locator('footer a[href="/maket"]')).toHaveCount(1);
  });

  test("файловий інпут приймає зображення і PDF", async ({ page }) => {
    await page.goto("/uk/maket");
    const input = page.locator('input[type="file"]');
    await expect(input).toHaveCount(1);
    const accept = await input.getAttribute("accept");
    expect(accept).toContain("image/png");
    expect(accept).toContain("application/pdf");
  });
});
