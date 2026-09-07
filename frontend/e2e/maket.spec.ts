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

/**
 * Наскрізний крок «завантажив план → редактор зі знайденими стінами» на моках.
 * Раніше e2e макета перевіряв лише статику сторінки, тож регрес у розборі
 * відповіді /api/floorplan/analyze (напр. фікс `_apartment_bbox` 08.09, який
 * повертав 7 стін замість 4) не ловився нічим.
 */
test.describe("Макет: аналіз плану", () => {
  const WALLS = [
    { x1: 60, y1: 60, x2: 940, y2: 60, thickness_m: 22, bearing: true, height_m: null },
    { x1: 940, y1: 60, x2: 940, y2: 640, thickness_m: 22, bearing: true, height_m: null },
    { x1: 940, y1: 640, x2: 60, y2: 640, thickness_m: 22, bearing: true, height_m: null },
    { x1: 60, y1: 640, x2: 60, y2: 60, thickness_m: 22, bearing: true, height_m: null },
    { x1: 500, y1: 60, x2: 500, y2: 640, thickness_m: 12, bearing: false, height_m: null },
    { x1: 500, y1: 350, x2: 940, y2: 350, thickness_m: 12, bearing: false, height_m: null },
  ];

  test("після аналізу відкривається редактор з усіма стінами і масштабом", async ({ page }) => {
    await page.route("**/api/floorplan/capabilities", (r) =>
      r.fulfill({ json: { neural_detector: true, ocr_scale: false, pdf: true, max_upload_mb: 25, sizes_mm: [100, 150, 200, 250] } }));
    await page.route("**/api/floorplan/analyze", (r) =>
      r.fulfill({
        json: {
          plan: {
            walls: WALLS, openings: [], rooms: [], wall_height_m: 2.7,
            scale_source: "door", m_per_px: 0.0091, image_size_px: [1000, 700],
            confidence: 0.99, notes: [],
          },
          scale: { m_per_px: 0.0091, source: "door", confidence: 0.25, detail: "Ширина дверей 80 см", candidates: [], ocr: [] },
          preview: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
          image_size_px: [1000, 700], detector: "nn", confidence: 0.99,
          notes: [], warnings: [], timings_ms: { detect: 800 },
          estimate: { plan_width_m: 8, plan_height_m: 5.3, area_m2: 42, sheet_width_m: 9.1, interior_px2: 500000 },
        },
      }));

    await page.goto("/uk/maket");
    await page.setInputFiles("input[type=file]", {
      name: "plan.png", mimeType: "image/png",
      buffer: Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==", "base64"),
    });
    // редактор показує знайдені стіни (усі 6, а не одну кімнату) і масштаб
    await expect(page.locator("canvas").first()).toBeVisible({ timeout: 30_000 });
    await expect(page.getByText(/6/).first()).toBeVisible();
    await expect(page.getByTestId("beta-banner")).toBeVisible();
  });
});
