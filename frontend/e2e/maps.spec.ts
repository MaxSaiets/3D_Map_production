import { test, expect } from "@playwright/test";

test.describe("Programmatic SEO: сторінки міст /maps", () => {
  test("/maps: індекс з усіма містами (23 УА + міста Європи)", async ({ page }) => {
    await page.goto("/uk/maps");
    await expect(page.getByRole("heading", { level: 1 })).toContainText("3D-мапи міст");
    // Хвиля 4 (2026-07-29) додала міста Європи — точна кількість плаває з
    // розширенням списку, тому фіксуємо нижню межу, а не exact count.
    const count = await page.locator("main ul li a").count();
    expect(count).toBeGreaterThanOrEqual(23);
  });

  test("/maps/lviv: h1, тексти, CTA, Product+Breadcrumb LD, інші міста", async ({ page }) => {
    await page.goto("/uk/maps/lviv");
    await expect(page.getByRole("heading", { level: 1 })).toContainText("Львів");
    await expect(page.getByText(/OpenStreetMap/).first()).toBeAttached();
    await expect(page.getByRole("link", { name: /Створити 3D-мапу міста Львів/ })).toBeVisible();
    await expect(page.getByRole("link", { name: /Брелок з картою міста Львів/ })).toBeVisible();
    const lds = await page.locator('script[type="application/ld+json"]').allTextContents();
    expect(lds.join("")).toContain('"Product"');
    expect(lds.join("")).toContain('"BreadcrumbList"');
    // Внутрішня перелінковка: чипи інших міст
    await expect(page.getByRole("link", { name: "Київ" })).toBeVisible();
  });

  test("/de/maps/kyiv: німецький екзонім Kiew + локалізовані мета", async ({ page }) => {
    await page.goto("/de/maps/kyiv");
    await expect(page.getByRole("heading", { level: 1 })).toContainText("Kiew");
    const title = await page.title();
    expect(title).toContain("Kiew");
    for (const lang of ["uk", "en", "de", "pl", "fr", "es", "x-default"]) {
      await expect(page.locator(`link[rel="alternate"][hreflang="${lang}"]`)).toHaveCount(1);
    }
  });

  test("невідоме місто → 404", async ({ page }) => {
    const resp = await page.goto("/uk/maps/atlantis");
    expect(resp?.status()).toBe(404);
  });
});
