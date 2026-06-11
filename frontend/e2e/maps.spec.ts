import { test, expect } from "@playwright/test";

test.describe("Programmatic SEO: сторінки міст /maps", () => {
  test("/maps: індекс із 23 містами", async ({ page }) => {
    await page.goto("/uk/maps");
    await expect(page.getByRole("heading", { level: 1 })).toContainText("3D-мапи міст");
    await expect(page.locator("main ul li a")).toHaveCount(23);
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
