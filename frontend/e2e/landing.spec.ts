import { test, expect } from "@playwright/test";

test.describe("Лендінг", () => {
  test("hero рендериться, 3D-моделі (Draco) декодуються, скелетони зникають", async ({ page }) => {
    await page.goto("/uk");
    await expect(page.getByRole("heading", { level: 1 })).toContainText("Твоє місто");

    // Підписи демо-панелей (узгодження PARIS/Київ)
    await expect(page.getByText("Брелок · Париж")).toBeVisible();
    await expect(page.getByText("3D-район · Київ, Поділ")).toBeVisible();

    // Ледачі вʼюери: канваси маунтяться, Draco-GLB декодується,
    // скелетони hero зникають (onReady) — наскрізна перевірка 3D-стека.
    await expect(page.locator("canvas").first()).toBeAttached({ timeout: 30_000 });
    const heroPanel = page.locator("div", { hasText: "Жива демонстрація" }).last();
    await expect(heroPanel.locator(".animate-spin")).toHaveCount(0, { timeout: 30_000 });
  });

  test("SEO: 9 FAQ + SEO-блок + структуровані дані (FAQPage/Organization/Product)", async ({ page }) => {
    await page.goto("/uk");
    // 9 видимих FAQ (включно з новими: топо-рельєф, доставка ЄС, магніт)
    await expect(page.locator("details")).toHaveCount(9);
    await expect(page.getByText(/топо-брелок/i).first()).toBeAttached();
    // Видимий SEO-блок з пошуковими фразами
    await expect(page.locator("#seo-title")).toBeAttached();
    // JSON-LD: FAQPage з 9 питаннями + Organization у layout
    const lds = await page.locator('script[type="application/ld+json"]').allTextContents();
    const faqLd = lds.map((s) => JSON.parse(s)).find((o) => o["@type"] === "FAQPage");
    expect(faqLd?.mainEntity?.length).toBe(9);
    expect(lds.join("")).toContain('"Organization"');
    // Product LD на /create і /keychains
    for (const path of ["/uk/create", "/uk/keychains"]) {
      await page.goto(path);
      const pageLds = await page.locator('script[type="application/ld+json"]').allTextContents();
      expect(pageLds.join("")).toContain('"Product"');
      expect(pageLds.join("")).toContain('"BreadcrumbList"');
    }
  });

  test("SEO: hreflang-альтернативи всіма 6 мовами + canonical", async ({ page }) => {
    await page.goto("/uk");
    for (const lang of ["uk", "en", "de", "pl", "fr", "es", "x-default"]) {
      await expect(page.locator(`link[rel="alternate"][hreflang="${lang}"]`)).toHaveCount(1);
    }
    await expect(page.locator('link[rel="canonical"]')).toHaveCount(1);
    // Німецька версія: локалізовані title/description рендеряться
    await page.goto("/de");
    await expect(page.locator("html")).toHaveAttribute("lang", "de");
    const desc = await page.locator('meta[name="description"]').getAttribute("content");
    expect(desc).toContain("Stadt");
  });

  test("футер: touch-targets посилань ≥ 44px", async ({ page }) => {
    await page.goto("/uk");
    const links = page.locator("footer a");
    const n = await links.count();
    expect(n).toBeGreaterThanOrEqual(5);
    for (let i = 0; i < n; i++) {
      const box = await links.nth(i).boundingBox();
      expect(box?.height ?? 0).toBeGreaterThanOrEqual(44);
    }
  });
});
