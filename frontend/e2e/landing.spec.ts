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
