import { test, expect } from "@playwright/test";

/**
 * Режим «опиши світ» + смуга «тестовий режим» (08.09.2026).
 * Генерацію НЕ запускаємо по-справжньому: мокаємо /api/generate-custom і
 * /api/status, щоб тест не залежав від бекенда і не вантажив його.
 */
test.describe("Світи (/worlds)", () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => {
      try { document.cookie = "mnd_consent=denied;path=/"; } catch { /* ignore */ }
    });
  });

  test("смуга «тестовий режим» приклеєна вгорі і не зникає при прокрутці", async ({ page }) => {
    await page.goto("/uk/worlds");
    const banner = page.getByTestId("beta-banner");
    await expect(banner).toBeVisible();
    await expect(banner).toContainText("Тестовий режим");
    await page.mouse.wheel(0, 1600);
    await page.waitForTimeout(400);
    // Після прокрутки смуга ЛИШАЄТЬСЯ у кадрі вгорі (шапка над нею стискається,
    // тож рівно нуля не буде — перевіряємо саме «видно зверху завжди»).
    await expect(banner).toBeInViewport();
    expect((await banner.boundingBox())!.y).toBeLessThan(120);
  });

  test("вибір форми: 10 варіантів українською, «За описом» за замовчуванням", async ({ page }) => {
    await page.goto("/uk/worlds");
    const chips = page.locator("[data-testid^=world-shape-]");
    await expect(chips).toHaveCount(10);
    await expect(page.getByTestId("world-shape-auto")).toHaveAttribute("aria-checked", "true");
    await expect(page.getByTestId("world-shape-volcano")).toContainText("Вулкан");
    await page.getByTestId("world-shape-crater").click();
    await expect(page.getByTestId("world-shape-crater")).toHaveAttribute("aria-checked", "true");
    await expect(page.getByTestId("world-shape-auto")).toHaveAttribute("aria-checked", "false");
  });

  test("готовий світ показує форму українською, «інший варіант» і обидва файли", async ({ page }) => {
    await page.route("**/api/generate-custom", (r) =>
      r.fulfill({ json: { task_id: "w-e2e-1", status: "processing", message: "" } }));
    await page.route("**/api/status/**", (r) =>
      r.fulfill({
        json: {
          status: "completed", progress: 100, message: "Готово · вулкан",
          download_url_glb: "/files/e2e.glb", download_url_3mf: "/files/e2e.3mf",
          world_spec: { shape: "volcano", shapeUk: "вулкан", seed: 1, source: "rules" },
        },
      }));
    await page.goto("/uk/worlds");
    await page.getByTestId("world-prompt").fill("Острів-вулкан у морі");
    await page.getByTestId("world-generate").click();
    await expect(page.getByTestId("world-built")).toContainText("вулкан", { timeout: 15_000 });
    await expect(page.getByTestId("world-reroll")).toBeVisible();
    await expect(page.locator("a[href*='e2e.3mf']")).toBeVisible();
    // Шлях до замовлення: до 08.09 його не було зовсім (глухий кут воронки).
    const order = page.getByTestId("world-order");
    await expect(order).toContainText("надрукувати цей світ");
    await expect(order.getByTestId("world-msg-tg")).toBeVisible();
    await expect(order.getByTestId("world-msg-ig")).toBeVisible();
    await expect(order.getByTestId("world-share")).toBeVisible();
  });

  test("макет (/maket) теж має смугу «тестовий режим»", async ({ page }) => {
    await page.goto("/uk/maket");
    await expect(page.getByTestId("beta-banner")).toContainText("Тестовий режим");
  });
});
