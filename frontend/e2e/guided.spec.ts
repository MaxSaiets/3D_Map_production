import { test, expect } from "@playwright/test";

/**
 * T-6.7 (F-27): guided-флоу /create — дефолтний досвід новачка, який раніше НЕ мав
 * e2e-покриття (create.spec.ts свідомо вимикає guided). Тут guided УВІМКНЕНО:
 * інтро/тур/cookie-банер вимкнені, щоб не перехоплювали кліки.
 */
test.describe("Guided /create (простий режим)", () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => {
      try {
        localStorage.clear();
        localStorage.setItem("intro_create_v1", "1");
        localStorage.setItem("onb_create_v1", "1");
        document.cookie = "mnd_consent=denied;path=/";
      } catch { /* ignore */ }
    });
  });

  test("крок 1: 4 продуктові картки + рядок «Ще:» (брелок/панно/повний)", async ({ page }) => {
    await page.goto("/uk/create");
    const flow = page.getByTestId("scenario-flow");
    await expect(flow).toBeVisible();
    await expect(flow.getByText("Що створюємо?")).toBeVisible();
    // Рівно 4 картки-продукти з фото (брелок/панно/повний — лінки, не картки)
    await expect(flow.locator("img")).toHaveCount(4);
    await expect(flow.getByText("Ще:")).toBeVisible();
    await expect(flow.getByTestId("scenario-full")).toBeVisible();
    await expect(flow.getByRole("link", { name: /Брелок з моїм місцем/ })).toBeVisible();
  });

  test("крок 2: чіп міста → «Місце обрано», CTA безкоштовне, ціна рядком, плитки з порівнянням", async ({ page }) => {
    await page.goto("/uk/create");
    const flow = page.getByTestId("scenario-flow");
    await flow.getByRole("button", { name: /Обʼємна мапа міста/ }).click();
    await expect(flow.getByText("Де ваше місце?")).toBeVisible();
    // Примітка: «Місце обрано» може стати true і без чіпа (гео-центрування /api/geo
    // зсуває дефолтну рамку), тому disabled-стан CTA тут НЕ асертимо.
    await flow.getByRole("button", { name: "Львів", exact: true }).click();
    const cta = page.getByTestId("scenario-create");
    await expect(flow.getByText("Місце обрано")).toBeVisible();
    await expect(cta).toBeEnabled();
    // F-08: ціна НЕ на кнопці безкоштовного превʼю, а рядком під нею
    await expect(cta).toHaveText(/Показати 3D-превʼю · безкоштовно/);
    await expect(cta).not.toHaveText(/₴/);
    await expect(flow.getByText(/Друк \d+ ₴ · доставка Новою Поштою по Україні/)).toBeVisible();
    // F-31: плитки розміру з побутовим порівнянням і ділянкою
    await expect(flow.getByRole("radio", { name: /M · 8 см/ })).toContainText("як банківська картка");
    await expect(flow.getByRole("radio", { name: /M · 8 см/ })).toContainText("≈560 м");
  });

  test("deep-link ?city=Lviv лишає простий режим і одразу ставить місце", async ({ page }) => {
    await page.goto("/uk/create?city=Lviv");
    const flow = page.getByTestId("scenario-flow");
    await expect(flow).toBeVisible();
    await expect(flow.getByText("Крок 2 із 2")).toBeVisible();
    await expect(flow.getByText("Місце обрано")).toBeVisible({ timeout: 10_000 });
    // guided НЕ записано в localStorage як вимкнений
    const guidedFlag = await page.evaluate(() => localStorage.getItem("3dmap_guided_v1"));
    expect(guidedFlag).not.toBe("0");
  });

  test("?grid= вмикає повний конструктор (без ScenarioFlow)", async ({ page }) => {
    await page.goto("/uk/create?grid=1");
    await expect(page.getByTestId("scenario-flow")).toHaveCount(0);
  });

  test("подія open-order відкриває РІВНО один діалог (F-06) у повному режимі", async ({ page }) => {
    await page.goto("/uk/create?grid=1");
    await page.waitForTimeout(800);
    await page.evaluate(() => window.dispatchEvent(new Event("monadruk:open-order")));
    await expect(page.getByRole("dialog")).toHaveCount(1);
    await expect(page.getByRole("dialog").getByText(/лише по Україні/)).toBeVisible();
  });
});

test.describe("Guided /create на телефоні", () => {
  test.use({ viewport: { width: 375, height: 812 }, isMobile: true, hasTouch: true });

  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => {
      try {
        localStorage.clear();
        localStorage.setItem("intro_create_v1", "1");
        localStorage.setItem("onb_create_v1", "1");
        document.cookie = "mnd_consent=denied;path=/";
      } catch { /* ignore */ }
    });
  });

  test("sticky-бар з ціною і CTA видно на кроці 2, без горизонтального overflow (F-04)", async ({ page }) => {
    await page.goto("/uk/create");
    const flow = page.getByTestId("scenario-flow");
    await flow.getByRole("button", { name: /Обʼємна мапа міста/ }).click();
    await flow.getByRole("button", { name: "Київ", exact: true }).click();
    const bar = page.getByTestId("guided-sticky-bar");
    await expect(bar).toBeVisible();
    await expect(bar).toContainText("₴");
    await expect(bar.getByRole("button")).toBeEnabled();
    const box = await bar.boundingBox();
    expect(box).not.toBeNull();
    expect(Math.round((box!.y + box!.height))).toBeLessThanOrEqual(812);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
    expect(overflow).toBe(false);
    // --sticky-h виставлено → cookie/FAB піднімаються над баром
    const stickyH = await page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue("--sticky-h"));
    expect(parseInt(stickyH, 10)).toBeGreaterThan(40);
  });
});
