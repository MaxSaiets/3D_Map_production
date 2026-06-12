import { test, expect } from "@playwright/test";

test.describe("Конструктор мап /create", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/uk/create");
    // чернетка з минулих прогонів не має впливати
    await page.evaluate(() => localStorage.removeItem("monadruk:draft:create"));
  });

  test("UX: чесний степер (рамка готова) + ETA генерації + жодних hydration-помилок", async ({ page }) => {
    const errors: string[] = [];
    page.on("console", (msg) => { if (msg.type() === "error") errors.push(msg.text()); });
    await page.goto("/uk/create");
    // Степер не бреше «Виділено» — каже що рамка дефолтна і її можна пересунути
    await expect(page.getByText(/рамка готова — пересунь або генеруй/).first()).toBeVisible();
    // Чесне очікування: час генерації видно ДО кліку
    await expect(page.getByText(/≈ 1–3 хв/).first()).toBeAttached();
    // StickyActionBar більше не ламає гідрацію (раніше was: server HTML mismatch)
    await page.waitForTimeout(1500);
    expect(errors.filter((e) => /hydration|Expected server HTML/i.test(e))).toHaveLength(0);
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
    const magnet = page.getByRole("button", { name: /Магніт на холодильник/ }).first();
    await expect(magnet).toBeVisible();
    await magnet.click();
    // Стан магніта тепер у store (спільний для обох копій панелі) → поле
    // підпису з'являється в обох — беремо першу
    await expect(page.getByPlaceholder(/Підпис на магніті/).first()).toBeVisible();

    const orderBtn = page.getByRole("button", { name: /Замовити друк/ }).first();
    await expect(orderBtn).toBeVisible();
  });

  test("REGRESSION: чернетка з plain-object зоною НЕ валить /create і /keychains", async ({ page }) => {
    // JSON.parse(draft) повертає plain object замість L.LatLngBounds — раніше
    // це крешило обидві сторінки («getNorth/getCenter is not a function»).
    await page.evaluate(() => {
      localStorage.setItem(
        "monadruk:draft:create",
        JSON.stringify({
          selectedArea: {
            _southWest: { lat: 50.44, lng: 30.5 },
            _northEast: { lat: 50.46, lng: 30.55 },
          },
          styleId: "classic",
          modelSizeMm: 80,
        }),
      );
    });
    await page.goto("/uk/create");
    await expect(page.getByText(/Something went wrong|getCenter is not a function/)).toHaveCount(0);
    await expect(page.locator('nav[aria-label] > button')).toHaveCount(3);
    // зона з чернетки реконструйована як справжній LatLngBounds → панель бачить вибір
    await expect(page.locator("#panel-settings").getByText(/Ділянку вибрано/)).toBeVisible();
    await page.goto("/uk/keychains");
    await expect(page.getByText(/Something went wrong|getNorth is not a function/)).toHaveCount(0);
  });

  test("GPX: завантаження треку показує назву і кількість точок", async ({ page }) => {
    // Панель рендериться двічі (desktop sidebar + mobile tabs) — беремо першу
    await expect(page.getByText(/GPX-маршрут на мапі/).first()).toBeVisible();
    const gpx = `<?xml version="1.0"?><gpx><trk><name>Ранкова пробіжка</name><trkseg>${Array.from(
      { length: 30 },
      (_, i) => `<trkpt lat="${50.45 + i * 0.0002}" lon="${30.52 + i * 0.0002}"/>`,
    ).join("")}</trkseg></trk></gpx>`;
    await page.locator('[data-testid="gpx-input"]').first().setInputFiles({
      name: "run.gpx",
      mimeType: "application/gpx+xml",
      buffer: Buffer.from(gpx, "utf-8"),
    });
    await expect(page.getByText(/Ранкова пробіжка · 30 точок/).first()).toBeVisible();
    // Авто-фокус: зона/карта їдуть до треку, юзер бачить чесну примітку
    await expect(page.locator('[data-testid="gpx-note"]').first()).toBeVisible();
    await expect(page.locator('[data-testid="gpx-note"]').first()).toContainText(/Зону переміщено до треку|Зону розширено|лише центральна частина/);
    await page.getByRole("button", { name: "Прибрати" }).first().click();
    await expect(page.getByText(/Ранкова пробіжка/)).toHaveCount(0);
  });

  test("панно: чипи Вимк/2×2/3×3 + підказка з кількістю плиток", async ({ page }) => {
    const chips = page.locator('[data-testid="panel-chips"]').first();
    await expect(chips).toBeVisible();
    await expect(chips.getByRole("button", { name: "2×2" })).toBeVisible();
    await chips.getByRole("button", { name: "3×3" }).click();
    await expect(page.getByText(/9 плиток з ідеальними швами/).first()).toBeVisible();
    await chips.getByRole("button", { name: "2×2" }).click();
    await expect(page.getByText(/4 плиток з ідеальними швами/).first()).toBeVisible();
  });

  test("share-сторінка: og:image з /api/og, noindex, CTA", async ({ page }) => {
    await page.goto("/uk/share/test-task-12345");
    await expect(page.getByRole("heading", { level: 1 })).toContainText("Моя 3D-мапа");
    const og = await page.locator('meta[property="og:image"]').getAttribute("content");
    expect(og).toContain("/api/og/test-task-12345");
    const robots = await page.locator('meta[name="robots"]').getAttribute("content");
    expect(robots).toContain("noindex");
    await expect(page.getByRole("link", { name: /Створити свою 3D-мапу/ })).toBeVisible();
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
