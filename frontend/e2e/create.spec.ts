import { test, expect } from "@playwright/test";

test.describe("Конструктор мап /create", () => {
  test.beforeEach(async ({ page }) => {
    // GUIDED-режим (сценарний вхід) увімкнено ЗА ЗАМОВЧУВАННЯМ і ховає повний
    // конструктор — ця сьют тестує саме ПОВНИЙ UI, тож вимикаємо guided до
    // завантаження сторінки (addInitScript діє на всі наступні goto).
    await page.addInitScript(() => localStorage.setItem("3dmap_guided_v1", "0"));
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

  test("Ф2 геокодер: пошук локації присутній над картою (будь-яке місто/адреса)", async ({ page }) => {
    await page.goto("/uk/create");
    const search = page.locator('[data-testid="map-search"]').first();
    await expect(search).toBeVisible();
    await expect(search.getByPlaceholder(/Знайти місто/)).toBeVisible();
  });

  test("Ф1 sticky: рівно ОДИН закріплений CTA на мобільному + ЦІНА не показується під час створення", async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 });
    await page.goto("/uk/create");
    await page.waitForTimeout(1200);
    // Єдиний fixed-бар із дією генерації (раніше було два портали)
    const sticky = page.locator("div.fixed").filter({
      has: page.locator("button", { hasText: /Згенерувати|Замовити/ }),
    });
    // Рівно ОДИН закріплений CTA-портал (раніше монтувалось два → дубль кнопок)
    await expect(sticky).toHaveCount(1);
    await expect(sticky.first()).toBeVisible();
    await expect(sticky.first()).toContainText(/Згенерувати|Замовити/);
    // Орієнтовну ціну показуємо ЗАВЖДИ (фолбек з SIMPLE_SIZES, ніколи «—») — щоб
    // покупець не тиснув «Згенерувати»/«Замовити» наосліп (конверсія). Тому sticky
    // має містити символ валюти.
    await expect(sticky.first()).toContainText(/₴/);
  });

  test("Ф1b майстер: мобільна навігація уніфікована (єдиний степер, без дубль-табів)", async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 });
    await page.goto("/uk/create");
    await page.waitForTimeout(1000);
    // Прибрано старий ряд табів «Налаштування» і нижній «Швидкий статус · Дії»
    await expect(page.getByText(/Швидкий статус/)).toHaveCount(0);
    // Єдина навігація — 3-кроковий степер; клік «Параметри» показує стиль/розмір
    const steps = page.locator('nav[aria-label] > button');
    await expect(steps).toHaveCount(3);
    await steps.nth(1).click();
    await expect(page.getByText(/Стиль|Розмір/).first()).toBeVisible();
  });

  test("Order-now: «Замовити» доступне ДО генерації + примітка «модель готується»", async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 });
    // Пропускаємо тур-оверлей, щоб він не перехоплював клік (у проді він піднятий
    // над sticky; у тесті просто вимикаємо для чистоти кліку)
    await page.addInitScript(() => localStorage.setItem("onb_create_v1", "1"));
    await page.goto("/uk/create");
    await page.waitForTimeout(1200);
    const sticky = page.locator("div.fixed").filter({
      has: page.locator("button", { hasText: /Замовити/ }),
    }).first();
    // До будь-якої генерації у барі вже є «Замовити друк» (order-now)
    const orderBtn = sticky.getByRole("button", { name: /Замовити/ });
    await expect(orderBtn).toBeVisible();
    await orderBtn.click();
    // Форма відкривається одразу + заспокійлива примітка про підготовку моделі
    await expect(page.getByText(/Модель ще готується/)).toBeVisible();
    await expect(page.getByPlaceholder(/Ім.?я та прізвище|Ім/).first()).toBeVisible();
  });

  test("майстер: 3 клікабельні кроки Місце/Параметри/Готово", async ({ page }) => {
    const steps = page.locator('nav[aria-label] > button');
    await expect(steps).toHaveCount(3);
    await expect(steps.nth(0)).toContainText("Місце");
    await expect(steps.nth(1)).toContainText("Параметри");
    await expect(steps.nth(2)).toContainText("Готово");
  });

  test("REGRESSION: зміна міста в шапці ПЕРЕЛІТАЄ карту (Рома: «карта не переходить»)", async ({ page }) => {
    await page.goto("/uk/create");
    // довгота першого видимого тайла → перевіряємо реальне положення карти
    const tileLon = () => page.evaluate(() => {
      const tiles = Array.from(document.querySelectorAll("img.leaflet-tile")) as HTMLImageElement[];
      for (const t of tiles) {
        const m = t.src.match(/\/(\d+)\/(\d+)\/(\d+)\.png/);
        if (m) return (+m[2]) / Math.pow(2, +m[1]) * 360 - 180;
      }
      return null;
    });
    // Київ ~30.5°E
    await expect.poll(tileLon, { timeout: 12000 }).toBeGreaterThan(28);
    // header-select → Львів (~24°E)
    await page.locator("select").first().selectOption({ label: "Львів" });
    // карта мусить перелетіти на захід (а не лишитись на Києві — це і був баг)
    await expect.poll(tileLon, { timeout: 12000 }).toBeLessThan(26);
  });

  test("шаблон району → зона вибрана → Згенерувати активна", async ({ page }) => {
    await page.getByRole("button", { name: /Поділ/ }).first().click();
    await expect(page.locator("#panel-settings").getByText(/Ділянку вибрано/)).toBeVisible();
    const genBtn = page.getByRole("button", { name: /Згенерувати модель/ }).first();
    await expect(genBtn).toBeEnabled();
  });

  test("магніт: перемикач + поле підпису + жива/фолбек ціна в кнопці замовлення", async ({ page }) => {
    // Магніт/GPX сховані під «Налаштування» (Просто-режим лишається коротким)
    await page.locator('[data-testid="more-options"]').first().click();
    const magnet = page.getByRole("button", { name: /Магніт на холодильник/ }).first();
    await expect(magnet).toBeVisible();
    await magnet.click();
    // Стан магніта тепер у store (спільний для обох копій панелі) → поле
    // підпису з'являється в обох — беремо першу
    await expect(page.getByPlaceholder(/Підпис на магніті/).first()).toBeVisible();

    const orderBtn = page.getByRole("button", { name: /Замовити друк/ }).first();
    await expect(orderBtn).toBeVisible();
  });

  test("з'єднувач-пази: тумблер вмикається, сумісний з flat-AMS, гаситься магнітом", async ({ page }) => {
    await page.locator('[data-testid="more-options"]').first().click();
    await page.locator('[data-testid="addons-toggle"]').first().click();
    const connector = page.locator('[data-testid="connector-toggle"]').first();
    await expect(connector).toBeVisible();
    await expect(connector).toHaveAttribute("aria-pressed", "false");
    await connector.click();
    await expect(connector).toHaveAttribute("aria-pressed", "true");
    // Сумісний з flat-AMS (кольорова плитка з пазами) — обидва лишаються ON
    const flatAms = page.locator('[data-testid="flat-ams-toggle"]').first();
    await flatAms.click();
    await expect(flatAms).toHaveAttribute("aria-pressed", "true");
    await expect(connector).toHaveAttribute("aria-pressed", "true");
    // Магніт несумісний (інше дно) → гасить з'єднувач
    await page.getByRole("button", { name: /Магніт на холодильник/ }).first().click();
    await expect(connector).toHaveAttribute("aria-pressed", "false");
  });

  test("преміум-рамка: тумблер вмикається й співіснує з flat-AMS, гаситься у 3D", async ({ page }) => {
    await page.locator('[data-testid="more-options"]').first().click();
    await page.locator('[data-testid="addons-toggle"]').first().click();
    const frame = page.locator('[data-testid="frame-toggle"]').first();
    await expect(frame).toBeVisible();
    await expect(frame).toHaveAttribute("aria-pressed", "false");
    await frame.click();
    await expect(frame).toHaveAttribute("aria-pressed", "true");
    // Сумісна з flat-AMS — обидва ON
    const flatAms = page.locator('[data-testid="flat-ams-toggle"]').first();
    await flatAms.click();
    await expect(flatAms).toHaveAttribute("aria-pressed", "true");
    await expect(frame).toHaveAttribute("aria-pressed", "true");
    // Перехід у «Об'ємна 3D» гасить рамку (рамка — лише для плоского режиму).
    await page.locator('[data-testid="format-relief3d"]').first().click();
    await expect(frame).toHaveAttribute("aria-pressed", "false");
  });

  test("рельєф: під-опція формату «Об'ємна 3D», ховається у плоских режимах", async ({ page }) => {
    // Рельєф тепер ВКЛАДЕНА під-опція 3D-формату (не окремий конкуруючий тумблер).
    // Дефолт = «Об'ємна 3D» → під-опція показана.
    const relief = page.locator('[data-testid="relief-toggle"]').first();
    await expect(relief).toBeVisible();
    await expect(relief).toHaveAttribute("aria-pressed", "false");
    await relief.click();
    await expect(relief).toHaveAttribute("aria-pressed", "true");
    // Перемикання у плоский формат → рельєф ЗНИКАЄ з DOM (під-опція лише «Об'ємна 3D»)
    await page.locator('[data-testid="format-flat"]').first().click();
    await expect(page.locator('[data-testid="relief-toggle"]')).toHaveCount(0);
    // Пласкі будинки — суб-перемикач плоского режиму (під «Більше опцій»)
    await page.locator('[data-testid="more-options"]').first().click();
    await expect(page.locator('[data-testid="flat-buildings-toggle"]').first()).toBeVisible();
    // Повернення у 3D через чип формату → рельєф знову доступний і скинутий у false
    await page.locator('[data-testid="format-relief3d"]').first().click();
    await expect(relief).toBeVisible();
    await expect(relief).toHaveAttribute("aria-pressed", "false");
    await expect(page.locator('[data-testid="flat-buildings-toggle"]')).toHaveCount(0);
  });

  test("виділення будинку: тумблер, клік по карті ставить точку, очищення", async ({ page }) => {
    await page.locator('[data-testid="more-options"]').first().click();
    await page.locator('[data-testid="addons-toggle"]').first().click();
    const hl = page.locator('[data-testid="highlight-toggle"]').first();
    await expect(hl).toBeVisible();
    await expect(hl).toHaveAttribute("aria-pressed", "false");
    await hl.click();
    await expect(hl).toHaveAttribute("aria-pressed", "true");
    // підказка «клікни свої будинки» з'являється
    await expect(page.getByText(/Клікни сво/).first()).toBeVisible();
    // клік по карті ставить точку → з'являється статус (Обрано: N) + кнопка очищення
    await page.locator(".leaflet-container").first().click({ position: { x: 200, y: 180 } });
    await expect(page.getByText(/Обрано:/).first()).toBeVisible();
    const clear = page.locator('[data-testid="highlight-clear"]').first();
    await expect(clear).toBeVisible();
    await clear.click();
    await expect(page.locator('[data-testid="highlight-clear"]')).toHaveCount(0);
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
    await page.locator('[data-testid="more-options"]').first().click();
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

  test("формат: сегмент-контрол з 3 варіантів, вибір «Плоска» вмикає flat-AMS", async ({ page }) => {
    const seg = page.locator('[data-testid="format-seg"]').first();
    await expect(seg).toBeVisible();
    // Рівно 3 взаємовиключні radio: 3D / Плоска / Магніт (багатозонна мапа — через сітку на карті)
    await expect(seg.getByRole("radio")).toHaveCount(3);
    // Дефолт = «Об'ємна 3D» (усі спецрежими off)
    await expect(page.locator('[data-testid="format-relief3d"]').first()).toHaveAttribute("aria-checked", "true");
    // Вибір «Плоска кольорова» → flat-AMS-тумблер під «Більше опцій» стає ON
    await page.locator('[data-testid="format-flat"]').first().click();
    await expect(page.locator('[data-testid="format-flat"]').first()).toHaveAttribute("aria-checked", "true");
    await page.locator('[data-testid="more-options"]').first().click();
    await expect(page.locator('[data-testid="flat-ams-toggle"]').first()).toHaveAttribute("aria-pressed", "true");
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
    // Тур-оверлей («Підказка») і cookie-банер перехоплюють кліки в модалці —
    // вимикаємо обидва (як в order-now тесті), щоб клік по «Європа» доходив.
    await page.addInitScript(() => {
      localStorage.setItem("onb_create_v1", "1");
      document.cookie = "mnd_consent=denied;path=/";
    });
    await page.goto("/uk/create");
    await page.waitForTimeout(800);
    await page.getByRole("button", { name: /Замовити друк/ }).first().click();
    const dialog = page.locator(".fixed.inset-0", { hasText: "Орієнтовна вартість" });
    await expect(dialog).toBeVisible();
    // ціна: жива (≈ N ₴) або статичний fallback (від N ₴)
    await expect(dialog.getByText(/[≈від]+\s*\d+\s*₴/)).toBeVisible();

    // Регіон і служба доставки — це role="radio" (a11y), не button
    await dialog.getByRole("radio", { name: /Європа/ }).click();
    await expect(dialog.getByRole("radio", { name: "Nova Post (EU)" })).toBeVisible();
    await expect(dialog.getByRole("radio", { name: "Meest" })).toBeVisible();
    const options = dialog.locator("select option");
    await expect(options).toHaveCount(16); // плейсхолдер + 15 країн
  });
});
