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

  test("крок 1: 4 картки-продукти + видимий список «Що ще ми вміємо»", async ({ page }) => {
    await page.goto("/uk/create");
    const flow = page.getByTestId("scenario-flow");
    await expect(flow).toBeVisible();
    await expect(flow.getByText("Що створюємо?")).toBeVisible();
    // Рівно 4 картки-продукти з фото (решта можливостей — список нижче)
    await expect(flow.locator("img")).toHaveCount(4);
    // Власник: «не зрозуміло, які взагалі можливості» → список має бути ВИДИМИЙ
    // і перелічувати всі інші продукти, а не ховатись у трьох дрібних лінках.
    await expect(flow.getByText("Що ще ми вміємо")).toBeVisible();
    const more = flow.getByTestId("scenario-more");
    await expect(more).toBeVisible();
    await expect(more.locator("> *")).toHaveCount(6);
    for (const label of ["Брелок з моїм місцем", "Панно на стіну", "Макет квартири з плану", "3D-світ за описом", "Готові моделі"]) {
      await expect(more.getByText(label, { exact: false })).toBeVisible();
    }
    await expect(flow.getByTestId("scenario-full")).toBeVisible();
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
    // Орієнтир «як це працює» — щоб було зрозуміло, що робить рамка на карті
    await expect(flow.getByText(/рамка на карті = що надрукуємо/)).toBeVisible();
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

/**
 * A-1…A-6 (2026-09-03): єдина шапка, товар до конструктора, CTA завжди активна,
 * «готово» = 2 дії, один розширений режим.
 */
test.describe("Guided /create — хвиля «простіше» (2026-09-03)", () => {
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

  test("A-1: builder-шапка з перемикачем мови на /create і /keychains; звичайна шапка на /prices", async ({ page }) => {
    await page.goto("/uk/create");
    const hdr = page.getByTestId("site-header-builder");
    await expect(hdr).toBeVisible();
    await expect(hdr.getByRole("button", { name: "Мова" })).toBeVisible();
    await expect(hdr.getByRole("link", { name: /Брелок/ })).toBeVisible();
    const box = await hdr.boundingBox();
    expect(box!.height).toBeLessThanOrEqual(64);
    await page.goto("/uk/keychains");
    await expect(page.getByTestId("site-header-builder")).toBeVisible();
    await page.goto("/uk/prices");
    await expect(page.locator("header").first()).toBeVisible();
    await expect(page.getByRole("button", { name: "Мова" }).first()).toBeVisible();
  });

  test("A-2: ?product=relief відкриває одразу крок 2 з рельєфною мапою", async ({ page }) => {
    await page.goto("/uk/create?product=relief");
    const flow = page.getByTestId("scenario-flow");
    await expect(flow.getByText("Крок 2 із 2")).toBeVisible();
    await expect(flow.getByRole("radio", { name: /M · 8 см/ })).toContainText("410 ₴");
  });

  test("A-4: CTA активна без жодної дії, бейдж каже «Центр Києва (за замовчуванням)»", async ({ page }) => {
    await page.goto("/uk/create?product=map3d");
    const flow = page.getByTestId("scenario-flow");
    await expect(flow.getByTestId("place-default")).toContainText("Центр Києва");
    await expect(page.getByTestId("scenario-create")).toBeEnabled();
    await flow.getByRole("button", { name: "Одеса", exact: true }).click();
    await expect(flow.getByTestId("place-picked")).toContainText("Місце обрано: Одеса");
  });

  test("A-3: «готово» = замовити + завантажити; «Оновити превʼю» лише після зміни", async ({ page }) => {
    await page.route("**/api/generate", (route) => route.fulfill({
      status: 200, contentType: "application/json",
      body: JSON.stringify({ task_id: "t-e2e-1", status: "processing", message: "ok", eta_s: 50 }),
    }));
    await page.route("**/api/status/t-e2e-1", (route) => route.fulfill({
      status: 200, contentType: "application/json",
      body: JSON.stringify({ task_id: "t-e2e-1", status: "completed", progress: 100, message: "done", download_url: "/files/e2e.glb", eta_s: 50, elapsed_s: 49 }),
    }));
    await page.goto("/uk/create?product=map3d");
    const flow = page.getByTestId("scenario-flow");
    await flow.getByRole("button", { name: "Львів", exact: true }).click();
    await page.waitForTimeout(1800); // доліт карти + авто-зона під розмір
    await page.getByTestId("scenario-create").click();
    const success = flow.getByTestId("guided-success");
    await expect(success).toBeVisible({ timeout: 15_000 });
    await expect(success.getByTestId("guided-order")).toContainText(/Замовити друк · \d+ ₴/);
    await expect(success.getByTestId("guided-download")).toBeVisible();
    await expect(success.getByText("Підлаштувати деталі")).toHaveCount(0);
    await expect(success.getByText("Створити ще одну")).toHaveCount(0);
    // Нічого не міняли → кнопки «Оновити превʼю» нема (sticky-бар — лише <lg, див. мобільний describe)
    await expect(page.getByTestId("scenario-create")).toHaveCount(0);
    // Змінили розмір → зʼявляється «Оновити превʼю»
    await flow.getByRole("radio", { name: /L · 11 см/ }).click();
    await expect(page.getByTestId("scenario-create")).toHaveText(/Оновити превʼю/);
  });

  test("A-6: єдиний вихід «Розширений режим»; ?mode=pro відкриває його одразу", async ({ page }) => {
    await page.goto("/uk/create");
    const flow = page.getByTestId("scenario-flow");
    await expect(flow.getByTestId("scenario-full")).toHaveText("Розширений режим");
    await expect(flow.getByText("Повний конструктор")).toHaveCount(0);
    await page.goto("/uk/create?mode=pro");
    await expect(page.getByTestId("scenario-flow")).toHaveCount(0);
  });
});

/** C-1…C-5 (2026-09-03): логіка ходу створення — черга, скасування, помилка з
 *  причиною та діями, прогрес друк-файлу, відновлення після перезавантаження. */
test.describe("Guided /create — хід створення (2026-09-03)", () => {
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

  test("C-4: стан «у черзі» показано окремо від прогресу", async ({ page }) => {
    await page.route("**/api/generate", (r) => r.fulfill({ status: 200, contentType: "application/json",
      body: JSON.stringify({ task_id: "t-q", status: "processing", eta_s: 90 }) }));
    await page.route("**/api/status/t-q", (r) => r.fulfill({ status: 200, contentType: "application/json",
      body: JSON.stringify({ task_id: "t-q", status: "queued", progress: 0, message: "У черзі", eta_s: 90, elapsed_s: 3 }) }));
    await page.goto("/uk/create?product=map3d");
    await page.getByTestId("scenario-create").click();
    const stages = page.getByTestId("generation-stages");
    await expect(stages).toBeVisible({ timeout: 15_000 });
    await expect(stages.getByTestId("gen-queued")).toBeVisible();
    await expect(stages).toContainText("У черзі");
    await expect(stages.getByTestId("gen-cancel")).toBeVisible();
  });

  test("C-3: помилка показує причину з бекенду і дії", async ({ page }) => {
    await page.route("**/api/generate", (r) => r.fulfill({ status: 200, contentType: "application/json",
      body: JSON.stringify({ task_id: "t-e", status: "processing", eta_s: 60 }) }));
    await page.route("**/api/status/t-e", (r) => r.fulfill({ status: 200, contentType: "application/json",
      body: JSON.stringify({ task_id: "t-e", status: "failed", progress: 0,
        message: "Зона завелика для моделі 8 см: виберіть меншу ділянку", eta_s: 60, elapsed_s: 5 }) }));
    await page.goto("/uk/create?product=map3d");
    await page.getByTestId("scenario-create").click();
    const err = page.getByTestId("guided-error");
    await expect(err).toBeVisible({ timeout: 15_000 });
    await expect(err).toContainText("Зона завелика");
    await expect(err.getByTestId("guided-retry")).toBeVisible();
    await expect(err.getByRole("button", { name: "Зменшити ділянку" })).toBeVisible();
  });

  test("C-1: після перезавантаження готова модель показується без повторної генерації", async ({ page }) => {
    await page.route("**/api/status/t-done", (r) => r.fulfill({ status: 200, contentType: "application/json",
      body: JSON.stringify({ task_id: "t-done", status: "completed", progress: 100, message: "Готово",
        download_url: "/files/restored.glb", eta_s: 50, elapsed_s: 50 }) }));
    await page.addInitScript(() => {
      localStorage.setItem("3dmap_task_group_id", "t-done");
      localStorage.setItem("3dmap_task_ids", JSON.stringify(["t-done"]));
      localStorage.setItem("3dmap_task_product", "map");
    });
    await page.goto("/uk/create?product=map3d");
    await expect(page.getByTestId("guided-success")).toBeVisible({ timeout: 20_000 });
    await expect(page.getByTestId("guided-order")).toBeVisible();
  });

  test("D-1/D-3: смуга повзе між стрибками сервера; сцена не дублює прогрес", async ({ page }) => {
    await page.route("**/api/generate", (r) => r.fulfill({ status: 200, contentType: "application/json",
      body: JSON.stringify({ task_id: "t-sm", status: "processing", eta_s: 60 }) }));
    await page.route("**/api/status/t-sm", (r) => r.fulfill({ status: 200, contentType: "application/json",
      body: JSON.stringify({ task_id: "t-sm", status: "processing", progress: 20, message: "Будую рельєф", eta_s: 60, elapsed_s: 10 }) }));
    await page.goto("/uk/create?product=map3d");
    await page.getByTestId("scenario-create").click();
    const stages = page.getByTestId("generation-stages");
    await expect(stages).toBeVisible({ timeout: 15_000 });
    const pct = async () => Number((await stages.locator("[role=progressbar]").getAttribute("aria-valuenow")) || 0);
    const first = await pct();
    await page.waitForTimeout(3000);
    const later = await pct();
    expect(later).toBeGreaterThan(first);   // повзе між стрибками
    expect(later).toBeLessThanOrEqual(27);  // але не обганяє сервер більш ніж на 7 п.п.
    // D-3: у guided прогрес рівно один — оверлей сцени не рендериться
    await expect(page.getByText(/^Генерація моделі/)).toHaveCount(0);
  });
});

test.describe("Guided /keychains — зона за замовчуванням (E-1, 2026-09-04)", () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => {
      try {
        localStorage.clear();
        localStorage.setItem("intro_keychain_v1", "1");
        localStorage.setItem("onb_keychain_v1", "1");
        document.cookie = "mnd_consent=denied;path=/";
      } catch { /* ignore */ }
    });
  });

  test("CTA активна одразу після вибору шаблону (раніше не вмикалась ніколи)", async ({ page }) => {
    await page.goto("/uk/keychains");
    const flow = page.getByTestId("kc-scenario-flow");
    await expect(flow).toBeVisible();
    await flow.getByTestId("kc-scenario-heart-46").click();
    const cta = page.getByTestId("kc-scenario-create");
    await expect(cta).toBeEnabled({ timeout: 20_000 });
    await expect(flow.getByTestId("kc-place-default")).toContainText("Центр Києва");
    // Зміна шаблону не має стирати рамку (скид setSelectedArea(null) прибрано)
    await flow.getByRole("button", { name: "Назад" }).click();
    await flow.getByTestId("kc-scenario-classic-wide").click();
    await expect(cta).toBeEnabled({ timeout: 20_000 });
  });
});
