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

  test("крок 1: 8 карток-рендерів (4 мапи + гора/панно/брелок/світ) + список «Що ще ми вміємо»", async ({ page }) => {
    await page.goto("/uk/create");
    const flow = page.getByTestId("scenario-flow");
    await expect(flow).toBeVisible();
    await expect(flow.getByText("Що створюємо?")).toBeVisible();
    // 18.09.2026: 8 карток — РЕНДЕРИ моделей (card-r-*), а не фото; 4 сценарії мап + 4 переходи
    await expect(flow.locator("img")).toHaveCount(8);
    await expect(flow.locator("img").first()).toHaveAttribute("src", /card-r-/);
    await expect(flow.getByTestId("scenario-card-mountain")).toHaveAttribute("href", /\/mountains/);
    await expect(flow.getByTestId("scenario-card-keychain")).toHaveAttribute("href", /\/keychains/);
    // Власник: «не зрозуміло, які взагалі можливості» → список має бути ВИДИМИЙ
    // і перелічувати всі інші продукти, а не ховатись у трьох дрібних лінках.
    await expect(flow.getByText("Що ще ми вміємо")).toBeVisible();
    const more = flow.getByTestId("scenario-more");
    await expect(more).toBeVisible();
    await expect(more.locator("> *")).toHaveCount(3);
    for (const label of ["Макет квартири з плану", "Готові моделі"]) {
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
    // A/B (lib/ab.ts, R-05): лейбл CTA залежить від visitor-id → приймаємо обидва варіанти.
    await expect(cta).toHaveText(/Показати (3D-превʼю|мою 3D-мапу) · безкоштовно/);
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
    await expect(flow.getByRole("radio", { name: /M · 8 см/ })).toContainText("575 ₴");
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
    // S-1/S-2: месенджер-замовлення (uk-локаль → без рядка «лише по Україні») + опитування
    // «що заважає» зʼявляється після кліку «Завантажити» (подія guided-download).
    const alt = success.getByTestId("sales-alternatives");
    await expect(alt.getByTestId("msg-telegram")).toBeVisible();
    await expect(alt.getByTestId("msg-instagram")).toBeVisible();
    await expect(alt.getByTestId("ua-only-note")).toHaveCount(0);
    await expect(alt.getByTestId("why-not-order")).toHaveCount(0);
    await page.evaluate(() => window.dispatchEvent(new Event("monadruk:guided-download")));
    await expect(alt.getByTestId("why-not-order")).toBeVisible();
    // Подія download відкриває модалку входу поверх усього — закриваємо її перед кліком по чипу.
    await page.keyboard.press("Escape");
    await alt.getByTestId("why-look").click({ force: true });
    // ⭐09.09.2026: замість глухого «Дякуємо» — відповідь на конкретну причину
    // («дивлюсь» → що саме можна змінити). Загальний why-thanks лишився тільки
    // для невідомої причини зі старого localStorage.
    const reply = alt.getByTestId("why-reply-look");
    await expect(reply).toBeVisible();
    await expect(reply).toContainText("генерація безкоштовна");
    await expect(alt.getByTestId("why-thanks")).toHaveCount(0);
    await expect(success.getByText("Підлаштувати деталі")).toHaveCount(0);
    await expect(success.getByText("Створити ще одну")).toHaveCount(0);
    // Нічого не міняли → кнопки «Оновити превʼю» нема (sticky-бар — лише <lg, див. мобільний describe)
    await expect(page.getByTestId("scenario-create")).toHaveCount(0);
    // Змінили розмір → зʼявляється «Оновити превʼю»
    await flow.getByRole("radio", { name: /L · 11 см/ }).click();
    await expect(page.getByTestId("scenario-create")).toHaveText(/Оновити превʼю/);
  });

  test("19.09: крок 2 = 4 пронумеровані секції; свій розмір 50–200 мм з живою ціною; секція «Вигляд» за сценарієм", async ({ page }) => {
    await page.goto("/uk/create?product=relief");
    const flow = page.getByTestId("scenario-flow");
    for (const n of [1, 2, 3, 4]) await expect(flow.getByTestId(`guided-sec-${n}`)).toBeVisible();
    await expect(flow.getByTestId("guided-sec-3")).toContainText("Висота рельєфу");
    await expect(flow.getByTestId("guided-sec-3")).toContainText("Висота будинків");
    // Довільний розмір: 125 мм → ціна лінійно між L(630) і XL(770) = 680 + рельєф 85 = 765
    const input = flow.getByTestId("size-input");
    await input.fill("125");
    await input.press("Enter");
    await expect(flow.getByTestId("custom-size-price")).toHaveText("765 ₴");
    await expect(flow.getByTestId("custom-size")).toContainText("12.5 см · ділянка ≈875 м");
    await expect(flow.getByText(/Друк 765 ₴/)).toBeVisible();
    // Пресет повертає поле до 80
    await flow.getByRole("radio", { name: /M · 8 см/ }).click();
    await expect(input).toHaveValue("80");
    // Поза межами → кламп до 200
    await input.fill("999");
    await input.press("Enter");
    await expect(input).toHaveValue("200");
    // Вигляд: вибір висоти рельєфу
    await flow.getByRole("radio", { name: /Виразна/ }).click();
    await expect(flow.getByRole("radio", { name: /Виразна/ })).toHaveAttribute("aria-checked", "true");
    // ?size= приймає будь-яке значення 50–200; map3d — без «Висота рельєфу», flat — «Будинки»/рамка
    await page.goto("/uk/create?product=map3d&size=125");
    await expect(page.getByTestId("scenario-flow").getByTestId("size-input")).toHaveValue("125");
    await expect(page.getByTestId("scenario-flow").getByTestId("guided-sec-3")).not.toContainText("Висота рельєфу");
    await page.goto("/uk/create?product=flat");
    await expect(page.getByTestId("scenario-flow").getByTestId("flat-buildings")).toBeVisible();
    await expect(page.getByTestId("scenario-flow").getByTestId("frame-toggle")).toBeVisible();
  });

  test("A-6: єдиний вихід «Розширений режим»; ?mode=pro відкриває його одразу", async ({ page }) => {
    await page.goto("/uk/create");
    const flow = page.getByTestId("scenario-flow");
    await expect(flow.getByTestId("scenario-full")).toContainText("Розширений режим");
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
    // Без queue_eta_s лишається старий загальний текст — числа не вигадуємо.
    await expect(stages.getByTestId("gen-queued")).toContainText("за кілька хвилин");
    await expect(stages.getByTestId("gen-cancel")).toBeVisible();
  });

  test("C-4b: коли сервер знає, скільки чекати — показує число, а не «кілька хвилин»", async ({ page }) => {
    // ⭐09.09.2026: прод 08.09 показав очікування 546…2609 с (43 хв) під незмінним
    // написом «за кілька хвилин». Тепер бекенд віддає queue_eta_s.
    await page.route("**/api/generate", (r) => r.fulfill({ status: 200, contentType: "application/json",
      body: JSON.stringify({ task_id: "t-q2", status: "processing", eta_s: 90 }) }));
    await page.route("**/api/status/t-q2", (r) => r.fulfill({ status: 200, contentType: "application/json",
      body: JSON.stringify({ task_id: "t-q2", status: "queued", progress: 0, message: "У черзі",
        eta_s: 90, elapsed_s: 3, queue_eta_s: 780 }) }));
    await page.goto("/uk/create?product=map3d");
    await page.getByTestId("scenario-create").click();
    const queued = page.getByTestId("generation-stages").getByTestId("gen-queued");
    await expect(queued).toBeVisible({ timeout: 15_000 });
    await expect(queued).toContainText("13 хв");
    await expect(queued).not.toContainText("за кілька хвилин");
  });

  test("C-4c: менше хвилини очікування — окремий текст без числа", async ({ page }) => {
    await page.route("**/api/generate", (r) => r.fulfill({ status: 200, contentType: "application/json",
      body: JSON.stringify({ task_id: "t-q3", status: "processing", eta_s: 90 }) }));
    await page.route("**/api/status/t-q3", (r) => r.fulfill({ status: 200, contentType: "application/json",
      body: JSON.stringify({ task_id: "t-q3", status: "queued", progress: 0, message: "У черзі",
        eta_s: 90, elapsed_s: 3, queue_eta_s: 35 }) }));
    await page.goto("/uk/create?product=map3d");
    await page.getByTestId("scenario-create").click();
    const queued = page.getByTestId("generation-stages").getByTestId("gen-queued");
    await expect(queued).toBeVisible({ timeout: 15_000 });
    await expect(queued).toContainText("ось-ось");
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
    // I-1/I-2/I-3: фраза-обіцянка, «Поділитись 3D» з uk-текстом (не сирий ключ), QR на десктопі
    await expect(page.getByTestId("guided-promise")).toContainText("з тих самих даних");
    await expect(page.getByTestId("guided-share")).toHaveText(/Поділитись 3D/);
    const qr = page.getByTestId("share-qr");
    await expect(qr).toBeVisible();
    await expect(qr.locator("img")).toHaveAttribute("src", /^data:image\/png/, { timeout: 10_000 });
    await expect(qr).toContainText("на телефоні");
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

test.describe("Guided /create — тап до готовності карти (H-4)", () => {
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

  test("CTA активна ще до появи рамки; намір не губиться", async ({ page }) => {
    let gen = 0;
    await page.route("**/api/generate", (r) => {
      gen += 1;
      return r.fulfill({ status: 200, contentType: "application/json",
        body: JSON.stringify({ task_id: "t-h4", status: "processing", eta_s: 50 }) });
    });
    await page.route("**/api/status/t-h4", (r) => r.fulfill({
      status: 200, contentType: "application/json",
      body: JSON.stringify({ task_id: "t-h4", status: "processing", progress: 20, message: "Будую", eta_s: 50, elapsed_s: 4 }),
    }));

    await page.goto("/uk/create?product=map3d");
    const cta = page.getByTestId("scenario-create");
    await cta.waitFor({ state: "attached" });
    // Кнопка НЕ мертва одразу (раніше була disabled, поки Leaflet не віддасть рамку)
    await expect(cta).toBeEnabled();
    await cta.click({ force: true });
    // Чи то рамка вже була, чи ні — генерація мусить стартувати рівно один раз
    await expect(page.getByTestId("generation-stages")).toBeVisible({ timeout: 30_000 });
    await expect(page.getByTestId("map-loading-wait")).toHaveCount(0);
    expect(gen).toBe(1);
  });
});

test.describe("Guided /create — купівля файлу без акаунта (2026-09-16)", () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => {
      try {
        localStorage.clear();
        localStorage.setItem("intro_create_v1", "1");
        localStorage.setItem("onb_create_v1", "1");
      } catch { /* ignore */ }
    });
  });

  test("гість без безкоштовних завантажень: «Завантажити» → друк-файл → діалог оплати 149 ₴ (без входу)", async ({ page }) => {
    let gen = 0;
    await page.route("**/api/generate", (route) => {
      gen += 1;
      route.fulfill({
        status: 200, contentType: "application/json",
        body: JSON.stringify({ task_id: gen === 1 ? "t-e2e-1" : "t-e2e-2", status: "processing", message: "ok", eta_s: 50 }),
      });
    });
    await page.route("**/api/status/t-e2e-1", (route) => route.fulfill({
      status: 200, contentType: "application/json",
      body: JSON.stringify({ task_id: "t-e2e-1", status: "completed", progress: 100, message: "done", download_url: "/files/e2e.glb", eta_s: 50, elapsed_s: 49 }),
    }));
    await page.route("**/api/status/t-e2e-2", (route) => route.fulfill({
      status: 200, contentType: "application/json",
      body: JSON.stringify({ task_id: "t-e2e-2", status: "completed", progress: 100, message: "done", download_url: "/files/e2e.3mf", download_url_3mf: "/files/e2e.3mf" }),
    }));
    // FREE_DOWNLOADS=0 на проді: файл лише за гроші, акаунт для купівлі не потрібен
    await page.route("**/api/file/access/**", (route) => route.fulfill({
      status: 200, contentType: "application/json",
      body: JSON.stringify({ paid: false, priceUah: 149, currency: "UAH", freeLimit: 0, approx: { EUR: 3.1 } }),
    }));
    await page.goto("/uk/create?product=map3d");
    const flow = page.getByTestId("scenario-flow");
    await flow.getByRole("button", { name: "Львів", exact: true }).click();
    await page.waitForTimeout(1800);
    await page.getByTestId("scenario-create").click();
    const success = flow.getByTestId("guided-success");
    await expect(success).toBeVisible({ timeout: 15_000 });
    await success.getByTestId("guided-download").click();
    // діалог купівлі, а не модалка входу і не форма замовлення друку
    const pay = page.getByTestId("buy-file-pay");
    await expect(pay).toBeVisible({ timeout: 20_000 });
    await expect(pay).toContainText("149");
    await expect(page.locator("#login-dialog-title")).toHaveCount(0);
    await expect(page.locator("#order-name")).toHaveCount(0);
    expect(gen).toBe(2);
  });
});
