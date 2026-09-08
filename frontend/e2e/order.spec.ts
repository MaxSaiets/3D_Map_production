import { test, expect } from "@playwright/test";

/**
 * F-6 (2026-09-05): грошовий шлях — замовлення БЕЗ входу. Бекенд `/api/order`
 * навмисно не вимагає авторизації (`main.py:1302`, логін лише мʼяко привʼязує
 * замовлення до кабінету), тож анонімний покупець мусить дійти до номера
 * замовлення. Раніше цей шлях не мав жодного тесту — регресія тут = нуль продажів.
 * Реальних запитів не шлемо: /api/order і статус задачі замоковані.
 */
test.describe("Замовлення без входу", () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => {
      try {
        localStorage.clear();
        localStorage.setItem("intro_create_v1", "1");
        localStorage.setItem("onb_create_v1", "1");
        document.cookie = "mnd_consent=denied;path=/";
      } catch { /* ignore */ }
    });
    await page.route("**/api/generate", (r) => r.fulfill({
      status: 200, contentType: "application/json",
      body: JSON.stringify({ task_id: "t-order", status: "processing", eta_s: 50 }),
    }));
    await page.route("**/api/status/t-order", (r) => r.fulfill({
      status: 200, contentType: "application/json",
      body: JSON.stringify({
        task_id: "t-order", status: "completed", progress: 100, message: "Готово",
        download_url: "/files/x.glb", eta_s: 50, elapsed_s: 48,
      }),
    }));
  });

  test("анонім проходить від готової моделі до номера замовлення", async ({ page }) => {
    let ordered: any = null;
    await page.route("**/api/order", async (r) => {
      ordered = JSON.parse(r.request().postData() || "{}");
      await r.fulfill({
        status: 200, contentType: "application/json",
        body: JSON.stringify({ ok: true, order_number: 4242, payment: null }),
      });
    });

    await page.goto("/uk/create?product=map3d");
    await page.getByTestId("scenario-create").click();
    await expect(page.getByTestId("guided-success")).toBeVisible({ timeout: 20_000 });

    // Замовлення відкривається з банера — той самий шлях, що й у покупця.
    await page.getByTestId("guided-order").click();
    const dialog = page.getByRole("dialog");
    await expect(dialog).toHaveCount(1);

    await dialog.locator("#order-name").fill("Тест Тестенко");
    await dialog.locator("#order-phone").fill("+380931234567");
    // Нова Пошта тягне живий пікер відділень (мережа) — для тесту беремо Укрпошту:
    // там прості поля міста/індексу/адреси, і саме цей шлях бек валідує суворіше.
    await dialog.getByRole("radio", { name: /Укрпошта/ }).click();
    await dialog.locator("#order-city").fill("Київ");
    await dialog.locator("#order-branch").fill("01001");
    await dialog.locator("#order-address").fill("вул. Тестова, 1");
    await dialog.getByRole("button", { name: /Оформити|Замовити/ }).last().click();

    // Успіх: номер замовлення + кнопка копіювання + кроки «що далі»
    await expect(dialog.getByText("4242")).toBeVisible({ timeout: 15_000 });
    await expect(dialog.getByText(/Скопіювати номер/)).toBeVisible();
    await expect(dialog.getByText(/Підтвердимо/)).toBeVisible();

    // Запит справді пішов без авторизації і з даними форми
    expect(ordered).toBeTruthy();
    expect(String(ordered.name)).toContain("Тест");
    expect(String(ordered.phone)).toContain("380931234567");
  });
});

/**
 * ⭐08.09.2026: чек LiqPay жив ЛИШЕ у стані OrderDialog. Клієнт ішов на LiqPay,
 * не завершував оплату і повертався на /order-success — там був статус «очікує
 * оплати» і ЖОДНОЇ кнопки заплатити. Тепер OrderDialog зберігає чек локально
 * (`mnd_pay_<номер>`), а сторінка подяки дає ним доплатити.
 */
test.describe("Сторінка подяки: доплатити пізніше", () => {
  test("у стані «очікує оплати» є кнопка оплати зі збереженого чека", async ({ page }) => {
    await page.route("**/api/liqpay/status/**", (r) =>
      r.fulfill({ json: { configured: true, paid: false, status: "pending" } }));
    await page.addInitScript(() => {
      try {
        localStorage.setItem("mnd_pay_4242", JSON.stringify({
          provider: "liqpay", action_url: "https://www.liqpay.ua/api/3/checkout",
          data: "ZGF0YQ==", signature: "sig", amount: 490, ts: Date.now(),
        }));
        document.cookie = "mnd_consent=denied;path=/";
      } catch { /* ignore */ }
    });
    await page.goto("/uk/order-success?order=4242");
    await expect(page.getByTestId("pay-again")).toBeVisible({ timeout: 30_000 });
    await expect(page.getByText("Посилання на оплату збережено у цьому браузері.")).toBeVisible();
  });

  test("без збереженого чека кнопки немає (нема що показувати)", async ({ page }) => {
    await page.route("**/api/liqpay/status/**", (r) =>
      r.fulfill({ json: { configured: true, paid: false, status: "pending" } }));
    await page.addInitScript(() => {
      try { document.cookie = "mnd_consent=denied;path=/"; } catch { /* ignore */ }
    });
    await page.goto("/uk/order-success?order=9999");
    await expect(page.getByText("#9999")).toBeVisible({ timeout: 30_000 });
    await expect(page.getByTestId("pay-again")).toHaveCount(0);
  });
});
