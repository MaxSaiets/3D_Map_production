import { test, expect, type Page } from "@playwright/test";

/**
 * 07.10.2026. Підписка Monadruk Pro: продаюча сторінка, оформлення і кабінет.
 * Вхід імітується dev-хуком AuthProvider (localStorage `__mnd_test_user`, працює
 * лише в dev-збірці), відповіді бекенду — моками page.route.
 */
// Локаль uk: інакше next-intl визначає мову за Accept-Language браузера тесту (en-US).
test.use({ locale: "uk-UA" });

const PLANS = { plans: { UAH: 2100, USD: 50 }, suggested: "UAH", country: "UA", configured: true, file: { UAH: 149, USD: 3.6 } };

async function mockPlans(page: Page) {
  await page.route("**/api/subscription/plans", (r) => r.fulfill({ json: PLANS }));
}
async function loginAsTester(page: Page) {
  await page.addInitScript(() => {
    localStorage.setItem("__mnd_test_user", JSON.stringify({ uid: "e2e-uid", email: "tester@example.com" }));
  });
}

test.describe("Сторінка Pro", () => {
  test("гість бачить ціну, калькулятор і кнопку входу", async ({ page }) => {
    await mockPlans(page);
    await page.goto("/uk/pro");
    await expect(page.getByRole("heading", { level: 1 })).toContainText("Безлім файлів");
    await expect(page.getByTestId("pro-price")).toContainText("2");
    await expect(page.getByTestId("pro-login")).toBeVisible();

    // Калькулятор: 5 файлів — поки дешевше поштучно; 30 — Pro вигідніший.
    const range = page.getByTestId("pro-calc-range");
    await range.fill("5");
    await expect(page.getByTestId("pro-calc-verdict")).toContainText("15-го файлу");
    await range.fill("30");
    await expect(page.getByTestId("pro-calc-verdict")).toContainText("заощаджуєте");
    // 30 × 149 − 2100 = 2370
    await expect(page.getByTestId("pro-calc-verdict")).toContainText(/2\s370/);
  });

  test("оформлення вимагає всіх трьох згод і веде на LiqPay", async ({ page }) => {
    await mockPlans(page);
    await loginAsTester(page);
    await page.route("**/api/subscription", (r) => r.fulfill({ json: { subscription: null } }));
    let checkoutBody: Record<string, unknown> | null = null;
    await page.route("**/api/subscription/checkout", async (r) => {
      checkoutBody = JSON.parse(r.request().postData() || "{}");
      await r.fulfill({ json: { payment: { action_url: "https://www.liqpay.ua/api/3/checkout", data: "DATA", signature: "SIG" } } });
    });
    let liqpayHit = false;
    await page.route("https://www.liqpay.ua/**", (r) => { liqpayHit = true; return r.fulfill({ status: 200, body: "liqpay" }); });

    await page.goto("/uk/pro");
    const btn = page.getByTestId("pro-subscribe");
    await expect(btn).toBeDisabled();
    const boxes = page.locator("#checkout input[type=checkbox]");
    await expect(boxes).toHaveCount(3);
    await boxes.nth(0).check();
    await boxes.nth(1).check();
    await expect(btn).toBeDisabled();
    await boxes.nth(2).check();
    await expect(btn).toBeEnabled();
    await btn.click();
    await expect.poll(() => liqpayHit).toBe(true);
    expect(checkoutBody).toMatchObject({ currency: "UAH", accept_terms: true, accept_autorenew: true, accept_digital: true });
    expect((checkoutBody as unknown as { consent_texts: string[] }).consent_texts).toHaveLength(3);
  });

  test("активна підписка: статус і скасування", async ({ page }) => {
    await mockPlans(page);
    await loginAsTester(page);
    const active = { status: "active", active: true, amount: 2100, currency: "UAH", renews_at: "2026-11-07T00:00:00Z", paid_until: "2026-11-07T00:00:00Z", payments: [{ amount: 2100, currency: "UAH", ts: "2026-10-07T10:00:00Z" }] };
    await page.route("**/api/subscription", (r) => r.fulfill({ json: { subscription: active } }));
    await page.route("**/api/subscription/cancel", (r) => r.fulfill({ json: { subscription: { ...active, status: "cancelled", cancelled_at: "2026-10-07T11:00:00Z" } } }));
    await page.goto("/uk/pro");
    await expect(page.getByTestId("pro-status")).toContainText("Підписка Pro активна");
    page.once("dialog", (d) => d.accept());
    await page.getByRole("button", { name: "Скасувати підписку" }).click();
    await expect(page.getByTestId("pro-status")).toContainText("Підписку скасовано");
  });

  test("сторінка цін: файл платний, є рядок і плашка Pro", async ({ page }) => {
    await page.goto("/uk/prices");
    await expect(page.getByText("Безкоштовно*")).toHaveCount(0);
    await expect(page.getByText(/2\s100 ₴ \/ міс/).first()).toBeVisible();
    const block = page.getByTestId("prices-pro");
    await expect(block).toContainText("15-го файлу");
    await block.getByRole("link").click();
    await expect(page).toHaveURL(/\/pro$/);
  });
});

test.describe("Кабінет і Pro", () => {
  const quota = (extra: Record<string, unknown>) => ({
    user: { email: "tester@example.com", is_admin: false, subscription_active: false, ...extra },
    quota: { downloads: 0, limit: 0, remaining: 0, is_admin: false, can_download: false },
  });

  test("без підписки: ціна файлу замість «0 / 0» і картка Pro", async ({ page }) => {
    await loginAsTester(page);
    await page.route("**/api/account/quota", (r) => r.fulfill({ json: quota({}) }));
    await page.route("**/api/account/models", (r) => r.fulfill({ json: { models: [] } }));
    await page.route("**/api/account/orders", (r) => r.fulfill({ json: { orders: [] } }));
    await page.route("**/api/account/grids**", (r) => r.fulfill({ json: { grids: [] } }));
    await page.goto("/uk/account");
    await expect(page.getByTestId("account-downloads")).toContainText("149 ₴ за файл");
    await expect(page.getByTestId("account-downloads")).not.toContainText("0 / 0");
    await expect(page.getByTestId("account-pro-card")).toBeVisible();
  });

  test("з активною підпискою: статус Pro без рекламної картки", async ({ page }) => {
    await loginAsTester(page);
    await page.route("**/api/account/quota", (r) => r.fulfill({ json: {
      user: { email: "tester@example.com", is_admin: false, subscription_active: true },
      quota: { downloads: 3, limit: 0, remaining: 1e9, is_admin: true, can_download: true },
    } }));
    await page.route("**/api/account/models", (r) => r.fulfill({ json: { models: [] } }));
    await page.route("**/api/account/orders", (r) => r.fulfill({ json: { orders: [] } }));
    await page.route("**/api/account/grids**", (r) => r.fulfill({ json: { grids: [] } }));
    await page.goto("/uk/account");
    await expect(page.getByTestId("account-pro-active")).toBeVisible();
    await expect(page.getByTestId("account-pro-card")).toHaveCount(0);
    await expect(page.getByTestId("account-downloads")).toContainText("Безліміт");
  });
});
