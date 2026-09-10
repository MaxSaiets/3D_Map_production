import { test, expect } from "@playwright/test";

/**
 * ⭐10.09.2026. Купівля друк-файлу (149 ₴) відбувається БЕЗ входу в акаунт —
 * чек просить лише пошту. А звичайне завантаження вимагає логіну з підтвердженою
 * поштою. Тобто без цієї сторінки людина заплатила б і вперлась у стіну.
 *
 * Пошти в проєкті немає взагалі (ні SMTP, ні сервісу), тож посилання ТУТ і є
 * доставкою. Тому тест перевіряє не «щось відрендерилось», а що кнопка веде
 * саме на оплачений ендпоінт із правильним task_id.
 */
const ORDER = "7100";
const TASK = "task-7100";

test.describe("Оплачений файл на сторінці подяки", () => {
  test("після підтвердженої оплати дає завантажити файл", async ({ page }) => {
    await page.route(`**/api/liqpay/status/${ORDER}`, (r) =>
      r.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ configured: true, paid: true, status: "success", task_id: TASK, amount: 149, currency: "UAH" }),
      }));

    await page.goto(`/uk/order-success?order=${ORDER}&file=1`);
    const block = page.getByTestId("paid-file");
    await expect(block).toBeVisible({ timeout: 15_000 });

    const link = page.getByTestId("paid-file-download");
    await expect(link).toBeVisible();
    await expect(link).toHaveAttribute("href", new RegExp(`/api/file/download/${TASK}$`));
    await expect(block).toContainText("посилання");
  });

  test("оплата без файлу (звичайний друк) кнопки не показує", async ({ page }) => {
    await page.route(`**/api/liqpay/status/${ORDER}`, (r) =>
      r.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ configured: true, paid: true, status: "success", task_id: "", amount: 490, currency: "UAH" }),
      }));

    await page.goto(`/uk/order-success?order=${ORDER}`);
    await expect(page.getByText("#" + ORDER)).toBeVisible({ timeout: 15_000 });
    await expect(page.getByTestId("paid-file")).toHaveCount(0);
  });

  test("поки оплата не підтверджена — файл не пропонується", async ({ page }) => {
    await page.route(`**/api/liqpay/status/${ORDER}`, (r) =>
      r.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ configured: true, paid: false, status: "wait_accept", task_id: TASK }),
      }));

    await page.goto(`/uk/order-success?order=${ORDER}&file=1`);
    await expect(page.getByText("#" + ORDER)).toBeVisible({ timeout: 15_000 });
    await expect(page.getByTestId("paid-file")).toHaveCount(0);
  });
});
