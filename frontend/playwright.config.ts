import { defineConfig } from "@playwright/test";

/**
 * e2e шляху покупця. Запуск: npm run test:e2e
 * Сам піднімає dev-сервер (reuse якщо вже запущений на :3000).
 * Бекенд НЕ обовʼязковий: ціни мають статичний fallback, генерацію не чіпаємо.
 */
export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  expect: { timeout: 15_000 },
  retries: 1,
  workers: 1,
  use: {
    baseURL: "http://localhost:3000",
    viewport: { width: 1280, height: 800 },
    // 07.10.2026: next-intl визначає мову за Accept-Language; без цього браузер тесту (en-US)
    // отримував англійську навіть на /uk/... і ВСІ перевірки українських текстів падали.
    locale: "uk-UA",
    trace: "retain-on-failure",
  },
  webServer: {
    command: "npm run dev",
    url: "http://localhost:3000",
    reuseExistingServer: true,
    timeout: 120_000,
  },
});
