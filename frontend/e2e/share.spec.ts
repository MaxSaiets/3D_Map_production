import { test, expect } from "@playwright/test";

/**
 * Share-сторінка `/share/{taskId}`: клієнтський ShareViewer підвантажує
 * `/api/share/{taskId}` і або монтує 3D-вʼювер (glb_url є), або лишає
 * статичний OG-скріншот із поясненням «модель недоступна» (glb_url = null,
 * напр. спливли 90 днів зберігання файлів).
 *
 * Ми НЕ мокаємо реальний вміст .glb (мінімальний валідний бінарник важко
 * зібрати надійно) — перевіряємо лише що контейнер вʼювера змонтувався і
 * текстові підказки (expires/unavailable) на місці.
 */
test.describe("Share сторінка", () => {
  test("glb_url є → монтується 3D-вʼювер + рядок «діє 90 днів»", async ({ page }) => {
    await page.route("**/api/share/t-share-1", (route) => route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ task_id: "t-share-1", glb_url: "/files/e2e.glb", png_url: null, product: "map" }),
    }));
    await page.route("**/files/e2e.glb", (route) => route.fulfill({
      status: 200,
      contentType: "model/gltf-binary",
      body: Buffer.from([]),
    }));
    await page.goto("/uk/share/t-share-1");
    await expect(page.getByTestId("share-viewer")).toBeVisible();
    // i18n-патч (share.expires/unavailable/viewerHint) ще не застосовано в
    // messages/*.json — до застосування next-intl рендерить сирий ключ,
    // тож перевіряємо контейнер за testid, а не за перекладеним текстом.
    await expect(page.getByTestId("share-expires")).toBeVisible();
  });

  test("404 (модель прострочена) → лишається картинка + «модель уже недоступна»", async ({ page }) => {
    await page.route("**/api/share/t-share-404", (route) => route.fulfill({
      status: 404,
      contentType: "application/json",
      body: JSON.stringify({ detail: "not found" }),
    }));
    await page.goto("/uk/share/t-share-404");
    await expect(page.getByTestId("share-viewer")).toHaveCount(0);
    await expect(page.getByTestId("share-unavailable")).toBeVisible();
  });
});
