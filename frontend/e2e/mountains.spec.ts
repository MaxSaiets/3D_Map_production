import { test, expect } from "@playwright/test";

/**
 * Режим «Гори» (/mountains) + агент (18.09.2026). Бекенд мокаємо повністю: пресети,
 * фігурки, превʼю, агент, генерацію і статус — тест не тягне DEM і не вантажить VM.
 */
const PRESETS = { presets: [
  { id: "hoverla", lat: 48.16, lon: 24.50028, elev: 2061, area_km: 5, name: "Говерла", country: "Україна", photo: "/mountains/presets/hoverla.jpg" },
  { id: "matterhorn", lat: 45.9764, lon: 7.6586, elev: 4478, area_km: 3.6, name: "Матергорн", country: "Швейцарія", photo: "/mountains/presets/matterhorn.jpg" },
] };
const FIGURES = { figures: [
  { id: "hiker_wave", name: "Альпініст махає рукою", kind: "standing", default_height_mm: 15, min_height_mm: 8, max_height_mm: 40, thumb: "/mountains/figures/hiker_wave.jpg" },
  { id: "climber_rope", name: "Скелелаз на канаті", kind: "climbing", default_height_mm: 15, min_height_mm: 8, max_height_mm: 40, thumb: "/mountains/figures/climber_rope.jpg" },
] };
const PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

async function mockApi(page: import("@playwright/test").Page) {
  await page.route("**/api/mountains/presets**", (r) => r.fulfill({ json: PRESETS }));
  await page.route("**/api/mountains/figures**", (r) => r.fulfill({ json: FIGURES }));
  await page.route("**/api/mountains/preview", (r) => r.fulfill({ json: { png: PNG, elev_min: 1075, elev_max: 2055, relief_m: 980, scale: 25000, relief_mm_natural: 39.2, zexag_for_height: null, sources: ["Copernicus GLO-30"] } }));
  await page.route("**/api/mountains/agent", async (r) => {
    const body = r.request().postDataJSON();
    r.fulfill({ json: {
      spec: { ...(body.base || {}), place: { name: "Говерла", lat: 48.16, lon: 24.50028, source: "preset", preset_id: "hoverla", area_km: 5 }, area_km: 5, size_mm: 150, height_mm: null,
        frame: { style: "none", width_mm: 0, height_mm: 25 }, sides: "rock", figures: [{ id: "climber_rope", where: "steepest", height_mm: 11 }], texture: "satellite" },
      understood: ["Місце: Говерла — з бібліотеки вершин.", "Плита 150 × 150 мм.", "Без ободка: рельєф до самого краю.", "Фігурка: Скелелаз на канаті, 11 мм, на найкрутішій стіні."],
      warnings: [], questions: [], confidence: 0.7, source: "rules" } });
  });
  await page.route("**/api/mountains/generate", (r) => r.fulfill({ json: { task_id: "mnt-test-1", status: "processing", message: "ok" } }));
  await page.route("**/api/status/mnt-test-1", (r) => r.fulfill({ json: {
    task_id: "mnt-test-1", status: "completed", progress: 100, message: "Готово", download_url: "/files/mountain_mnttest1_print.3mf", download_url_3mf: "/files/mountain_mnttest1_print.3mf", download_url_glb: "/files/mountain_mnttest1.glb",
    world_spec: { mode: "mountain", place: "Говерла", scale: 43215, zexag: 1.0, height_mm: 51, size_mm: 150, sources: ["Copernicus GLO-30"], tiles: [], figures: [], tiles_zip: null, preview_png: "/files/mountain_mnttest1_preview.png", paint_jpg: "/files/mountain_mnttest1_paint.jpg", seconds: 40, watertight: true } } }));
  // GLB не існує — вʼюер покаже скелет; нам важливі кнопки й підпис
  await page.route("**/files/**", (r) => r.fulfill({ status: 404, body: "" }));
}

test.describe("Гори (/mountains)", () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => { try { document.cookie = "mnd_consent=denied;path=/"; } catch { /* ignore */ } });
    await mockApi(page);
  });

  test("Говерла обрана одразу → вигляд → генерація → результат із файлами", async ({ page }) => {
    await page.goto("/uk/mountains");
    await expect(page.getByTestId("beta-banner")).toContainText("Тестовий режим");
    // 25.09: без порожнього стану — перша вершина вже обрана, превʼю й підсумок видно одразу
    await expect(page.getByTestId("mnt-place")).toContainText("Говерла");
    await expect(page.getByTestId("mnt-preview")).toBeVisible();
    await expect(page.getByTestId("mnt-summary")).toContainText("Говерла · 20×20 см · 1:25 000");
    await expect(page.getByTestId("mnt-price-info")).toContainText("Ціна індивідуальна");
    await expect(page.getByTestId("mnt-price-info")).toContainText("Можемо розмалювати");
    await page.getByTestId("mnt-preset-matterhorn").click();
    await expect(page.getByTestId("mnt-place")).toContainText("Матергорн");
    await page.getByTestId("mnt-fig-add-hiker_wave").click();
    await expect(page.getByTestId("mnt-fig-add-hiker_wave")).toHaveAttribute("aria-pressed", "true");
    await page.getByTestId("mnt-style-bare").click();
    await page.getByTestId("mnt-advanced").click();
    await expect(page.getByTestId("mnt-frame-none")).toHaveAttribute("aria-checked", "true");
    await expect(page.getByTestId("mnt-fig-list")).toContainText("Альпініст махає рукою");
    await page.getByTestId("mnt-generate").first().click();
    await expect(page.getByTestId("mnt-built")).toContainText("Говерла · 1:43 215 · 51 мм");
    // 25.09: друк-файл — кнопка через вхід (3 гори безкоштовно), а не пряме посилання на /api/download
    await expect(page.getByTestId("mnt-dl-print")).toBeVisible();
    await expect(page.getByTestId("mnt-result").locator('a[href*="format=3mf"]')).toHaveCount(0);
    await expect(page.getByTestId("mnt-dl-note")).toContainText("3 гори безкоштовно");
    await expect(page.getByTestId("mnt-want-paint")).toBeVisible();
    await expect(page.getByTestId("mnt-result").getByRole("link", { name: /Гайд розпису/ })).toBeVisible();
    await expect(page.getByTestId("mnt-order")).toBeVisible();
  });

  test("пошук гори за назвою обирає місце, мапа завжди видима", async ({ page }) => {
    await page.route("**/api/mountains/search**", (r) => r.fulfill({ json: { results: [
      { name: "Ай-Петрі", lat: 44.45, lon: 34.06, source: "geocode", type: "peak", area_km: 4, display: "Крим, Україна" }] } }));
    await page.goto("/uk/mountains");
    await expect(page.getByTestId("mnt-place")).toContainText("Говерла");
    await page.getByTestId("mnt-search").fill("Ай-Петрі");
    await page.getByTestId("mnt-search-results").getByText("Ай-Петрі").click();
    await expect(page.getByTestId("mnt-place")).toContainText("Ай-Петрі");
    await expect(page.getByTestId("mountain-map")).toBeVisible();
  });

  test("агент: показує «як зрозумів» і заповнює форму лише після «Застосувати»", async ({ page }) => {
    await page.goto("/uk/mountains");
    await expect(page.getByTestId("mnt-place")).toContainText("Говерла");
    await page.getByTestId("mnt-agent-toggle").click();
    await page.getByTestId("mnt-agent-input").fill("Говерла 15 см без ободка, скельні боки, скелелаз");
    await page.getByTestId("mnt-agent-run").click();
    const ans = page.getByTestId("mnt-agent-answer");
    await expect(ans).toContainText("Як я зрозумів");
    await expect(ans).toContainText("Без ободка");
    // до «Застосувати» форма НЕ змінена
    await expect(page.getByTestId("mnt-style-classic")).toHaveAttribute("aria-checked", "true");
    await expect(page.getByTestId("mnt-size-200")).toHaveAttribute("aria-checked", "true");
    await page.getByTestId("mnt-agent-apply").click();
    await expect(page.getByTestId("mnt-size-150")).toHaveAttribute("aria-checked", "true");
    await expect(page.getByTestId("mnt-fig-add-climber_rope")).toHaveAttribute("aria-pressed", "true");
    await page.getByTestId("mnt-advanced").click();
    await expect(page.getByTestId("mnt-frame-none")).toHaveAttribute("aria-checked", "true");
    await expect(page.getByTestId("mnt-sides-rock")).toHaveAttribute("aria-checked", "true");
    await expect(page.getByTestId("mnt-fig-list")).toContainText("Скелелаз на канаті");
  });

  test("«Уточнити на мапі»: клік ставить центр і показує превʼю", async ({ page }) => {
    await page.goto("/uk/mountains");
    const map = page.getByTestId("mountain-map");
    await expect(map).toBeVisible();
    const canvas = map.locator(".leaflet-container");
    await expect(canvas.locator(".leaflet-tile-loaded").first()).toBeVisible({ timeout: 15000 });
    await canvas.click({ position: { x: 250, y: 170 } });
    await expect(page.getByTestId("mnt-place")).toContainText(/\d+\.\d{4}, \d+\.\d{4}/);
    await expect(page.getByTestId("mnt-preview")).toBeVisible();
  });
});

test.describe("Світи: агент", () => {
  test("агент пояснює форму і застосовує її до чипів", async ({ page }) => {
    await page.addInitScript(() => { try { document.cookie = "mnd_consent=denied;path=/"; } catch { /* ignore */ } });
    await page.route("**/api/worlds/agent", (r) => r.fulfill({ json: { spec: { shape: "volcano", shapeUk: "вулкан", size_mm: 180, max_height_mm: 28 }, understood: ["Форма: вулкан (за описом)."], warnings: [], questions: [], confidence: 0.65, source: "rules" } }));
    await page.goto("/uk/worlds");
    await page.getByTestId("world-agent-input").fill("острів-вулкан у морі");
    await page.getByTestId("world-agent-run").click();
    await expect(page.getByTestId("world-agent-answer")).toContainText("вулкан");
    await expect(page.getByTestId("world-shape-auto")).toHaveAttribute("aria-checked", "true");
    await page.getByTestId("world-agent-apply").click();
    await expect(page.getByTestId("world-shape-volcano")).toHaveAttribute("aria-checked", "true");
    await expect(page.locator("#world-prompt")).toHaveValue("острів-вулкан у морі");
  });
});
