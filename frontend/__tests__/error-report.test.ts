/**
 * @jest-environment jsdom
 */
import { shouldReportError, isExtensionNoise, stackHint, BENIGN_ERROR } from "@/lib/analytics";

/**
 * ⭐09.09.2026. За ВЕСЬ прод-лог сайт зібрав рівно 19 JS-помилок:
 *    8×  unhandledrejection: TypeError: Cannot read properties of undefined (reading 'map')
 *    6×  unhandledrejection: TypeError: Failed to fetch
 *    4×  unhandledrejection: i: Failed to connect to MetaMask      ← чуже розширення
 *    1×  CompileError: Wasm code generation disallowed by embedder
 * І в кожної — порожній `src`: обробник `unhandledrejection` викликав звіт без
 * другого аргументу. Тобто найчастішу помилку (вісім разів, і вона НАША)
 * не було де шукати, а п'яту частину журналу займав чужий крипто-гаманець.
 */
describe("stackHint · звідки прилетіла помилка", () => {
  it("бере перший справжній кадр і прибирає хост", () => {
    const err = new Error("boom");
    err.stack = [
      "TypeError: Cannot read properties of undefined (reading 'map')",
      "    at https://monadruk.com/_next/static/chunks/create-1a2b.js:14:2201",
      "    at https://monadruk.com/_next/static/chunks/main.js:2:100",
    ].join("\n");
    expect(stackHint(err)).toBe("/_next/static/chunks/create-1a2b.js:14:2201");
  });

  it("перший рядок стека — це повідомлення, а не кадр: його пропускаємо", () => {
    const err = new Error("x");
    err.stack = "Error: x\n    at foo (/app/page.js:3:11)";
    expect(stackHint(err)).toContain("/app/page.js:3:11");
    expect(stackHint(err)).not.toContain("Error: x");
  });

  it("причина без стека (рядок, число, null) не ламає звіт", () => {
    for (const reason of ["просто рядок", 42, null, undefined, {}]) {
      expect(stackHint(reason)).toBe("");
    }
  });

  it("обрізає надто довгий кадр", () => {
    const err = new Error("x");
    err.stack = "Error: x\n    at " + "a".repeat(400) + ".js:1:1";
    expect(stackHint(err, 120).length).toBeLessThanOrEqual(120);
  });
});

describe("isExtensionNoise · чужі розширення браузера", () => {
  it.each([
    ["i: Failed to connect to MetaMask", ""],
    ["TypeError: x", "chrome-extension://abcdef/inpage.js:1:1"],
    ["TypeError: x", "moz-extension://abcdef/inpage.js:1:1"],
    ["Extension context invalidated.", ""],
  ])("впізнає шум: %s", (msg, stack) => {
    expect(isExtensionNoise(msg, stack)).toBe(true);
    expect(shouldReportError(msg, stack)).toBe(false);
  });

  it("НЕ чіпає наші помилки", () => {
    const msg = "unhandledrejection: TypeError: Cannot read properties of undefined (reading 'map')";
    const src = "/_next/static/chunks/create-1a2b.js:14:2201";
    expect(isExtensionNoise(msg, src)).toBe(false);
    expect(shouldReportError(msg, src)).toBe(true);
  });
});

describe("shouldReportError · що доходить до /admin", () => {
  it("нешкідливий браузерний шум лишається відфільтрованим", () => {
    for (const msg of [
      "Connection to Indexed Database server lost",
      "ResizeObserver loop completed with undelivered notifications",
      "Script error.",
      "Load failed",
    ]) {
      expect(BENIGN_ERROR.test(msg)).toBe(true);
      expect(shouldReportError(msg)).toBe(false);
    }
  });

  it("порожнє повідомлення не надсилаємо", () => {
    expect(shouldReportError("")).toBe(false);
  });

  it("реальні помилки з прод-логу проходять", () => {
    for (const msg of [
      "unhandledrejection: TypeError: Cannot read properties of undefined (reading 'map')",
      "unhandledrejection: TypeError: Failed to fetch",
      "unhandledrejection: CompileError: Wasm code generation disallowed by embedder",
    ]) {
      expect(shouldReportError(msg)).toBe(true);
    }
  });
});
