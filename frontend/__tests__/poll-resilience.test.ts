import { isTransientStatusError, POLL_FAILS_GONE, POLL_FAILS_TRANSIENT, POLL_FAILS_NOTICE } from "@/lib/poll";

/**
 * ⭐09.09.2026, з логів Caddy за тиждень: 377 відповідей 5xx, усі 502, і всі
 * вікнами — 03.09 22:00 (151), 06.09 16–17:00 (155). Це рестарти бекенду й
 * інцидент OOM. Серед них **80 поспіль** на `/api/status/<одна задача>`: людина
 * опитувала статус кожні 2.5 с, поки сервер лежав.
 *
 * Опитувач здавався після 4 невдач (~10 с) і писав «модель застаріла» — хоча
 * рестарт триває 15–25 с, а `/api/status` уміє віддати готову модель із диска
 * (`main.py`, гілка `disk_file` перед 404). Тобто ми оголошували втрату там,
 * де файл лежав готовий.
 */
const httpError = (status: number) => ({ response: { status } });

describe("isTransientStatusError · чекати чи здаватись", () => {
  it.each([500, 502, 503, 504, 429, 408])("%d — сервер піднімається, чекаємо", (status) => {
    expect(isTransientStatusError(httpError(status))).toBe(true);
  });

  it.each([404, 410])("%d — задачі справді немає, чекати нема чого", (status) => {
    expect(isTransientStatusError(httpError(status))).toBe(false);
  });

  it("обрив мережі (немає відповіді взагалі) вважаємо тимчасовим", () => {
    expect(isTransientStatusError(new Error("Network Error"))).toBe(true);
    expect(isTransientStatusError(null)).toBe(true);
    expect(isTransientStatusError(undefined)).toBe(true);
    expect(isTransientStatusError({})).toBe(true);
  });

  it("400/401/403 — не тимчасові: повторювати їх безглуздо", () => {
    for (const s of [400, 401, 403]) {
      expect(isTransientStatusError(httpError(s))).toBe(false);
    }
  });
});

describe("пороги очікування", () => {
  it("вікно для рестарту перекриває реальні 15–25 с простою", () => {
    const stepMs = 2500;
    const windowS = (POLL_FAILS_TRANSIENT * stepMs) / 1000;
    expect(windowS).toBeGreaterThanOrEqual(60);
    expect(windowS).toBeLessThanOrEqual(180); // але не тримаємо людину вічно
  });

  it("для «задачі немає» лишається швидка здача (~10 с)", () => {
    expect((POLL_FAILS_GONE * 2500) / 1000).toBeLessThanOrEqual(15);
    expect(POLL_FAILS_GONE).toBeLessThan(POLL_FAILS_TRANSIENT);
  });

  it("повідомлення «зв'язок перервався» з'являється раніше за здачу", () => {
    expect(POLL_FAILS_NOTICE).toBeLessThan(POLL_FAILS_GONE);
    expect(POLL_FAILS_NOTICE).toBeGreaterThan(0);
  });
});

describe("тексти для людини", () => {
  it("у всіх шести локалях є пояснення простою", () => {
    const locales = ["uk", "en", "de", "pl", "fr", "es"];
    for (const loc of locales) {
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      const m = require(`@/messages/${loc}.json`);
      for (const ns of ["scenario", "kcScenario"]) {
        expect(typeof m[ns].reconnecting).toBe("string");
        expect(m[ns].reconnecting.length).toBeGreaterThan(20);
      }
    }
  });
});
