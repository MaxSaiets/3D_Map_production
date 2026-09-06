/**
 * A/B-інфраструктура (lib/ab.ts): детермінований варіант на visitorId,
 * приблизний баланс 50/50 на великій вибірці, SSR-safe дефолт "A".
 */
import { getVariant, abProps, ACTIVE_EXPERIMENTS } from "@/lib/ab";

describe("getVariant", () => {
  it("is deterministic for the same visitor id", () => {
    for (const id of ["abc123", "xyz789", "0000000000", "user-42"]) {
      const v1 = variantForId(id, "cta");
      const v2 = variantForId(id, "cta");
      expect(v1).toBe(v2);
    }
  });

  it("is roughly balanced 40-60% over 2000 ids", () => {
    let aCount = 0;
    const total = 2000;
    for (let i = 0; i < total; i++) {
      const id = `visitor-${i}-${Math.random().toString(36).slice(2)}`;
      if (variantForId(id, "cta") === "A") aCount++;
    }
    const pct = (aCount / total) * 100;
    expect(pct).toBeGreaterThanOrEqual(40);
    expect(pct).toBeLessThanOrEqual(60);
  });

  it("returns A on the server (no window)", () => {
    const originalWindow = global.window;
    // @ts-expect-error simulate SSR
    delete global.window;
    try {
      expect(getVariant("cta")).toBe("A");
    } finally {
      global.window = originalWindow;
    }
  });
});

describe("abProps", () => {
  it("returns {} on the server", () => {
    const originalWindow = global.window;
    // @ts-expect-error simulate SSR
    delete global.window;
    try {
      expect(abProps()).toEqual({});
    } finally {
      global.window = originalWindow;
    }
  });

  it("returns ab_<exp> for every active experiment on the client", () => {
    window.localStorage.setItem("mnd_vid", "fixed-test-visitor");
    const props = abProps();
    for (const exp of ACTIVE_EXPERIMENTS) {
      expect(props[`ab_${exp}`]).toMatch(/^[AB]$/);
    }
  });
});

// Helper: exercise getVariant with a specific visitor id by seeding localStorage.
function variantForId(id: string, exp: string): "A" | "B" {
  window.localStorage.setItem("mnd_vid", id);
  return getVariant(exp);
}
