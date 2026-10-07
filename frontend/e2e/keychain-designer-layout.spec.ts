import { test, expect } from "@playwright/test";

/**
 * 07.10.2026. Макет брелока (KeychainDesigner): на серці три підписи розмірів лягали
 * один на одний, «Зворот» мав темний текст на темному тлі (клас `text-white/72` поза
 * шкалою Tailwind не генерувався), а підказка «Тягни карту…» залазила під перемикач.
 */
for (const vp of [{ width: 375, height: 812 }, { width: 1280, height: 860 }]) {
  test(`макет брелока читабельний (${vp.width}px)`, async ({ page }) => {
    await page.setViewportSize(vp);
    await page.goto("/uk/keychains");
    const svg = page.getByTestId("keychain-designer-svg");
    await expect(svg).toBeVisible({ timeout: 60_000 });

    const r = await page.evaluate(() => {
      const s = document.querySelector('[data-testid="keychain-designer-svg"]')!;
      const root = s.parentElement!;
      const back = [...root.querySelectorAll("button")].find((b) => /Зворот/.test(b.textContent || ""))!;
      const hint = root.querySelector("div.pointer-events-none")!.getBoundingClientRect();
      const toggle = back.parentElement!.getBoundingClientRect();
      const boxes = [...s.querySelectorAll("text")]
        .filter((t) => /mm/.test(t.textContent || ""))
        .map((t) => ({ s: t.textContent, b: t.getBoundingClientRect() }))
        .filter((t) => t.b.width > 0);
      const overlaps: string[] = [];
      for (let i = 0; i < boxes.length; i++) for (let j = i + 1; j < boxes.length; j++) {
        const a = boxes[i].b, c = boxes[j].b;
        if (a.left < c.right && c.left < a.right && a.top < c.bottom && c.top < a.bottom) overlaps.push(`${boxes[i].s} / ${boxes[j].s}`);
      }
      return { backColor: getComputedStyle(back).color, hintRight: hint.right, toggleLeft: toggle.left, overlaps };
    });
    expect(r.overlaps).toEqual([]);
    expect(r.hintRight).toBeLessThanOrEqual(r.toggleLeft);
    // світлий текст на темному тлі: червоний канал високий (rgb/rgba 255,…)
    expect(r.backColor).toMatch(/^rgba?\(255, 255, 255/);
  });
}
