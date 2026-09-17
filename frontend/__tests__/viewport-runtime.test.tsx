/**
 * AppViewportHeight: `--app-vh` НЕ має ставати 0px, коли вкладка змонтувалась
 * без розміру (фонова вкладка / прихована панель / prerender) — інакше карта
 * конструктора отримує висоту 0 і крок 2 виглядає порожнім (знайдено 16.09.2026).
 */
import { act, render } from "@testing-library/react";
import { AppViewportHeight } from "@/components/ViewportRuntime";

const setInnerHeight = (h: number) => {
  Object.defineProperty(window, "innerHeight", { configurable: true, value: h });
};

describe("AppViewportHeight", () => {
  afterEach(() => {
    document.documentElement.style.removeProperty("--app-vh");
  });

  it("не записує 0px і підхоплює перше чесне значення на resize", () => {
    setInnerHeight(0);
    render(<AppViewportHeight />);
    expect(document.documentElement.style.getPropertyValue("--app-vh")).toBe("");
    setInnerHeight(720);
    act(() => { window.dispatchEvent(new Event("resize")); });
    act(() => { jest.advanceTimersByTime?.(200); });
    // debounce 120 мс — чекаємо реальним таймером
    return new Promise<void>((resolve) => setTimeout(() => {
      expect(document.documentElement.style.getPropertyValue("--app-vh")).toBe("720px");
      resolve();
    }, 200));
  });

  it("на маунті з нормальним вікном пише висоту одразу", () => {
    setInnerHeight(812);
    render(<AppViewportHeight />);
    expect(document.documentElement.style.getPropertyValue("--app-vh")).toBe("812px");
  });
});
