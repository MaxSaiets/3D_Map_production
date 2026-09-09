/**
 * @jest-environment jsdom
 */
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { BuyFileDialog } from "@/components/BuyFileDialog";
import uk from "@/messages/uk.json";

/**
 * ⭐09.09.2026, рішення власника: друк-файл коштує 149 ₴.
 *
 * Заміряно за 30 днів: модель створили 23 людини, лише 8 з України. Друк і
 * доставка — тільки по Україні, тож дві третини тих, хто зробив усю роботу,
 * не мали що купити. Файл не має логістики — його можна продати куди завгодно.
 *
 * Ціну тести НЕ хардкодять: вона приходить з бекенду (pricing.json), щоб
 * правка ціни не вимагала релізу.
 */
jest.mock("@/lib/analytics", () => ({ track: jest.fn() }));

const sc = (uk as { scenario: Record<string, string> }).scenario;
const PRICE = 149;

function mockFetch(handlers: Record<string, unknown>) {
  return jest.fn(async (url: string, init?: RequestInit) => {
    const key = String(url).includes("/api/file/checkout") ? "checkout" : "access";
    const body = handlers[key];
    if (body === "422") return { ok: false, status: 422, json: async () => ({}) } as Response;
    if (body === "500") return { ok: false, status: 500, json: async () => ({}) } as Response;
    return { ok: true, status: 200, json: async () => body } as Response;
  });
}

beforeEach(() => { jest.restoreAllMocks(); });

describe("BuyFileDialog · купівля друк-файлу", () => {
  it("показує ціну з бекенду, а не зашиту в коді", async () => {
    global.fetch = mockFetch({ access: { paid: false, priceUah: PRICE, currency: "UAH" } }) as never;
    render(<BuyFileDialog taskId="t1" open onClose={() => {}} />);
    await waitFor(() => expect(screen.getByTestId("buy-file-pay")).toHaveTextContent(String(PRICE)));
    expect(screen.getByTestId("buy-file-pay")).toHaveTextContent("₴");
  });

  it("уже оплачений файл не пропонує платити ще раз", async () => {
    global.fetch = mockFetch({ access: { paid: true, priceUah: PRICE, currency: "UAH" } }) as never;
    render(<BuyFileDialog taskId="t1" open onClose={() => {}} />);
    await waitFor(() => expect(screen.getByTestId("buy-file-already")).toHaveTextContent(sc.buyFileAlready));
    expect(screen.queryByTestId("buy-file-pay")).toBeNull();
  });

  it("погана пошта — зрозуміла помилка, а не мовчання", async () => {
    global.fetch = mockFetch({
      access: { paid: false, priceUah: PRICE, currency: "UAH" },
      checkout: "422",
    }) as never;
    const user = userEvent.setup();
    render(<BuyFileDialog taskId="t1" open onClose={() => {}} />);
    await waitFor(() => screen.getByTestId("buy-file-pay"));
    await user.type(screen.getByTestId("buy-file-email"), "не пошта");
    await user.click(screen.getByTestId("buy-file-pay"));
    await waitFor(() => expect(screen.getByTestId("buy-file-error")).toHaveTextContent(sc.buyFileBadEmail));
  });

  it("збій платежу пояснюється, а не лишає порожній екран", async () => {
    global.fetch = mockFetch({
      access: { paid: false, priceUah: PRICE, currency: "UAH" },
      checkout: "500",
    }) as never;
    const user = userEvent.setup();
    render(<BuyFileDialog taskId="t1" open onClose={() => {}} />);
    await waitFor(() => screen.getByTestId("buy-file-pay"));
    await user.type(screen.getByTestId("buy-file-email"), "a@example.com");
    await user.click(screen.getByTestId("buy-file-pay"));
    await waitFor(() => expect(screen.getByTestId("buy-file-error")).toHaveTextContent(sc.buyFileError));
  });

  it("відповідь «вже оплачено» веде одразу до завантаження", async () => {
    global.fetch = mockFetch({
      access: { paid: false, priceUah: PRICE, currency: "UAH" },
      checkout: { alreadyPaid: true, taskId: "t1" },
    }) as never;
    const onAlreadyPaid = jest.fn();
    const user = userEvent.setup();
    render(<BuyFileDialog taskId="t1" open onClose={() => {}} onAlreadyPaid={onAlreadyPaid} />);
    await waitFor(() => screen.getByTestId("buy-file-pay"));
    await user.type(screen.getByTestId("buy-file-email"), "a@example.com");
    await user.click(screen.getByTestId("buy-file-pay"));
    await waitFor(() => expect(onAlreadyPaid).toHaveBeenCalled());
  });

  it("закритий діалог нічого не рендерить і нічого не питає", () => {
    const f = mockFetch({ access: { paid: false, priceUah: PRICE, currency: "UAH" } });
    global.fetch = f as never;
    render(<BuyFileDialog taskId="t1" open={false} onClose={() => {}} />);
    expect(screen.queryByTestId("buy-file-dialog")).toBeNull();
    expect(f).not.toHaveBeenCalled();
  });

  it("усі тексти діалогу перекладені — сирих ключів немає", () => {
    for (const k of ["buyFileTitle", "buyFileBody", "buyFileEmailLabel", "buyFilePay",
                     "buyFileNote", "buyFileAlready", "buyFileBadEmail", "buyFileError", "buyFileClose"]) {
      expect(typeof sc[k]).toBe("string");
      expect(sc[k]).not.toContain("buyFile");
    }
    expect(sc.buyFilePay).toContain("{price}");   // ціна підставляється, не зашита
  });
});
