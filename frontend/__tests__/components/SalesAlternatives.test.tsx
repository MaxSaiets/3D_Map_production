/**
 * @jest-environment jsdom
 */
import { render, screen, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SalesAlternatives } from "@/components/SalesAlternatives";
import uk from "@/messages/uk.json";

/**
 * ⭐09.09.2026. Опитування «що заважає замовити» — єдиний прямий зворотний
 * звʼязок, який сайт збирає. 07.09 реальна людина (прийшла з Google, натиснула
 * «замовити в Telegram» на мапі за 770 ₴) обрала «Надрукую сам» — і побачила
 * глухе «Дякуємо». Тепер на кожну причину є відповідь по суті.
 *
 * Тест навмисно перевіряє САМ ТЕКСТ із messages/uk.json, а не просто наявність
 * елемента: next-intl-мок повертає КЛЮЧ, якщо перекладу немає, тож помилка в
 * імені ключа (`whyReply_self` ↔ `why_reply_self`) інакше пройшла б мовчки —
 * рівно той клас багів, що з'їв цілий день 08.09.
 */
// Компонент шле подію через динамічний import — без мока її проміс резолвиться
// вже після тесту і React лається act-ворнінгом у чистому прогоні.
jest.mock("@/lib/analytics", () => ({ track: jest.fn() }));

const REASONS = ["price", "self", "look", "abroad", "other"] as const;
const sc = (uk as { scenario: Record<string, string> }).scenario;

const props = {
  product: "map" as const,
  taskId: "task-1",
  summary: "3D-мапа · M · Київ",
  priceUah: 770,
};

beforeEach(() => {
  localStorage.clear();
  jest.useFakeTimers();
});
afterEach(() => {
  jest.useRealTimers();
});

/** Опитування показується лише через 30 с або після кліку «Завантажити». */
function openSurvey() {
  act(() => {
    window.dispatchEvent(new Event("monadruk:guided-download"));
  });
}

describe("SalesAlternatives · відповідь на причину відмови", () => {
  it.each(REASONS)("на причину «%s» відповідає по суті, а не «дякуємо»", async (reason) => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
    render(<SalesAlternatives {...props} />);
    openSurvey();

    await user.click(screen.getByTestId(`why-${reason}`));

    const reply = screen.getByTestId(`why-reply-${reason}`);
    const expected = sc[`whyReply_${reason}`];
    expect(expected).toBeTruthy();
    expect(reply).toHaveTextContent(expected);
    expect(reply).not.toHaveTextContent(sc.whyThanks);
    expect(screen.queryByTestId("why-not-order")).toBeNull();
  });

  it("памʼятає причину між візитами і показує ту саму відповідь", () => {
    localStorage.setItem("mnd_why_task-1", "self");
    render(<SalesAlternatives {...props} />);
    expect(screen.getByTestId("why-reply-self")).toHaveTextContent(sc.whyReply_self);
  });

  it("невідома збережена причина не ламає екран — лишається загальне «дякуємо»", () => {
    localStorage.setItem("mnd_why_task-1", "щось_із_майбутньої_версії");
    render(<SalesAlternatives {...props} />);
    expect(screen.getByTestId("why-thanks")).toHaveTextContent(sc.whyThanks);
  });

  it("усі причини мають переклад — інакше людина побачила б сирий ключ", () => {
    for (const r of REASONS) {
      expect(typeof sc[`whyReply_${r}`]).toBe("string");
      expect(sc[`whyReply_${r}`].length).toBeGreaterThan(20);
      expect(sc[`whyReply_${r}`]).not.toContain("whyReply_");
    }
  });
});
