/**
 * @jest-environment jsdom
 */
import { render, screen } from "@testing-library/react";
import { GenerationStages } from "@/components/GenerationStages";

/**
 * A-14 (08.09.2026): «Завантажити файл» запускає ДРУГУ генерацію (друк-3MF,
 * 1–3 хв). За 7 днів проду — 35 таких очікувань і 0 замовлень. Це єдиний момент,
 * коли увага людини гарантовано з нами, тож саме там показуємо пропозицію друку.
 * Тест тримає контракт: слот рендериться РАЗОМ із printPrep і тільки з ним.
 */
const base = {
  progress: 40,
  kind: "map" as const,
  title: "Готуємо",
  stages: { data: "Дані", terrain: "Рельєф", detail: "Деталі", file: "Файл" },
};

describe("GenerationStages · пропозиція під час підготовки друк-файлу", () => {
  it("показує слот пропозиції, коли йде підготовка друк-файлу", () => {
    render(
      <GenerationStages
        {...base}
        printPrep={35}
        printPrepLabel="Готуємо файл для друку —"
        printPrepOffer={<div data-testid="wait-offer">Замовити друк · 490 ₴</div>}
      />,
    );
    expect(screen.getByTestId("gen-printprep")).toHaveTextContent("35%");
    expect(screen.getByTestId("wait-offer")).toHaveTextContent("490 ₴");
  });

  it("НЕ показує пропозицію, поки друк-файл не готується (звичайне превʼю)", () => {
    render(
      <GenerationStages
        {...base}
        printPrepOffer={<div data-testid="wait-offer">Замовити друк · 490 ₴</div>}
      />,
    );
    expect(screen.queryByTestId("gen-printprep")).toBeNull();
    expect(screen.queryByTestId("wait-offer")).toBeNull();
  });

  it("printPrep=0 — це теж підготовка (0%), а не «немає»", () => {
    render(
      <GenerationStages
        {...base}
        printPrep={0}
        printPrepLabel="Готуємо файл для друку —"
        printPrepOffer={<div data-testid="wait-offer">оферта</div>}
      />,
    );
    expect(screen.getByTestId("wait-offer")).toBeInTheDocument();
  });
});
