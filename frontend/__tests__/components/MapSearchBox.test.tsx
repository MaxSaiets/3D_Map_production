/**
 * Пошук місця: порожня відповідь ≠ мовчання. 11.09.2026 віденець вісім разів
 * клацав по полю пошуку — нічого не відбувалось. Тепер: «нічого не знайдено»,
 * «пошук недоступний», а Enter обирає перший результат.
 */
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { NextIntlClientProvider } from "next-intl";
import { MapSearchBox } from "@/components/MapSearchBox";

jest.mock("@/lib/geocode", () => {
  class GeocodeUnavailableError extends Error { code = "geocode_unavailable"; }
  return {
    GeocodeUnavailableError,
    isGeocodeUnavailable: (e: unknown) => !!e && (e as { code?: string }).code === "geocode_unavailable",
    geocodeSearch: jest.fn(),
    reverseGeocode: jest.fn(),
  };
});

const geo = jest.requireMock("@/lib/geocode") as {
  geocodeSearch: jest.Mock;
  GeocodeUnavailableError: new (m?: string) => Error;
};

const messages = {
  search: {
    placeholder: "Місто, адреса",
    myPlace: "Я тут",
    myLocation: "Моє місцезнаходження",
    clear: "Очистити",
    noResults: "Нічого не знайдено.",
    unavailable: "Пошук недоступний.",
  },
};

function mount() {
  return render(
    <NextIntlClientProvider locale="uk" messages={messages}>
      <MapSearchBox variant="panel" />
    </NextIntlClientProvider>,
  );
}

async function type(value: string) {
  const input = screen.getByRole("combobox");
  fireEvent.change(input, { target: { value } });
  // дебаунс 450 мс
  await act(async () => { await new Promise((r) => setTimeout(r, 500)); });
  return input;
}

describe("MapSearchBox", () => {
  beforeEach(() => geo.geocodeSearch.mockReset());

  it("каже «нічого не знайдено», коли геокодер відповів порожньо", async () => {
    geo.geocodeSearch.mockResolvedValue([]);
    mount();
    await type("Andergasse 83 Wien");
    await waitFor(() => expect(screen.getByTestId("map-search-note")).toHaveTextContent(/Нічого не знайдено/));
  });

  it("каже «пошук недоступний», коли геокодер упав", async () => {
    geo.geocodeSearch.mockRejectedValue(new geo.GeocodeUnavailableError("429"));
    mount();
    await type("Wien");
    await waitFor(() => expect(screen.getByTestId("map-search-note")).toHaveTextContent(/недоступн/));
  });

  it("Enter обирає перший результат і шле map-goto", async () => {
    geo.geocodeSearch.mockResolvedValue([
      { lat: 48.2, lon: 16.37, label: "Wien", full: "Wien, Österreich" },
      { lat: 1, lon: 1, label: "Інше", full: "Інше" },
    ]);
    const seen: Array<{ lat: number; lon: number; label: string }> = [];
    window.addEventListener("monadruk:map-goto", (e) => seen.push((e as CustomEvent).detail));
    mount();
    const input = await type("Wien");
    await waitFor(() => expect(screen.getByRole("listbox")).toBeInTheDocument());
    fireEvent.keyDown(input, { key: "Enter" });
    expect(seen).toEqual([{ lat: 48.2, lon: 16.37, label: "Wien" }]);
    expect(screen.queryByRole("listbox")).toBeNull();
    expect(screen.queryByTestId("map-search-note")).toBeNull();
  });
});
