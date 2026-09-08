/**
 * @jest-environment jsdom
 */
import { useGenerationStore } from "@/store/generation-store";

/**
 * ⭐Прод-випадок 07.09.2026 (знайдено в analytics 08.09): ОДИН відвідувач натиснув
 * «Завантажити файл» 31 раз за 5 хвилин (паузи 0–13 с) і згенерував 35 подій
 * `download_wait`. Причина: guided-кнопка не блокувалась, а слухач
 * `monadruk:guided-download` викликав `doGatedDownload()` без жодної перевірки —
 * тож КОЖЕН клік запускав нову повну генерацію друк-3MF (1–3 хв) на 2-ядерній VM.
 *
 * Тут тримаємо контракт стору: прапорець існує, за замовчуванням вимкнений і
 * перемикається. Саму re-entrancy захищає ref у SimpleControlPanel (стан
 * оновлюється асинхронно й за чергою кліків не встигає).
 */
describe("downloadBusy — захист від повторних кліків «Завантажити»", () => {
  beforeEach(() => {
    useGenerationStore.getState().setDownloadBusy(false);
  });

  it("за замовчуванням вимкнений", () => {
    expect(useGenerationStore.getState().downloadBusy).toBe(false);
  });

  it("вмикається і вимикається через сетер (кнопка читає його зі стору)", () => {
    useGenerationStore.getState().setDownloadBusy(true);
    expect(useGenerationStore.getState().downloadBusy).toBe(true);
    useGenerationStore.getState().setDownloadBusy(false);
    expect(useGenerationStore.getState().downloadBusy).toBe(false);
  });

  it("повторне вмикання не ламає стан (ідемпотентно)", () => {
    const set = useGenerationStore.getState().setDownloadBusy;
    set(true);
    set(true);
    expect(useGenerationStore.getState().downloadBusy).toBe(true);
    set(false);
    expect(useGenerationStore.getState().downloadBusy).toBe(false);
  });
});
