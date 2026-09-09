/**
 * @jest-environment jsdom
 */
import { clickLabel } from "@/lib/analytics";

/**
 * ⭐09.09.2026, на реальному прод-логу: з ~2400 кліків на /create **73 %**
 * приходили без жодної інформації — `div` 971, `path` 527, `canvas` 215,
 * `select` 28, `a` 23. Тобто найчастіші дії користувача були для нас невидимі,
 * а вирішувати за цими даними доводиться власнику.
 *
 * Три причини, кожна закріплена тестом нижче:
 *   1) перша перевірка шукала `data-track`, якого в проєкті НЕМАЄ ЖОДНОГО;
 *   2) підйом лише на 4 рівні — іконка в кнопці часто глибша;
 *   3) текст брався лише з `button`/`a`, тож картки на `label`/`role="radio"`
 *      (а це весь guided-вибір) лишались безіменними.
 */
function build(html: string): HTMLElement {
  document.body.innerHTML = html;
  return document.body.firstElementChild as HTMLElement;
}

const deepest = (root: HTMLElement): HTMLElement => {
  let n: HTMLElement = root;
  while (n.firstElementChild) n = n.firstElementChild as HTMLElement;
  return n;
};

describe("clickLabel · клік має мати ім'я", () => {
  it("бере data-testid — саме він є в розмітці (109 штук), а data-track немає ніде", () => {
    const root = build('<div data-testid="preview-3d"><canvas></canvas></div>');
    expect(clickLabel(deepest(root))).toBe("preview-3d");
  });

  it("data-track усе ще має пріоритет, якщо колись з'явиться", () => {
    const root = build('<div data-track="hero_cta" data-testid="ignored"><span>x</span></div>');
    expect(clickLabel(deepest(root))).toBe("hero_cta");
  });

  it("піднімається глибше за 4 рівні: іконка всередині кнопки", () => {
    // Саме така вкладеність давала `path` — 527 кліків без імені.
    const root = build(
      '<button aria-label="Згенерувати превʼю">' +
      '<span><span><span><span><svg><g><path d="M0 0"></path></g></svg></span></span></span></span>' +
      "</button>",
    );
    const path = root.querySelector("path") as unknown as HTMLElement;
    expect(clickLabel(path)).toBe("Згенерувати превʼю");
  });

  it("картка вибору на label отримує свій текст", () => {
    const root = build('<label><input type="radio" /> <span>Шестикутник</span></label>');
    expect(clickLabel(root.querySelector("span") as HTMLElement)).toBe("Шестикутник");
  });

  it("картка на role=radio теж має ім'я, а не «div»", () => {
    const root = build('<div role="radio"><span><b>Рельєф місцевості</b></span></div>');
    expect(clickLabel(deepest(root))).toBe("Рельєф місцевості");
  });

  it("кнопка без тексту падає на title, а не на голий тег", () => {
    const root = build('<button title="Повернути на 15°"><svg></svg></button>');
    expect(clickLabel(root.querySelector("svg") as unknown as HTMLElement)).toBe("Повернути на 15°");
  });

  it("довгий підпис обрізається, пробіли схлопуються", () => {
    const root = build('<button>   Дуже\n  довгий    підпис кнопки, який точно не влізе у сорок символів  </button>');
    const label = clickLabel(root);
    expect(label.length).toBeLessThanOrEqual(40);
    expect(label.startsWith("Дуже довгий підпис")).toBe(true);
    expect(label).not.toContain("\n");
  });

  it("коли впізнати нічим — чесно віддає тег, а не вигадує", () => {
    const root = build("<div><span><i></i></span></div>");
    expect(clickLabel(deepest(root))).toBe("i");
  });

  it("не падає на null і на не-елементі", () => {
    expect(clickLabel(null)).toBe("?");
    expect(clickLabel(document as unknown as EventTarget)).toBe("?");
  });

  it("aria-label найближчого предка перемагає текст далекої кнопки", () => {
    const root = build('<button>Зовнішня</button>');
    document.body.innerHTML = '<button>Зовнішня<span aria-label="Закрити"><i></i></span></button>';
    const i = document.querySelector("i") as HTMLElement;
    expect(clickLabel(i)).toBe("Закрити");
    expect(root).toBeTruthy();
  });
});
