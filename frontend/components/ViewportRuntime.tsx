"use client";

import { useEffect, useRef, useState, type RefObject } from "react";

/**
 * T-6.4 — СТАБІЛЬНА ВИСОТА ЕКРАНА для конструкторів.
 *
 * ПРОБЛЕМА: `100dvh` в iOS Safari змінюється щоразу, коли ховається/показується
 * адресний рядок → карта і панелі «стрибають» під пальцем прямо під час вибору
 * зони (найгірше саме на кроці, де людина щось тягне).
 *
 * РІШЕННЯ: один раз на маунті фіксуємо `--app-vh` = window.innerHeight у ПІКСЕЛЯХ
 * і більше НЕ чіпаємо на resize (саме resize від URL-бару і є багом). Оновлюємо
 * лише на зміну орієнтації — там висота дійсно інша.
 *
 * Фолбек (немає JS / до маунта) лишається у globals.css: `--app-vh: 100dvh`
 * (а де dvh не підтримується — `100vh`). На десктопі значення дорівнює висоті
 * вікна, тому вигляд не змінюється.
 */
export function AppViewportHeight() {
  useEffect(() => {
    // 16.09.2026: вкладка, що змонтувалась НЕ на екрані (фонова/прихована
    // панель, prerender), має innerHeight 0 → `--app-vh: 0px` → карта
    // конструктора висотою 0 px і порожній крок 2. Нереалістичне значення не
    // записуємо (лишається CSS-фолбек 100dvh) і чекаємо resize, коли вікно
    // отримає справжній розмір.
    const MIN_SANE_PX = 200;
    let applied = false;
    const apply = () => {
      const h = window.innerHeight;
      if (!(h >= MIN_SANE_PX)) return false;
      document.documentElement.style.setProperty("--app-vh", `${h}px`);
      applied = true;
      return true;
    };
    apply();
    // iOS повідомляє orientationchange ДО того, як innerHeight оновиться —
    // тому міряємо ще раз із затримкою.
    const onOrientation = () => {
      apply();
      window.setTimeout(apply, 300);
    };
    // На пристроях без тачу resize — це справжня зміна вікна (не URL-бар):
    // стежимо за ним завжди; на тачі — лише поки не отримали перше чесне значення.
    const finePointer = typeof window.matchMedia === "function" && window.matchMedia("(pointer: fine)").matches && !("ontouchstart" in window);
    let timer: number | undefined;
    const onResize = () => {
      if (applied && !finePointer) return;
      window.clearTimeout(timer);
      timer = window.setTimeout(apply, 120);
    };
    window.addEventListener("orientationchange", onOrientation);
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("orientationchange", onOrientation);
      window.removeEventListener("resize", onResize);
      window.clearTimeout(timer);
    };
  }, []);
  return null;
}

/**
 * T-6.5 — ГЕЙТ МОНТУВАННЯ ВАЖКОГО КОМПОНЕНТА ЗА ВИДИМІСТЮ.
 *
 * `next/dynamic(ssr:false)` просить чанк одразу, щойно компонент відрендерився —
 * навіть якщо той за екраном. Повертає true, коли елемент наблизився до вікна,
 * і НІКОЛИ не повертається у false (змонтоване лишається змонтованим).
 * Без IntersectionObserver (старі движки, jsdom у тестах) — одразу true.
 *
 * НЕ ЗАСТОСОВУВАТИ до MapSelector у конструкторах: карта — ще й СЛУХАЧ глобальної
 * події `monadruk:map-goto`, яку СИНХРОННО шлють чіпи міст і пошук
 * (KeychainScenarioFlow.tsx:388, ScenarioFlow.tsx:262, SimpleControlPanel.tsx:658)
 * та deep-links `?city=/?template=` (ScenarioFlow.tsx:258). Поки карта за екраном,
 * подія летить у порожнечу → «Париж» не переносить рамку. Ре-диспатч навздогін теж
 * не рятує: KeychainScenarioFlow.tsx:188 на кожному map-goto чистить напис, тож
 * повтор затер би «PARIS», який чіп поставив синхронно (на це є e2e).
 */
export function useNearViewport(ref: RefObject<Element | null>, rootMargin = "200px"): boolean {
  const [near, setNear] = useState(false);
  const done = useRef(false);
  useEffect(() => {
    if (done.current) return;
    const el = ref.current;
    if (!el || typeof IntersectionObserver === "undefined") {
      done.current = true;
      setNear(true);
      return;
    }
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          done.current = true;
          setNear(true);
          io.disconnect();
        }
      },
      { rootMargin },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [ref, rootMargin]);
  return near;
}
