import Link from "next/link";
import type { Metadata } from "next";
import { pageMetadata } from "@/i18n/metadata";

export async function generateMetadata({ params }: { params: { locale: string } }): Promise<Metadata> {
  return pageMetadata({ locale: params.locale, path: "/privacy", ns: "privacyMeta" });
}

export default function PrivacyPage() {
  return (
    <div className="mx-auto max-w-[760px] px-5 py-12 lg:px-8">
      <Link href="/" className="text-[13px] font-semibold text-ink-2 hover:text-ink">← monadruk</Link>
      <h1 className="mt-4 font-serif text-[clamp(28px,4vw,42px)] text-ink">Політика конфіденційності</h1>
      <p className="mt-2 text-[13px] text-ink-3">Оновлено: 13 червня 2026</p>

      <div className="mt-8 space-y-6 text-[15px] leading-relaxed text-ink-2">
        <section>
          <h2 className="mb-2 font-serif text-xl text-ink">Які дані ми збираємо</h2>
          <p>Ім'я, email або номер телефону (при вході/реєстрації), а також контактні дані та адресу доставки, які ви вказуєте при оформленні замовлення. Технічні дані (історія згенерованих моделей) зберігаються у вашому кабінеті.</p>
        </section>
        <section>
          <h2 className="mb-2 font-serif text-xl text-ink">Як ми використовуємо дані</h2>
          <p>Виключно для надання послуги: вхід в акаунт, збереження історії моделей, обробка та доставка замовлень, звʼязок із вами щодо замовлення. Ми не продаємо й не передаємо ваші дані третім особам для реклами.</p>
        </section>
        <section>
          <h2 className="mb-2 font-serif text-xl text-ink">Завантажені маршрути (GPX) та геодані</h2>
          <p>
            Якщо ви завантажуєте GPX-файл (наприклад, експорт власної активності зі Strava чи іншого додатка),
            ми обробляємо координати маршруту <b>виключно для побудови вашої 3D-моделі</b>. Це ваші власні дані —
            ми їх не публікуємо, не передаємо третім особам і не використовуємо для реклами. Точки маршруту
            проріджуються й зберігаються рівно стільки, скільки потрібно для генерації та (якщо ви залогінені)
            ведення історії моделей у кабінеті; ви можете попросити їх видалити будь-коли.
          </p>
          <p className="mt-2">
            Пошук місця на карті надсилає ваш запит до сервісу геокодування OpenStreetMap (Nominatim), а самі
            карти підвантажуються з тайлів OpenStreetMap — згідно з їхніми умовами використання. Ми не передаємо
            цим сервісам ваше імʼя чи контактні дані.
          </p>
        </section>
        <section>
          <h2 className="mb-2 font-serif text-xl text-ink">Зберігання та сервіси</h2>
          <p>Авторизація працює через Google Firebase Authentication. Сайт під захистом Cloudflare. Замовлення обробляються вручну. Дані зберігаються на захищеному сервері рівно стільки, скільки потрібно для виконання замовлення та ведення історії.</p>
        </section>
        <section>
          <h2 className="mb-2 font-serif text-xl text-ink">Cookie та аналітика</h2>
          <p>Ми використовуємо приватну аналітику Cloudflare без сторонніх рекламних кукі. Cookie застосовуються лише для роботи входу в акаунт.</p>
        </section>
        <section>
          <h2 className="mb-2 font-serif text-xl text-ink">Ваші права</h2>
          <p>Ви можете попросити видалити свій акаунт і повʼязані дані. Напишіть нам через кнопку «Звʼязатися» на сайті.</p>
        </section>
        <section>
          <h2 className="mb-2 font-serif text-xl text-ink">Контакт</h2>
          <p>З питань конфіденційності: <a className="text-forest underline-offset-2 hover:underline" href="mailto:saietsmax@gmail.com">saietsmax@gmail.com</a>.</p>
        </section>
      </div>
    </div>
  );
}
