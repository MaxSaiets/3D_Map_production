import Link from "next/link";
import type { Metadata } from "next";

export const metadata: Metadata = { title: "Політика конфіденційності" };

export default function PrivacyPage() {
  return (
    <div className="mx-auto max-w-[760px] px-5 py-12 lg:px-8">
      <Link href="/" className="text-[13px] font-semibold text-ink-2 hover:text-ink">← monadruk</Link>
      <h1 className="mt-4 font-serif text-[clamp(28px,4vw,42px)] text-ink">Політика конфіденційності</h1>
      <p className="mt-2 text-[13px] text-ink-3">Оновлено: 2026</p>

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
