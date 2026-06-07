import Link from "next/link";
import type { Metadata } from "next";

export const metadata: Metadata = { title: "Умови використання" };

export default function TermsPage() {
  return (
    <div className="mx-auto max-w-[760px] px-5 py-12 lg:px-8">
      <Link href="/" className="text-[13px] font-semibold text-ink-2 hover:text-ink">← monadruk</Link>
      <h1 className="mt-4 font-serif text-[clamp(28px,4vw,42px)] text-ink">Умови використання</h1>
      <p className="mt-2 text-[13px] text-ink-3">Оновлено: 2026</p>

      <div className="mt-8 space-y-6 text-[15px] leading-relaxed text-ink-2">
        <section>
          <h2 className="mb-2 font-serif text-xl text-ink">Про сервіс</h2>
          <p>Monadruk дозволяє створити 3D-модель ділянки міста чи брелка з мапою на основі відкритих даних OpenStreetMap і завантажити готовий файл для 3D-друку (3MF/STL) або замовити друк.</p>
        </section>
        <section>
          <h2 className="mb-2 font-serif text-xl text-ink">Акаунт і безкоштовні завантаження</h2>
          <p>Для завантаження повної моделі потрібен акаунт. Кожному користувачу доступно 5 безкоштовних завантажень. Далі — за домовленістю (друк/оплата), звʼязок через сайт.</p>
        </section>
        <section>
          <h2 className="mb-2 font-serif text-xl text-ink">Дані та авторські права</h2>
          <p>Картографічні дані © OpenStreetMap contributors (ODbL). Згенеровані файли ви можете використовувати для особистого друку. Перепродаж сервісу або масове комерційне використання потребує окремої домовленості.</p>
        </section>
        <section>
          <h2 className="mb-2 font-serif text-xl text-ink">Замовлення та оплата</h2>
          <p>Замовлення оформлюється через сайт; ми звʼязуємось для підтвердження деталей, друку та оплати. Терміни й вартість узгоджуються індивідуально.</p>
        </section>
        <section>
          <h2 className="mb-2 font-serif text-xl text-ink">Відповідальність</h2>
          <p>Сервіс надається «як є». Ми прагнемо максимальної точності моделей, але не гарантуємо повну відповідність реальним обʼєктам через обмеження вихідних даних OSM.</p>
        </section>
        <section>
          <h2 className="mb-2 font-serif text-xl text-ink">Контакт</h2>
          <p>Питання: <a className="text-forest underline-offset-2 hover:underline" href="mailto:saietsmax@gmail.com">saietsmax@gmail.com</a> або кнопка «Звʼязатися».</p>
        </section>
      </div>
    </div>
  );
}
