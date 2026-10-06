/**
 * Умови підписки Monadruk Pro (сторінка /pro-terms). Українська — основна редакція,
 * англійська — для покупців з-за меж України (інші мови показують англійську).
 * Токени й посилання — як у content.ts ({email}, [offer:…], [privacy:…]).
 *
 * ⚠️ Ціни тут = backend SUB_PRICE_UAH / SUB_PRICE_USD (services/subscriptions.py).
 * Змінюєш текст умов → підніми SUB_TERMS_VERSION і TERMS_VERSION у бекенді
 * (у записі згоди покупця фіксується, яку редакцію він прийняв).
 */
import type { LegalDoc } from "./content";

export const SUB_TERMS_VERSION = "2026-10-06";
export const SUB_PRICE = { UAH: 2100, USD: 50 } as const;

const uk: LegalDoc = {
  title: "Умови підписки Monadruk Pro",
  intro: [
    "Продавець — {ownerFull} (далі — «Продавець»). Ці умови регулюють платну підписку Monadruk Pro (далі — «Підписка») і є невідʼємною частиною [offer:Договору публічної оферти]. У частині, не врегульованій цими умовами, діє Договір публічної оферти.",
    "Покупець приймає ці умови (акцепт), коли в кабінеті ставить позначки згоди та оплачує перший місяць Підписки. Редакція умов: {subVersion}.",
  ],
  sections: [
    { h: "1. Що входить у Підписку", blocks: [
      { ul: [
        "Необмежене завантаження файлів для 3D-друку (3MF/STL) усіх моделей, згенерованих в акаунті Покупця, поки Підписка активна: мапи, брелоки, магніти, панно, «Гори».",
        "Генерація моделей без обмежень за кількістю. Технічний захист сервера від перевантаження (тимчасова черга при великій кількості одночасних задач) зберігається.",
        "Розширена ліцензія на моделі, включно з комерційним використанням (розділ 5).",
      ] },
      { p: "Підписка — це доступ до цифрового сервісу. Фізичні вироби (друк і доставка) у Підписку не входять і замовляються окремо за цінами Сайту." },
    ] },
    { h: "2. Ціна", blocks: [
      { ul: [
        "2 100 ₴ на місяць — при оплаті в гривнях;",
        "50 USD на місяць — при оплаті в доларах (для покупців з-за меж України).",
      ] },
      { p: "Валюту Покупець обирає перед оплатою; списання щомісяця відбувається в тій самій валюті й тій самій сумі. Ціна вже включає податки, які Продавець зобовʼязаний сплатити. Банк Покупця може окремо стягнути комісію за конвертацію валюти — вона не залежить від Продавця." },
    ] },
    { h: "3. Оплата та автоматичне продовження", blocks: [
      { p: "Оплата здійснюється через платіжний сервіс LiqPay (АТ КБ «ПриватБанк») регулярним платежем. Перше списання відбувається в момент оформлення Підписки. Далі Підписка продовжується автоматично щомісяця, і LiqPay списує ту саму суму з тієї самої картки в той самий день місяця (якщо в місяці немає такого дня — в останній день місяця), доки Покупець не скасує Підписку." },
      { p: "Згоду на автоматичні щомісячні списання Покупець надає окремою позначкою перед оплатою. Дані картки обробляє LiqPay; Продавець їх не отримує й не зберігає." },
      { p: "Дату наступного списання та історію платежів Покупець бачить у кабінеті. Якщо списання не вдалося (недостатньо коштів тощо), доступ зберігається ще 3 дні; якщо оплата так і не надійде, доступ призупиняється до наступного успішного списання." },
      { p: "Про зміну ціни Продавець повідомляє Покупця на email щонайменше за 30 днів. Нова ціна застосовується лише з першого платежу після цього строку; Покупець може до того скасувати Підписку." },
    ] },
    { h: "4. Скасування Підписки", blocks: [
      { p: "Покупець може скасувати Підписку будь-коли, в один клік, кнопкою «Скасувати підписку» в кабінеті (або написавши на {email}). Після скасування жодних списань більше не буде." },
      { p: "Доступ зберігається до кінця вже оплаченого місяця. Кошти за неповний місяць не повертаються, оскільки доступ до цифрового контенту надано одразу та на весь оплачений строк." },
      { p: "Якщо з вини Продавця сервіс був недоступний понад 3 дні поспіль, Продавець на вибір Покупця продовжує Підписку на відповідний строк або повертає пропорційну частину оплати." },
    ] },
    { h: "5. Ліцензія на моделі", blocks: [
      { p: "На кожну модель, файл якої Покупець завантажив під час активної Підписки, Продавець надає Покупцеві невиключну, безстрокову ліцензію, що діє на території всього світу. Ліцензія зберігається й після завершення Підписки. Покупець має право:" },
      { ul: [
        "друкувати модель у будь-якій кількості примірників;",
        "використовувати модель і надруковані вироби в особистих і комерційних цілях, зокрема продавати надруковані вироби;",
        "використовувати зображення та фото моделі й виробів для реклами та в інтернет-магазинах.",
      ] },
      { p: "Покупцеві заборонено:" },
      { ul: [
        "продавати, поширювати, публікувати чи передавати третім особам самі цифрові файли (3MF, STL, GLB тощо) або їх змінені версії як цифровий товар, зокрема на маркетплейсах 3D-моделей і файлообмінниках;",
        "передавати іншим особам доступ до свого акаунта або продавати його;",
        "автоматизовано масово завантажувати моделі (скрипти, боти) чи використовувати Сайт для створення конкуруючого сервісу.",
      ] },
      { p: "Мапи побудовані на даних OpenStreetMap (© OpenStreetMap contributors, ліцензія ODbL), рельєф — на відкритих даних висот. Продаючи вироби, Покупець вказує «Map data © OpenStreetMap contributors» в описі товару, на упаковці або на вкладиші." },
      { p: "Окремі обʼєкти на мапі (будівлі, памʼятки, торговельні марки тощо) у деяких країнах можуть бути захищені правами третіх осіб. За дотримання таких прав під час комерційного використання відповідає Покупець." },
    ] },
    { h: "6. Право на відмову від договору", blocks: [
      { p: "Підписка надає доступ до цифрового контенту одразу після оплати. Перед оплатою Покупець окремою позначкою просить почати надання доступу негайно й підтверджує, що з початком надання втрачає право відмовитися від договору протягом 14 днів. Це передбачено законодавством про захист прав споживачів, а для споживачів з ЄС — ст. 16(m) Директиви 2011/83/ЄС. Скасувати Підписку на майбутнє можна будь-коли (розділ 4)." },
      { p: "Якщо Покупець оплатив Підписку, але не зміг скористатися нею з технічних причин на боці Продавця, кошти повертаються повністю." },
    ] },
    { h: "7. Добросовісне використання", blocks: [
      { p: "Безлім розрахований на звичайну роботу людини або компанії. Якщо акаунт використовується з порушенням розділу 5 або створює надмірне навантаження (тисячі генерацій скриптом), Продавець може призупинити Підписку, попередньо повідомивши Покупця на email, і повернути кошти за невикористаний строк." },
    ] },
    { h: "8. Персональні дані", blocks: [
      { p: "Для Підписки обробляються: email і ідентифікатор акаунта, статус і строк Підписки, історія списань (без даних картки), а також запис згоди: час, редакція умов, мова, код країни, хеш IP-адреси. Детально — у [privacy:Політиці конфіденційності]." },
    ] },
    { h: "9. Зміна умов", blocks: [
      { p: "Продавець може змінювати ці умови, публікуючи нову редакцію на цій сторінці й повідомляючи Покупців з активною Підпискою на email щонайменше за 30 днів. Нова редакція діє з наступного платіжного періоду після цього строку; якщо Покупець не згоден, він може скасувати Підписку." },
    ] },
    { h: "10. Застосовне право та звернення", blocks: [
      { p: "До цих умов застосовується право України. Для споживачів з інших країн зберігаються обовʼязкові гарантії захисту прав споживачів їхньої країни проживання. Звернення: {email}, {phone}; відповідаємо протягом 2–4 робочих днів." },
    ] },
  ],
};

const en: LegalDoc = {
  title: "Monadruk Pro Subscription Terms",
  intro: [
    "These terms are an integral part of the [offer:Public Offer Agreement] of {ownerFull} (the \"Seller\") and govern the paid Monadruk Pro subscription (the \"Subscription\"). Anything not covered here is governed by the Public Offer Agreement. If translations differ, the Ukrainian version prevails, except where the mandatory consumer law of your country gives you more protection.",
    "You accept these terms when you tick the consent boxes in your account and pay for the first month of the Subscription. Terms version: {subVersion}.",
  ],
  sections: [
    { h: "1. What the Subscription includes", blocks: [
      { ul: [
        "Unlimited downloads of 3D-printing files (3MF/STL) for every model generated in your account while the Subscription is active: maps, keychains, magnets, panels, \"Mountains\".",
        "Unlimited model generation. Technical protection of the server against overload (a temporary queue when many jobs run at once) still applies.",
        "An extended licence for the models, including commercial use (section 5).",
      ] },
      { p: "The Subscription is access to a digital service. Physical products (printing and delivery) are not included and are ordered separately at the prices shown on the Website." },
    ] },
    { h: "2. Price", blocks: [
      { ul: [
        "UAH 2,100 per month when paying in Ukrainian hryvnia;",
        "USD 50 per month when paying in US dollars (for customers outside Ukraine).",
      ] },
      { p: "You choose the currency before paying; every monthly charge is made in the same currency and amount. The price includes the taxes the Seller is required to pay. Your bank may charge its own currency-conversion fee, which the Seller does not control." },
    ] },
    { h: "3. Payment and automatic renewal", blocks: [
      { p: "Payment is processed by LiqPay (JSC CB PrivatBank) as a recurring payment. The first charge is made when you subscribe. The Subscription then renews automatically every month: LiqPay charges the same amount to the same card on the same day of the month (or on the last day of the month if that day does not exist) until you cancel." },
      { p: "You consent to automatic monthly charges by ticking a separate box before paying. Card details are processed by LiqPay; the Seller never receives or stores them." },
      { p: "Your next charge date and payment history are shown in your account. If a charge fails (e.g. insufficient funds), access continues for 3 more days; if payment is still not received, access is paused until the next successful charge." },
      { p: "The Seller will notify you of any price change by email at least 30 days in advance. The new price applies only from the first payment after that period, and you may cancel before then." },
    ] },
    { h: "4. Cancellation", blocks: [
      { p: "You can cancel at any time with one click, using the \"Cancel subscription\" button in your account (or by writing to {email}). No further charges will be made after cancellation." },
      { p: "Access continues until the end of the month you have already paid for. Partial months are not refunded, because access to the digital content is provided immediately and for the whole paid period." },
      { p: "If the service is unavailable through the Seller's fault for more than 3 consecutive days, the Seller will, at your choice, extend the Subscription by the same period or refund the proportional part of the payment." },
    ] },
    { h: "5. Licence for the models", blocks: [
      { p: "For every model whose file you download while the Subscription is active, the Seller grants you a non-exclusive, perpetual, worldwide licence. The licence remains in force after the Subscription ends. You may:" },
      { ul: [
        "print the model in any number of copies;",
        "use the model and the printed products for personal and commercial purposes, including selling the printed products;",
        "use images and photos of the model and products in advertising and online shops.",
      ] },
      { p: "You may not:" },
      { ul: [
        "sell, distribute, publish or give third parties the digital files themselves (3MF, STL, GLB, etc.) or modified versions of them as a digital product, including on 3D-model marketplaces and file-sharing sites;",
        "share access to your account with others or sell it;",
        "download models in bulk by automated means (scripts, bots) or use the Website to build a competing service.",
      ] },
      { p: "Maps are built from OpenStreetMap data (© OpenStreetMap contributors, ODbL licence) and terrain from open elevation data. When you sell products, include \"Map data © OpenStreetMap contributors\" in the product description, on the packaging or on an insert." },
      { p: "In some countries, individual objects shown on a map (buildings, monuments, trademarks, etc.) may be protected by third-party rights. You are responsible for respecting such rights when using the models commercially." },
    ] },
    { h: "6. Right of withdrawal", blocks: [
      { p: "The Subscription gives access to digital content immediately after payment. Before paying, you tick a separate box asking us to start providing access immediately and acknowledging that you lose your 14-day right of withdrawal once it starts. This is permitted by consumer protection law (for EU consumers, Art. 16(m) of Directive 2011/83/EU). You can always cancel future renewals at any time (section 4)." },
      { p: "If you paid for the Subscription but could not use it because of a technical problem on the Seller's side, you will receive a full refund." },
    ] },
    { h: "7. Fair use", blocks: [
      { p: "\"Unlimited\" is designed for the normal work of a person or a company. If an account is used in breach of section 5 or creates excessive load (thousands of scripted generations), the Seller may suspend the Subscription after notifying you by email and will refund the unused period." },
    ] },
    { h: "8. Personal data", blocks: [
      { p: "For the Subscription we process: your email and account ID, Subscription status and period, charge history (no card data) and a consent record: time, terms version, language, country code and hashed IP address. Details are in the [privacy:Privacy Policy]." },
    ] },
    { h: "9. Changes to these terms", blocks: [
      { p: "The Seller may change these terms by publishing a new version on this page and notifying subscribers by email at least 30 days in advance. The new version applies from the next billing period after that notice; if you do not agree, you may cancel." },
    ] },
    { h: "10. Governing law and contact", blocks: [
      { p: "These terms are governed by the law of Ukraine. Consumers from other countries keep the mandatory consumer protection rights of their country of residence. Contact: {email}, {phone}; we reply within 2–4 business days." },
    ] },
  ],
};

export function getSubscriptionTerms(locale: string): LegalDoc {
  return locale === "uk" ? uk : en;
}
