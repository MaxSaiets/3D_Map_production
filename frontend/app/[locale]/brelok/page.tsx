import type { Metadata } from "next";
import { setRequestLocale } from "next-intl/server";
import { BASE, localeUrl } from "@/i18n/metadata";
import { routing, locales, localeMeta, defaultLocale, type AppLocale } from "@/i18n/routing";
import { Link } from "@/i18n/navigation";
import { CITY_PAGES } from "@/lib/cityPages";

/**
 * Індекс-хаб /brelok — раніше «голий» /brelok давав 404 (лише /brelok/[city] існував),
 * тож брелок-сторінки міст не мали власної точки входу в сайтмапі/навігації. Ця
 * сторінка: H1 + проза + грід усіх міст (link-equity на /brelok/[city]) + FAQ + LD.
 * Повністю локалізована в 6 мовах (це грошовий keychain-шлях, органіка UA→Google→брелки).
 */
type HubCopy = {
  title: string; description: string; h1: string; intro: string;
  h2cities: string; h2how: string; pHow: string; h2faq: string;
  faq: { q: string; a: string }[]; ctaKeychain: string; ctaMap: string;
};

const HUB: Record<AppLocale, HubCopy> = {
  uk: {
    title: "Брелки з картою міста — 3D-друк на замовлення",
    description: "3D-брелок із картою будь-якого міста України чи світу: вулиці, будівлі, річки у мініатюрі на жетоні. Ваш район на ключах — персональний подарунок від 120 ₴.",
    h1: "Брелок з картою вашого міста",
    intro: "Мініатюра рідного району на ключах — вулиці, квартали й річки у точному 3D. Оберіть місто нижче або будь-яку точку світу в конструкторі. Друкуємо з еко-PLA й доставляємо Україною.",
    h2cities: "Оберіть місто",
    h2how: "Як це працює",
    pHow: "Обираєте район, стиль і напис — ми генеруємо точну 3D-модель з даних OpenStreetMap, друкуємо на жетоні 55×30 мм і надсилаємо. Класична або квадратна форма, з вашим написом.",
    h2faq: "Часті запитання",
    faq: [
      { q: "Чи можна будь-яке місто?", a: "Так. У списку — найпопулярніші міста України, але в конструкторі доступна будь-яка точка світу, не лише зі списку." },
      { q: "Скільки коштує брелок?", a: "Базова ціна брелка-жетона — від 120 ₴. Фінальна залежить від формату й напису; точну суму видно перед замовленням." },
      { q: "Що можна написати на брелку?", a: "Ваш напис — назву району, дату, координати чи ім'я. Текст додається спереду або ззаду жетона." },
    ],
    ctaKeychain: "Створити свій брелок",
    ctaMap: "3D-мапа на стіну",
  },
  en: {
    title: "City-map keychains — custom 3D print",
    description: "A 3D keychain with the map of any city — streets, buildings and rivers in miniature on a tag. Your neighbourhood on your keys, a personal gift from €3.",
    h1: "A keychain with your city's map",
    intro: "A miniature of your neighbourhood on your keys — streets, blocks and rivers in precise 3D. Pick a city below, or any point on Earth in the builder. Printed in eco-PLA and shipped across Ukraine.",
    h2cities: "Choose a city",
    h2how: "How it works",
    pHow: "Pick an area, style and text — we generate a precise 3D model from OpenStreetMap data, print it on a 55×30 mm tag and ship it. Classic or square shape, with your inscription.",
    h2faq: "FAQ",
    faq: [
      { q: "Can I get any city?", a: "Yes. The list shows the most popular Ukrainian cities, but the builder lets you pick any point on Earth, not just from the list." },
      { q: "How much is a keychain?", a: "A keychain tag starts from €3. The final price depends on format and inscription; the exact amount is shown before you order." },
      { q: "What can I write on it?", a: "Your text — a district name, a date, coordinates or a name. It's added to the front or back of the tag." },
    ],
    ctaKeychain: "Create your keychain",
    ctaMap: "3D wall map",
  },
  de: {
    title: "Schlüsselanhänger mit Stadtkarte — 3D-Druck",
    description: "Ein 3D-Schlüsselanhänger mit der Karte jeder Stadt — Straßen, Gebäude und Flüsse als Miniatur auf einem Anhänger. Dein Viertel am Schlüssel, ein persönliches Geschenk ab 3 €.",
    h1: "Schlüsselanhänger mit der Karte deiner Stadt",
    intro: "Eine Miniatur deines Viertels am Schlüsselbund — Straßen, Blocks und Flüsse in präzisem 3D. Wähle unten eine Stadt oder im Konfigurator jeden beliebigen Punkt der Erde. Gedruckt aus Öko-PLA, Versand in die Ukraine.",
    h2cities: "Stadt wählen",
    h2how: "So funktioniert es",
    pHow: "Bereich, Stil und Text wählen — wir erzeugen ein präzises 3D-Modell aus OpenStreetMap-Daten, drucken es auf einen 55×30-mm-Anhänger und versenden es. Klassische oder quadratische Form, mit deiner Gravur.",
    h2faq: "Häufige Fragen",
    faq: [
      { q: "Ist jede Stadt möglich?", a: "Ja. Die Liste zeigt die beliebtesten ukrainischen Städte, aber im Konfigurator lässt sich jeder Punkt der Erde wählen, nicht nur aus der Liste." },
      { q: "Was kostet ein Anhänger?", a: "Ein Anhänger startet ab 3 €. Der Endpreis hängt von Format und Text ab; der genaue Betrag wird vor der Bestellung angezeigt." },
      { q: "Was kann man daraufschreiben?", a: "Deinen Text — einen Stadtteilnamen, ein Datum, Koordinaten oder einen Namen. Er kommt auf die Vorder- oder Rückseite." },
    ],
    ctaKeychain: "Anhänger erstellen",
    ctaMap: "3D-Wandkarte",
  },
  pl: {
    title: "Breloki z mapą miasta — druk 3D na zamówienie",
    description: "Brelok 3D z mapą dowolnego miasta — ulice, budynki i rzeki w miniaturze na zawieszce. Twoja dzielnica przy kluczach, osobisty prezent od 3 €.",
    h1: "Brelok z mapą Twojego miasta",
    intro: "Miniatura Twojej dzielnicy przy kluczach — ulice, kwartały i rzeki w precyzyjnym 3D. Wybierz miasto poniżej lub dowolny punkt świata w kreatorze. Drukujemy z eko-PLA i wysyłamy na Ukrainę.",
    h2cities: "Wybierz miasto",
    h2how: "Jak to działa",
    pHow: "Wybierasz obszar, styl i napis — generujemy precyzyjny model 3D z danych OpenStreetMap, drukujemy na zawieszce 55×30 mm i wysyłamy. Kształt klasyczny lub kwadratowy, z Twoim napisem.",
    h2faq: "Częste pytania",
    faq: [
      { q: "Czy może być dowolne miasto?", a: "Tak. Lista pokazuje najpopularniejsze miasta Ukrainy, ale w kreatorze można wybrać dowolny punkt na świecie, nie tylko z listy." },
      { q: "Ile kosztuje brelok?", a: "Brelok-zawieszka od 3 €. Cena końcowa zależy od formatu i napisu; dokładna kwota jest widoczna przed zamówieniem." },
      { q: "Co można napisać na breloku?", a: "Twój napis — nazwę dzielnicy, datę, współrzędne lub imię. Dodajemy go z przodu lub z tyłu zawieszki." },
    ],
    ctaKeychain: "Stwórz swój brelok",
    ctaMap: "Mapa 3D na ścianę",
  },
  fr: {
    title: "Porte-clés avec carte de ville — impression 3D",
    description: "Un porte-clés 3D avec la carte de n'importe quelle ville — rues, bâtiments et rivières en miniature sur une plaque. Votre quartier sur vos clés, un cadeau personnel dès 3 €.",
    h1: "Un porte-clés avec la carte de votre ville",
    intro: "Une miniature de votre quartier sur vos clés — rues, îlots et rivières en 3D précis. Choisissez une ville ci-dessous, ou n'importe quel point du monde dans le configurateur. Imprimé en PLA écologique, livré en Ukraine.",
    h2cities: "Choisir une ville",
    h2how: "Comment ça marche",
    pHow: "Choisissez une zone, un style et un texte — nous générons un modèle 3D précis à partir des données OpenStreetMap, l'imprimons sur une plaque de 55×30 mm et l'expédions. Forme classique ou carrée, avec votre gravure.",
    h2faq: "Questions fréquentes",
    faq: [
      { q: "Peut-on avoir n'importe quelle ville ?", a: "Oui. La liste montre les villes ukrainiennes les plus populaires, mais le configurateur permet de choisir n'importe quel point du monde, pas seulement dans la liste." },
      { q: "Combien coûte un porte-clés ?", a: "Une plaque porte-clés à partir de 3 €. Le prix final dépend du format et du texte ; le montant exact s'affiche avant la commande." },
      { q: "Que peut-on y inscrire ?", a: "Votre texte — un nom de quartier, une date, des coordonnées ou un prénom. Il est ajouté à l'avant ou à l'arrière de la plaque." },
    ],
    ctaKeychain: "Créer votre porte-clés",
    ctaMap: "Carte 3D murale",
  },
  es: {
    title: "Llaveros con mapa de ciudad — impresión 3D",
    description: "Un llavero 3D con el mapa de cualquier ciudad — calles, edificios y ríos en miniatura sobre una placa. Tu barrio en las llaves, un regalo personal desde 3 €.",
    h1: "Un llavero con el mapa de tu ciudad",
    intro: "Una miniatura de tu barrio en las llaves — calles, manzanas y ríos en 3D preciso. Elige una ciudad abajo, o cualquier punto del mundo en el configurador. Impreso en PLA ecológico y enviado a Ucrania.",
    h2cities: "Elige una ciudad",
    h2how: "Cómo funciona",
    pHow: "Eliges una zona, un estilo y un texto — generamos un modelo 3D preciso a partir de datos de OpenStreetMap, lo imprimimos en una placa de 55×30 mm y lo enviamos. Forma clásica o cuadrada, con tu inscripción.",
    h2faq: "Preguntas frecuentes",
    faq: [
      { q: "¿Puede ser cualquier ciudad?", a: "Sí. La lista muestra las ciudades ucranianas más populares, pero el configurador permite elegir cualquier punto del mundo, no solo de la lista." },
      { q: "¿Cuánto cuesta un llavero?", a: "Una placa llavero desde 3 €. El precio final depende del formato y el texto; el importe exacto se muestra antes de pedir." },
      { q: "¿Qué se puede escribir?", a: "Tu texto — el nombre de un barrio, una fecha, coordenadas o un nombre. Se añade en el frente o el reverso de la placa." },
    ],
    ctaKeychain: "Crea tu llavero",
    ctaMap: "Mapa 3D de pared",
  },
};

export async function generateMetadata({ params }: { params: { locale: string } }): Promise<Metadata> {
  const locale = ((routing.locales as readonly string[]).includes(params.locale)
    ? params.locale
    : defaultLocale) as AppLocale;
  const c = HUB[locale];
  const path = "/brelok";
  const languages: Record<string, string> = {};
  for (const l of locales) languages[localeMeta[l].htmlLang] = localeUrl(l, path);
  languages["x-default"] = localeUrl(defaultLocale, path);
  return {
    title: c.title,
    description: c.description,
    alternates: { canonical: localeUrl(locale, path), languages },
    openGraph: {
      title: c.title, description: c.description, url: localeUrl(locale, path),
      siteName: "Monadruk", type: "website", locale: localeMeta[locale].ogLocale,
      images: [`${BASE}/opengraph-image`],
    },
    twitter: { card: "summary_large_image", title: c.title, description: c.description, images: [`${BASE}/opengraph-image`] },
  };
}

export default async function BrelokIndexPage({ params }: { params: { locale: string } }) {
  const locale = ((routing.locales as readonly string[]).includes(params.locale)
    ? params.locale
    : defaultLocale) as AppLocale;
  setRequestLocale(locale);
  const c = HUB[locale];

  const ld = {
    "@context": "https://schema.org",
    "@graph": [
      { "@type": "CollectionPage", name: c.title, description: c.description, url: localeUrl(locale, "/brelok") },
      {
        "@type": "BreadcrumbList",
        itemListElement: [
          { "@type": "ListItem", position: 1, name: "Monadruk", item: localeUrl(locale, "/") },
          { "@type": "ListItem", position: 2, name: c.h1, item: localeUrl(locale, "/brelok") },
        ],
      },
      {
        "@type": "FAQPage",
        mainEntity: c.faq.map((f) => ({ "@type": "Question", name: f.q, acceptedAnswer: { "@type": "Answer", text: f.a } })),
      },
    ],
  };

  return (
    <main id="main-content" tabIndex={-1} className="mx-auto max-w-[920px] px-5 py-14 lg:py-20">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(ld) }} />
      <h1 className="text-[clamp(28px,4vw,46px)] leading-tight">{c.h1}</h1>
      <p className="mt-4 max-w-[640px] text-[15px] leading-relaxed text-ink-2">{c.intro}</p>

      <h2 className="mt-12 text-[20px] font-semibold">{c.h2cities}</h2>
      <ul className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
        {CITY_PAGES.map((city) => (
          <li key={city.slug}>
            <Link
              href={`/brelok/${city.slug}`}
              className="block rounded-[18px] border border-line-soft bg-white/70 px-4 py-3.5 text-[15px] font-semibold text-ink transition hover:border-[var(--accent)]"
            >
              {city.names[locale]}
            </Link>
          </li>
        ))}
      </ul>

      <section className="mt-14 max-w-[680px]">
        <h2 className="text-[20px] font-semibold">{c.h2how}</h2>
        <p className="mt-3 text-[15px] leading-relaxed text-ink-2">{c.pHow}</p>
      </section>

      <section className="mt-12 max-w-[680px]">
        <h2 className="text-[20px] font-semibold">{c.h2faq}</h2>
        <dl className="mt-4 flex flex-col gap-4">
          {c.faq.map((f) => (
            <div key={f.q}>
              <dt className="text-[15px] font-semibold text-ink">{f.q}</dt>
              <dd className="mt-1.5 text-[14.5px] leading-relaxed text-ink-2">{f.a}</dd>
            </div>
          ))}
        </dl>
      </section>

      <section className="mt-10 flex flex-wrap gap-3">
        <Link href="/keychains" className="inline-flex min-h-[44px] items-center justify-center rounded-[22px] bg-[var(--accent-strong)] px-5 py-2.5 text-sm font-semibold text-white transition hover:opacity-90">
          {c.ctaKeychain}
        </Link>
        <Link href="/maps" className="inline-flex min-h-[44px] items-center justify-center rounded-[22px] border border-line-soft bg-white/80 px-5 py-2.5 text-sm font-semibold text-ink transition hover:border-[var(--accent)]">
          {c.ctaMap}
        </Link>
      </section>
    </main>
  );
}
