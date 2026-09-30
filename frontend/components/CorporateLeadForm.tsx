"use client";

import { useState } from "react";
import { useLocale } from "next-intl";

/**
 * Заявка на корпоративний наклад прямо на /corporate (30.09.2026).
 * Раніше кнопка вела на загальну /contacts, де людина мала сама сформулювати
 * запит — для B2B це зайвий крок. Тут одразу все, що потрібно для розрахунку:
 * скільки, що, до якої дати. Йде тим самим /api/contact → Telegram CRM.
 */
const API = process.env.NEXT_PUBLIC_API_URL || "";

type L = {
  title: string; sub: string; company: string; name: string; phone: string; email: string;
  product: string; products: string[]; qty: string; deadline: string; deadlines: string[];
  comment: string; commentPh: string; send: string; sending: string; ok: string; err: string; phoneReq: string;
};

const COPY: Record<string, L> = {
  uk: {
    title: "Заявка на наклад", sub: "Відповімо протягом робочого дня з розрахунком і безкоштовним 3D-превʼю.",
    company: "Компанія", name: "Ваше імʼя", phone: "Телефон (Telegram / Viber)", email: "Пошта (необовʼязково)",
    product: "Що потрібно", products: ["Брелоки з районом офісу", "Мапи міста з текстом", "Магніти з локацією", "Різне / порадьте"],
    qty: "Кількість, шт", deadline: "До якої дати", deadlines: ["До Миколая (6 грудня)", "До Нового року", "До події / інша дата", "Не горить"],
    comment: "Деталі", commentPh: "Місто чи адреса офісу, текст на звороті, чи потрібні різні райони для кожного…",
    send: "Надіслати заявку", sending: "Надсилаємо…", ok: "Дякуємо! Заявку отримали — напишемо вам найближчим часом.",
    err: "Не вдалося надіслати. Напишіть нам у Telegram або спробуйте ще раз.", phoneReq: "Вкажіть телефон, щоб ми могли звʼязатися",
  },
  en: {
    title: "Request a quote", sub: "We reply within one working day with a price and a free 3D preview.",
    company: "Company", name: "Your name", phone: "Phone (Telegram / WhatsApp)", email: "Email (optional)",
    product: "What you need", products: ["Office-district keychains", "City maps with text", "Location magnets", "Mixed / advise me"],
    qty: "Quantity, pcs", deadline: "Needed by", deadlines: ["St Nicholas Day (6 Dec)", "New Year", "An event / other date", "No rush"],
    comment: "Details", commentPh: "Office city or address, text on the back, different districts per person…",
    send: "Send request", sending: "Sending…", ok: "Thank you! We got your request and will get back to you shortly.",
    err: "Could not send. Message us on Telegram or try again.", phoneReq: "Please add a phone number so we can reach you",
  },
  de: {
    title: "Angebot anfragen", sub: "Wir antworten innerhalb eines Werktags mit Preis und kostenloser 3D-Vorschau.",
    company: "Firma", name: "Ihr Name", phone: "Telefon (Telegram / WhatsApp)", email: "E-Mail (optional)",
    product: "Was Sie brauchen", products: ["Anhänger mit Büro-Viertel", "Stadtkarten mit Text", "Standort-Magnete", "Gemischt / beraten Sie mich"],
    qty: "Menge, Stk.", deadline: "Benötigt bis", deadlines: ["Nikolaus (6. Dez.)", "Neujahr", "Event / anderes Datum", "Keine Eile"],
    comment: "Details", commentPh: "Stadt oder Adresse des Büros, Text auf der Rückseite, verschiedene Viertel pro Person…",
    send: "Anfrage senden", sending: "Wird gesendet…", ok: "Danke! Wir haben Ihre Anfrage erhalten und melden uns bald.",
    err: "Senden fehlgeschlagen. Schreiben Sie uns auf Telegram oder versuchen Sie es erneut.", phoneReq: "Bitte Telefonnummer angeben",
  },
  fr: {
    title: "Demander un devis", sub: "Réponse sous un jour ouvré avec un prix et un aperçu 3D gratuit.",
    company: "Entreprise", name: "Votre nom", phone: "Téléphone (Telegram / WhatsApp)", email: "E-mail (facultatif)",
    product: "Ce qu'il vous faut", products: ["Porte-clés du quartier du bureau", "Cartes de ville avec texte", "Magnets de lieu", "Mixte / conseillez-moi"],
    qty: "Quantité, pcs", deadline: "Pour quand", deadlines: ["Saint-Nicolas (6 déc.)", "Nouvel An", "Un événement / autre date", "Pas pressé"],
    comment: "Détails", commentPh: "Ville ou adresse du bureau, texte au dos, quartiers différents par personne…",
    send: "Envoyer la demande", sending: "Envoi…", ok: "Merci ! Nous avons reçu votre demande et revenons vers vous rapidement.",
    err: "Échec de l'envoi. Écrivez-nous sur Telegram ou réessayez.", phoneReq: "Indiquez un téléphone pour que nous puissions vous joindre",
  },
  es: {
    title: "Pedir presupuesto", sub: "Respondemos en un día laborable con precio y vista previa 3D gratis.",
    company: "Empresa", name: "Tu nombre", phone: "Teléfono (Telegram / WhatsApp)", email: "Email (opcional)",
    product: "Qué necesitas", products: ["Llaveros del barrio de la oficina", "Mapas de ciudad con texto", "Imanes de ubicación", "Mixto / aconséjame"],
    qty: "Cantidad, uds.", deadline: "Para cuándo", deadlines: ["San Nicolás (6 dic.)", "Año Nuevo", "Un evento / otra fecha", "Sin prisa"],
    comment: "Detalles", commentPh: "Ciudad o dirección de la oficina, texto al dorso, barrios distintos por persona…",
    send: "Enviar solicitud", sending: "Enviando…", ok: "¡Gracias! Hemos recibido tu solicitud y te responderemos pronto.",
    err: "No se pudo enviar. Escríbenos por Telegram o inténtalo de nuevo.", phoneReq: "Indica un teléfono para poder contactarte",
  },
  pl: {
    title: "Zapytanie o wycenę", sub: "Odpowiadamy w ciągu dnia roboczego z ceną i darmowym podglądem 3D.",
    company: "Firma", name: "Imię", phone: "Telefon (Telegram / WhatsApp)", email: "E-mail (opcjonalnie)",
    product: "Czego potrzebujesz", products: ["Breloki z dzielnicą biura", "Mapy miasta z tekstem", "Magnesy z lokalizacją", "Różne / doradźcie"],
    qty: "Ilość, szt.", deadline: "Na kiedy", deadlines: ["Mikołajki (6 grudnia)", "Nowy Rok", "Wydarzenie / inna data", "Nie pilne"],
    comment: "Szczegóły", commentPh: "Miasto lub adres biura, tekst z tyłu, różne dzielnice dla każdej osoby…",
    send: "Wyślij zapytanie", sending: "Wysyłanie…", ok: "Dziękujemy! Otrzymaliśmy zapytanie i wkrótce się odezwiemy.",
    err: "Nie udało się wysłać. Napisz do nas na Telegramie lub spróbuj ponownie.", phoneReq: "Podaj telefon, abyśmy mogli się skontaktować",
  },
};

const clean = (v: string) => v.replace(/[<>]/g, "").trim();

export function CorporateLeadForm() {
  const locale = useLocale();
  const c = COPY[locale] || COPY.en;
  const [f, setF] = useState({ company: "", name: "", phone: "", email: "", product: 0, qty: "20", deadline: 1, comment: "" });
  const [state, setState] = useState<"idle" | "sending" | "ok" | "err">("idle");
  const [phoneErr, setPhoneErr] = useState(false);
  const set = (k: keyof typeof f) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) =>
    setF((p) => ({ ...p, [k]: k === "product" || k === "deadline" ? Number(e.target.value) : e.target.value }));

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (state === "sending") return;
    if (clean(f.phone).length < 6) { setPhoneErr(true); return; }
    setPhoneErr(false);
    setState("sending");
    const message = [
      "🏢 КОРПОРАТИВНИЙ НАКЛАД",
      `Компанія: ${clean(f.company) || "—"}`,
      `Що: ${COPY.uk.products[f.product]}`,
      `Кількість: ${clean(f.qty) || "—"} шт`,
      `Термін: ${COPY.uk.deadlines[f.deadline]}`,
      f.email ? `Пошта: ${clean(f.email)}` : "",
      f.comment ? `Деталі: ${clean(f.comment)}` : "",
      `Мова сторінки: ${locale}`,
    ].filter(Boolean).join("\n");
    try {
      const r = await fetch(`${API}/api/contact`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: clean(`${f.name}${f.company ? ` (${f.company})` : ""}`).slice(0, 80),
          phone: clean(f.phone).slice(0, 32),
          message: message.slice(0, 2000),
          source: "corporate-form",
        }),
      });
      if (!r.ok) throw new Error(String(r.status));
      setState("ok");
      import("@/lib/analytics").then((m) => m.track("b2b_lead", { qty: Number(f.qty) || 0, product: f.product, deadline: f.deadline })).catch(() => {});
    } catch {
      setState("err");
    }
  };

  const inp = "mt-1 w-full rounded-[12px] border border-line-soft bg-white px-3 py-2.5 text-[15px] text-ink outline-none focus:border-[var(--accent)]";
  const lab = "text-[13px] font-semibold text-ink-2";

  if (state === "ok") {
    return (
      <div id="b2b-form" className="scroll-mt-24 rounded-[20px] border border-[rgba(15,118,110,0.35)] bg-[rgba(15,118,110,0.07)] px-5 py-6 text-[15px] font-semibold text-ink" data-testid="b2b-ok">
        {c.ok}
      </div>
    );
  }

  return (
    <form id="b2b-form" onSubmit={submit} className="scroll-mt-24 rounded-[20px] border border-line-soft bg-white/80 px-5 py-5 sm:px-6" data-testid="b2b-form">
      <h2 className="text-[20px] font-semibold">{c.title}</h2>
      <p className="mt-1 text-[13.5px] leading-relaxed text-ink-2">{c.sub}</p>
      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <label className={lab}>{c.company}<input className={inp} value={f.company} onChange={set("company")} maxLength={80} autoComplete="organization" /></label>
        <label className={lab}>{c.name}<input className={inp} value={f.name} onChange={set("name")} maxLength={60} autoComplete="name" /></label>
        <label className={lab}>{c.phone} *
          <input className={inp} value={f.phone} onChange={set("phone")} maxLength={32} inputMode="tel" autoComplete="tel" required aria-invalid={phoneErr} />
          {phoneErr && <span className="mt-1 block text-[12px] font-semibold text-red-700">{c.phoneReq}</span>}
        </label>
        <label className={lab}>{c.email}<input className={inp} value={f.email} onChange={set("email")} maxLength={120} inputMode="email" autoComplete="email" /></label>
        <label className={lab}>{c.product}
          <select className={inp} value={f.product} onChange={set("product")}>
            {c.products.map((p, i) => <option key={p} value={i}>{p}</option>)}
          </select>
        </label>
        <label className={lab}>{c.qty}<input className={inp} value={f.qty} onChange={set("qty")} inputMode="numeric" maxLength={6} /></label>
        <label className={`${lab} sm:col-span-2`}>{c.deadline}
          <select className={inp} value={f.deadline} onChange={set("deadline")}>
            {c.deadlines.map((d, i) => <option key={d} value={i}>{d}</option>)}
          </select>
        </label>
        <label className={`${lab} sm:col-span-2`}>{c.comment}
          <textarea className={`${inp} min-h-[88px]`} value={f.comment} onChange={set("comment")} maxLength={1000} placeholder={c.commentPh} />
        </label>
      </div>
      {state === "err" && <p className="mt-3 text-[13px] font-semibold text-red-700">{c.err}</p>}
      <button type="submit" disabled={state === "sending"} data-testid="b2b-submit"
        className="mt-4 inline-flex min-h-[46px] w-full items-center justify-center rounded-[22px] bg-[var(--accent-strong)] px-5 py-2.5 text-[15px] font-semibold text-white transition hover:opacity-90 disabled:opacity-60 sm:w-auto">
        {state === "sending" ? c.sending : c.send}
      </button>
    </form>
  );
}
