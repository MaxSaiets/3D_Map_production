import type { LegalSet } from "./content";

export const de: LegalSet = {
  offer: {
    title: "Vertrag des öffentlichen Angebots",
    intro: [
      "Dieses Dokument ist ein offizielles öffentliches Angebot (Offerte) von {ownerFull} (nachfolgend „Verkäufer“), einen Kaufvertrag über Waren und die Erbringung von Dienstleistungen zu den nachstehenden Bedingungen abzuschließen. Mit der Bezahlung einer Bestellung auf der Website {domain} (nachfolgend „Website“) bestätigst du, dass du die Bedingungen dieses Vertrags vollständig gelesen, verstanden und vorbehaltlos angenommen hast (Annahme des Angebots gemäß Art. 633, 641, 642 des Zivilgesetzbuchs der Ukraine).",
    ],
    sections: [
      { h: "1. Begriffe", blocks: [
        { p: "Verkäufer — {ownerShort}, Steuernummer {taxId}." },
        { p: "Käufer — jede geschäftsfähige Person, die eine Bestellung auf der Website aufgegeben hat." },
        { p: "Ware / Dienstleistung — ein digitales 3D-Modell (3MF-/STL-Datei) einer Stadtkarte oder eines Schlüsselanhänger-Kartenmodells und/oder die Herstellung (3D-Druck) eines physischen Produkts aus dem Biokunststoff Eco PLA nach individueller Bestellung des Käufers." },
      ] },
      { h: "2. Gegenstand des Vertrags", blocks: [
        { p: "Der Verkäufer verpflichtet sich, dem Käufer ein digitales 3D-Modell bereitzustellen und/oder ein physisches Produkt auf Bestellung herzustellen und zu übergeben, und der Käufer verpflichtet sich, diese gemäß den Bedingungen dieses Vertrags anzunehmen und zu bezahlen." },
        { p: "Jedes Produkt wird individuell nach den Parametern (Ort auf der Karte, Form, Größe, Text) hergestellt, die du selbst im Konfigurator der Website auswählst, und ist somit eine auf Bestellung angefertigte Ware." },
      ] },
      { h: "3. Bestellabwicklung", blocks: [
        { p: "Du erstellst die Bestellung im Konfigurator auf der Website und gibst deine Kontaktdaten an (Name, Telefon, Versandart und Lieferadresse). Die Bestellung gilt als angenommen, sobald sie bezahlt oder vom Operator bestätigt wurde." },
        { p: "Du bist für die Richtigkeit der angegebenen Daten verantwortlich. Der Verkäufer haftet nicht für Folgen, die durch Fehler in den von dir angegebenen Daten entstehen." },
      ] },
      { h: "4. Preis und Zahlung", blocks: [
        { p: "Die Preise für Waren und Dienstleistungen sind auf der Website in Hrywnja (für Bestellungen innerhalb der Ukraine) und in Euro (für die Lieferung in die EU) angegeben und sind bis zur Bestätigung der Bestellung Richtwerte. Den endgültigen Preis siehst du im Bestellschritt." },
        { p: "Die Zahlung erfolgt online über den Zahlungsdienst LiqPay (mit Visa-/Mastercard-Bankkarte) oder auf eine andere vereinbarte Weise. Der Download der fertigen digitalen Datei im Rahmen des kostenlosen Limits ist kostenlos." },
        { p: "Details findest du auf der Seite [delivery:„Zahlung und Lieferung“]." },
      ] },
      { h: "5. Herstellung und Lieferung", blocks: [
        { p: "Die digitale Datei wird sofort oder nach Bestätigung der Bestellung im Konto/per E-Mail bereitgestellt. Das physische Produkt wird innerhalb der auf der Seite „Zahlung und Lieferung“ angegebenen Frist hergestellt und mit den Diensten Nova Poshta oder Ukrposhta (Ukraine) bzw. Nova Post EU oder Meest (EU) versendet." },
      ] },
      { h: "6. Rückerstattung", blocks: [
        { p: "Da die Waren auf individuelle Bestellung gefertigt werden und digitale Dateien den Charakter elektronischer Inhalte haben, wird die Rückerstattung durch ein gesondertes Dokument geregelt — [refund:„Rückgabe und Rückerstattung“], das ein untrennbarer Bestandteil dieses Vertrags ist." },
      ] },
      { h: "7. Rechte des geistigen Eigentums", blocks: [
        { p: "Die Kartendaten stammen von OpenStreetMap (ODbL), die Höhendaten aus offenen Quellen. Das generierte 3D-Modell wird dir für den persönlichen, nicht kommerziellen Gebrauch und Druck bereitgestellt. Eine von dir hochgeladene GPX-Route sind deine eigenen Daten und werden ausschließlich zur Erstellung des Modells verarbeitet (siehe [privacy:Datenschutzerklärung])." },
      ] },
      { h: "8. Haftung der Parteien", blocks: [
        { p: "Der Verkäufer haftet nicht für die Druckqualität auf deinem eigenen Gerät, wenn du die heruntergeladene Datei selbst druckst. Die Gesamthaftung des Verkäufers ist auf den Betrag der bezahlten Bestellung begrenzt." },
        { p: "Die Parteien werden von der Haftung für die Nichterfüllung von Verpflichtungen infolge höherer Gewalt (Force Majeure) befreit." },
      ] },
      { h: "9. Personenbezogene Daten", blocks: [
        { p: "Mit der Aufgabe der Bestellung erteilst du deine Einwilligung zur Verarbeitung deiner personenbezogenen Daten zur Ausführung der Bestellung gemäß dem ukrainischen Gesetz „Über den Schutz personenbezogener Daten“ und der [privacy:Datenschutzerklärung]." },
      ] },
      { h: "10. Streitbeilegung und Laufzeit", blocks: [
        { p: "Streitigkeiten werden im Wege von Verhandlungen beigelegt und, falls keine Einigung erzielt wird, gemäß dem geltenden Recht der Ukraine. Der Vertrag gilt ab dem Zeitpunkt der Annahme bis zur vollständigen Erfüllung der Verpflichtungen durch die Parteien. Der Verkäufer ist berechtigt, die Bedingungen zu ändern, indem er eine neue Fassung auf dieser Seite veröffentlicht." },
      ] },
      { h: "11. Angaben zum Verkäufer", blocks: [
        { kv: [
          { k: "Name", v: "{ownerFull}" },
          { k: "Steuernummer (IPN/RNOKPP)", v: "{taxId}" },
          { k: "IBAN", v: "{iban}" },
          { k: "Geschäft", v: "{storeName}, {storeAddress}" },
          { k: "Email", v: "{email}" },
          { k: "Telefon", v: "{phone}" },
        ] },
      ] },
    ],
  },

  refund: {
    title: "Rückgabe und Rückerstattung",
    sections: [
      { h: "Art der Ware", blocks: [
        { p: "Alle Produkte von {storeName} werden individuell nach den Parametern hergestellt, die du selbst auswählst (Ort auf der Karte, Form, Größe, Text), und die digitalen 3D-Modelle sind elektronische Inhalte. Das wirkt sich auf die Rückgabebedingungen gemäß dem ukrainischen Verbraucherschutzgesetz aus (eine mangelfreie Ware, die auf individuelle Bestellung gefertigt wurde, ist von Rückgabe und Umtausch ausgeschlossen)." },
      ] },
      { h: "Digitale Dateien (3MF-/STL-Download)", blocks: [
        { p: "Die Kosten für eine digitale Datei werden nicht erstattet, nachdem die Datei generiert und zum Download bereitgestellt wurde, da die Leistung bereits vollständig erbracht wurde. Wenn sich die Datei aus technischen Gründen nicht herunterladen lässt oder von unserer Seite beschädigt ist, generieren wir sie kostenlos neu oder erstatten dir den Betrag." },
      ] },
      { h: "Druck auf Bestellung (physisches Produkt)", blocks: [
        { ul: [
          "Vor Produktionsbeginn — du kannst die Bestellung stornieren und eine vollständige Rückerstattung erhalten, sofern wir mit dem Druck noch nicht begonnen haben.",
          "Nach Produktionsbeginn — der Betrag wird nicht erstattet, da das Produkt persönlich für deine Bestellung gefertigt wird.",
          "Mangel, Transportschaden oder Abweichung von der Bestellung — wir drucken das Produkt kostenlos neu und versenden es oder erstatten den vollen Betrag (nach deiner Wahl).",
        ] },
      ] },
      { h: "So beantragst du eine Rückerstattung", blocks: [
        { p: "Schreib an {email} oder ruf {phone} an und gib die Bestellnummer, den Grund und (bei einem Mangel) ein Foto des Produkts an. Wir bearbeiten dein Anliegen innerhalb von 1–3 Werktagen." },
        { p: "Die Rückerstattung erfolgt auf demselben Weg, auf dem die Zahlung getätigt wurde (Rückerstattung auf die Karte über LiqPay), innerhalb der durch die Regeln des Zahlungssystems und der Bank vorgesehenen Fristen." },
      ] },
      { h: "Kontakt", blocks: [
        { p: "Dieses Dokument ist ein untrennbarer Bestandteil des [offer:Vertrags des öffentlichen Angebots]. Fragen: {email}." },
      ] },
    ],
  },

  delivery: {
    title: "Zahlung und Lieferung",
    sections: [
      { h: "Produkte und Preise", blocks: [
        { p: "Produktpreise (Lieferung wird separat nach Tarif des Zustellers berechnet):" },
        { ul: [
          "Schlüsselanhänger-Karte — ab 120 ₴ (≈ 3 €).",
          "3D-Karte eines Stadtviertels: S 5,5 cm — 250 ₴, M 8 cm — 350 ₴, L 11 cm — 450 ₴, XL 15 cm — 550 ₴ (Karten — ab 6 €).",
          "Kühlschrankmagnet (Karte) — 150 ₴.",
          "Download der fertigen 3MF-/STL-Datei zum Selbstdrucken — kostenlos im Rahmen des Kontolimits.",
        ] },
      ] },
      { h: "Zahlung", blocks: [
        { p: "Online-Zahlung mit Visa-/Mastercard-Bankkarte über den sicheren Dienst LiqPay. Die Kartendaten werden auf der Seite des Zahlungssystems verarbeitet — wir speichern sie nicht. Eine Zahlung nach Absprache mit dem Operator ist ebenfalls möglich. Die digitale Datei wird im Rahmen des kostenlosen Limits ohne Bezahlung bereitgestellt." },
      ] },
      { h: "Herstellung", blocks: [
        { p: "Die Produkte werden auf Bestellung aus dem Biokunststoff Eco PLA gedruckt. Die voraussichtliche Herstellungszeit beträgt 1–3 Werktage nach Bestätigung der Bestellung (je nach Auslastung und Komplexität)." },
      ] },
      { h: "Lieferung", blocks: [
        { p: "Ukraine:" },
        { ul: ["Nova Poshta — Filiale oder Paketautomat.", "Ukrposhta — Filiale."] },
        { p: "Europäische Union (15 Länder):" },
        { ul: ["Nova Post EU.", "Meest."] },
        { p: "Die Versandkosten werden nach den Tarifen des Transportunternehmens berechnet und separat bezahlt (in der Regel bei Erhalt). Die voraussichtliche Lieferzeit innerhalb der Ukraine beträgt 2–4 Werktage nach dem Versand." },
      ] },
      { h: "Rückgabe", blocks: [
        { p: "Die Bedingungen für die Rückerstattung sind auf der Seite [refund:„Rückgabe und Rückerstattung“] beschrieben. Die allgemeinen Bedingungen findest du im [offer:Vertrag des öffentlichen Angebots]." },
      ] },
      { h: "Kontakt", blocks: [
        { p: "Fragen zur Zahlung oder Lieferung: {email}, {phone}." },
      ] },
    ],
  },

  contacts: {
    title: "Kontakt und Angaben",
    sections: [
      { h: "Kontaktiere uns", blocks: [
        { kv: [
          { k: "Email", v: "{email}" },
          { k: "Telefon", v: "{phone}" },
          { k: "Website", v: "{domain}" },
        ] },
        { p: "Bearbeitungszeiten der Bestellungen: täglich, wir antworten innerhalb eines Tages." },
      ] },
      { h: "Geschäft", blocks: [
        { kv: [
          { k: "Name", v: "{storeName}" },
          { k: "Geschäftsadresse", v: "{storeAddress}" },
        ] },
      ] },
      { h: "Verkäufer (Einzelunternehmer/FOP)", blocks: [
        { kv: [
          { k: "Name", v: "{ownerFull}" },
          { k: "Steuernummer (IPN/RNOKPP)", v: "{taxId}" },
          { k: "Tätigkeit (KVED)", v: "{ved}" },
          { k: "IBAN", v: "{iban}" },
        ] },
      ] },
      { h: "Was wir verkaufen", blocks: [
        { p: "{storeName} — das sind 3D-Modelle von Stadtkarten und Schlüsselanhänger-Karten auf Bestellung. Du kannst die fertige Datei zum Drucken (3MF/STL) im Rahmen des kostenlosen Limits herunterladen oder den Druck eines Produkts aus dem Biokunststoff Eco PLA mit Lieferung bestellen. Die Richtpreise findest du im Konfigurator und auf der Seite [delivery:„Zahlung und Lieferung“]." },
      ] },
      { h: "Dokumente", blocks: [
        { ul: [
          "[offer:Vertrag des öffentlichen Angebots]",
          "[refund:Rückgabe und Rückerstattung]",
          "[delivery:Zahlung und Lieferung]",
          "[privacy:Datenschutzerklärung]",
          "[terms:Nutzungsbedingungen]",
        ] },
      ] },
    ],
  },

  privacy: {
    title: "Datenschutzerklärung",
    sections: [
      { h: "Welche Daten wir erheben", blocks: [
        { p: "Name, E-Mail oder Telefonnummer (beim Anmelden/Registrieren) sowie die Kontaktdaten und die Lieferadresse, die du bei der Bestellung angibst. Technische Daten (der Verlauf der generierten Modelle) werden in deinem Konto gespeichert." },
      ] },
      { h: "Wie wir die Daten verwenden", blocks: [
        { p: "Ausschließlich zur Erbringung der Dienstleistung: Anmeldung im Konto, Speicherung des Modellverlaufs, Bearbeitung und Lieferung von Bestellungen, Kontakt mit dir bezüglich der Bestellung. Wir verkaufen deine Daten nicht und geben sie nicht zu Werbezwecken an Dritte weiter." },
      ] },
      { h: "Hochgeladene Routen (GPX) und Geodaten", blocks: [
        { p: "Wenn du eine GPX-Datei hochlädst (zum Beispiel den Export deiner eigenen Aktivität aus Strava oder einer anderen App), verarbeiten wir die Koordinaten der Route ausschließlich zur Erstellung deines 3D-Modells. Das sind deine eigenen Daten — wir veröffentlichen sie nicht, geben sie nicht an Dritte weiter und verwenden sie nicht für Werbung. Die Punkte der Route werden ausgedünnt und genau so lange gespeichert, wie es für die Generierung und (falls du angemeldet bist) für die Führung des Modellverlaufs in deinem Konto erforderlich ist; du kannst jederzeit ihre Löschung verlangen." },
        { p: "Die Ortssuche auf der Karte sendet deine Anfrage an den Geokodierungsdienst von OpenStreetMap (Nominatim), und die Karten selbst werden aus den Kacheln von OpenStreetMap geladen — gemäß deren Nutzungsbedingungen. Wir geben deinen Namen oder deine Kontaktdaten nicht an diese Dienste weiter." },
      ] },
      { h: "Speicherung und Dienste", blocks: [
        { p: "Die Authentifizierung erfolgt über Google Firebase Authentication. Die Website ist durch Cloudflare geschützt. Bestellungen werden manuell bearbeitet. Die Daten werden auf einem geschützten Server genau so lange gespeichert, wie es für die Ausführung der Bestellung und die Führung des Verlaufs erforderlich ist." },
      ] },
      { h: "Cookies und Analyse", blocks: [
        { p: "Wir verwenden die datenschutzfreundliche Analyse von Cloudflare ohne Werbe-Cookies von Drittanbietern. Cookies werden ausschließlich für die Funktion der Konto-Anmeldung verwendet." },
      ] },
      { h: "Deine Rechte", blocks: [
        { p: "Du kannst die Löschung deines Kontos und der damit verbundenen Daten verlangen. Schreib uns an {email}." },
      ] },
      { h: "Kontakt", blocks: [
        { p: "Bei Fragen zum Datenschutz: {email}." },
      ] },
    ],
  },

  terms: {
    title: "Nutzungsbedingungen",
    sections: [
      { h: "Über den Dienst", blocks: [
        { p: "{storeName} ermöglicht es dir, ein 3D-Modell eines Stadtausschnitts oder eines Schlüsselanhängers mit einer Karte auf Basis der offenen Daten von OpenStreetMap zu erstellen und die fertige Datei für den 3D-Druck (3MF/STL) herunterzuladen oder den Druck zu bestellen." },
      ] },
      { h: "Konto und kostenlose Downloads", blocks: [
        { p: "Für den Download des vollständigen Modells ist ein Konto erforderlich. Jedem Nutzer stehen 5 kostenlose Downloads zur Verfügung. Danach — nach Absprache (Druck/Zahlung), Kontakt über die Website." },
      ] },
      { h: "Daten und Urheberrecht", blocks: [
        { p: "Kartendaten © OpenStreetMap contributors (ODbL). Die generierten Dateien darfst du für den persönlichen Druck verwenden. Der Weiterverkauf des Dienstes oder eine massenhafte kommerzielle Nutzung bedarf einer gesonderten Vereinbarung." },
      ] },
      { h: "Bestellung und Zahlung", blocks: [
        { p: "Die Bestellung erfolgt über die Website; die Zahlung — online über LiqPay oder nach Absprache. Details zu Preisen und Lieferung findest du auf der Seite [delivery:„Zahlung und Lieferung“], die vollständigen Bedingungen im [offer:Vertrag des öffentlichen Angebots]." },
      ] },
      { h: "Haftung", blocks: [
        { p: "Der Dienst wird „wie besehen“ bereitgestellt. Wir streben höchste Genauigkeit der Modelle an, garantieren jedoch aufgrund der Beschränkungen der OSM-Ausgangsdaten keine vollständige Übereinstimmung mit realen Objekten." },
      ] },
      { h: "Kontakt", blocks: [
        { p: "Fragen: {email}." },
      ] },
    ],
  },
};
