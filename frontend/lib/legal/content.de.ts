import type { LegalSet } from "./content";

export const de: LegalSet = {
  offer: {
    title: "Vertrag des öffentlichen Angebots",
    intro: [
      "Dieses Dokument ist ein offizielles öffentliches Angebot (Offerte) von {ownerFull} (nachfolgend „Verkäufer“), einen Vertrag über den Fernabsatz von Waren und die Erbringung von Dienstleistungen zu den nachstehenden Bedingungen abzuschließen, gemäß Art. 633, 641, 642 des Zivilgesetzbuchs der Ukraine sowie den Gesetzen der Ukraine „Über den elektronischen Handel“ und „Über den Schutz der Verbraucherrechte“.",
      "Mit der Aufgabe und/oder Bezahlung einer Bestellung auf der Website {domain} (nachfolgend „Website“) bestätigt der Käufer, dass er die Bedingungen dieses Vertrags vollständig gelesen, verstanden und vorbehaltlos angenommen hat (Annahme des Angebots). Die Annahme dieses Angebots ist dem Abschluss eines Vertrags in Schriftform gleichgestellt.",
    ],
    sections: [
      { h: "1. Begriffe und Definitionen", blocks: [
        { kv: [
          { k: "Verkäufer", v: "{ownerFull}, Steuernummer (IPN/RNOKPP) {taxId}, Zahler der Einheitssteuer." },
          { k: "Käufer", v: "jede geschäftsfähige natürliche oder juristische Person, die eine Bestellung auf der Website aufgegeben und dieses Angebot angenommen hat." },
          { k: "Website", v: "der Online-Shop {storeName} unter der Adresse {domain}, einschließlich des Online-Konfigurators für 3D-Modelle." },
          { k: "Konfigurator", v: "der Software-Dienst der Website, in dem der Käufer die Parameter des künftigen Produkts selbst auswählt: Kartenausschnitt, Form, Größe, Format, Gravurtext usw." },
          { k: "Ware", v: "ein physisches Produkt (3D-gedruckte Stadtkarte, Wandpanneau, Schlüsselanhänger-Karte, Magnet usw.) aus dem Biokunststoff Eco PLA, das nach individueller Bestellung des Käufers gefertigt wird." },
          { k: "Digitaler Inhalt", v: "ein generiertes digitales 3D-Modell (Datei im Format 3MF/STL), das für den eigenständigen 3D-Druck geeignet ist." },
          { k: "Bestellung", v: "die über die Website aufgegebene Anfrage des Käufers zum Erwerb der Ware und/oder des Digitalen Inhalts." },
        ] },
      ] },
      { h: "2. Vertragsgegenstand", blocks: [
        { p: "Der Verkäufer verpflichtet sich, im Auftrag des Käufers ein digitales 3D-Modell zu generieren und/oder ein physisches Produkt nach den individuellen Parametern des Käufers zu fertigen und ihm zu übergeben; der Käufer verpflichtet sich, die Ware/den Digitalen Inhalt zu den Bedingungen dieses Vertrags anzunehmen und zu bezahlen." },
        { p: "Jedes Produkt wird individuell nach den Parametern (Ort auf der Karte, Form, Größe, Text, Farben) gefertigt, die der Käufer selbst im Konfigurator auswählt. Die Ware ist somit ein auf individuelle Bestellung des Käufers gefertigtes Erzeugnis im Sinne des Gesetzes der Ukraine „Über den Schutz der Verbraucherrechte“." },
        { p: "Sortiment, Eigenschaften und Richtpreise der Waren sind auf der Website angegeben. Fotos und 3D-Vorschauen dienen der Information: Das fertige Produkt kann geringfügige Abweichungen im Farbton des Materials und in der Schichtstruktur aufweisen; dies ist eine natürliche Eigenschaft der 3D-Drucktechnologie und stellt keinen Mangel der Ware dar." },
      ] },
      { h: "3. Bestellvorgang", blocks: [
        { ul: [
          "Der Käufer erstellt das Modell selbst im Konfigurator: Er wählt Stadt/Kartenausschnitt, Form, Größe, Format (Relief/flach, Magnet) und optional einen Gravurtext oder eine GPX-Route.",
          "Vor der Bestellung sieht der Käufer eine 3D-Vorschau des Modells und den endgültigen Bestellwert.",
          "Zur Bestellung gibt der Käufer seine Kontaktdaten an: Name, Telefon, bei Bedarf E-Mail, Versandart und Adresse/Filiale des Transportdienstleisters.",
          "Die Bestellung gilt als zur Ausführung angenommen, nachdem sie online bezahlt oder vom Operator bestätigt wurde (nach Vereinbarung).",
        ] },
        { p: "Der Käufer ist selbst für die Richtigkeit der gewählten Modellparameter (Kartenausschnitt, Text, Größe) und die Korrektheit der Kontaktdaten verantwortlich. Der Verkäufer haftet nicht für die Folgen von Fehlern in den vom Käufer angegebenen Daten, insbesondere nicht für die Fertigung eines Produkts mit fehlerhaftem Text oder Kartenausschnitt, die der Käufer selbst im Konfigurator bestätigt hat." },
      ] },
      { h: "4. Preis und Zahlung", blocks: [
        { p: "Die Preise auf der Website sind in Hrywnja (für Bestellungen innerhalb der Ukraine) und in Euro (Richtwert) angegeben. Den endgültigen Bestellwert sieht der Käufer im Bestellschritt vor der Zahlung. Die Versandkosten sind nicht im Warenpreis enthalten und werden gesondert nach den Tarifen des Transportdienstleisters bezahlt." },
        { p: "Die Zahlung erfolgt online über den Zahlungsdienst LiqPay (JSC CB „PrivatBank“): mit Visa-/Mastercard-Bankkarte und über andere in LiqPay verfügbare Methoden. Die Daten der Zahlungskarte werden auf Seiten des Zahlungssystems verarbeitet; der Verkäufer erhält und speichert sie nicht." },
        { p: "Nach Vereinbarung mit dem Operator ist die Zahlung auf eine andere abgestimmte Weise möglich. Der Download des fertigen Digitalen Inhalts im Rahmen des kostenlosen Kontolimits ist kostenlos." },
        { p: "Die Ware wird vor der Übergabe in die Produktion vollständig bezahlt, sofern die Parteien nichts anderes vereinbart haben. Details finden Sie auf der Seite [delivery:„Zahlung und Lieferung“]." },
      ] },
      { h: "5. Fertigungsfristen", blocks: [
        { p: "Der Digitale Inhalt wird automatisch generiert und im Konto des Käufers/per E-Mail unmittelbar nach der Generierung oder nach Bestätigung der Bestellung bereitgestellt." },
        { p: "Die voraussichtliche Fertigungsdauer eines physischen Produkts beträgt 2–4 Werktage ab Zahlung/Bestätigung der Bestellung. Bei erhöhter Auslastung oder technischer Komplexität des Produkts kann sich die Frist verlängern; der Verkäufer informiert den Käufer darüber." },
      ] },
      { h: "6. Lieferung", blocks: [
        { p: "Die Lieferung innerhalb der Ukraine erfolgt durch „Nova Poshta“ (Filiale, Paketautomat) oder „Ukrposhta“. Die voraussichtliche Lieferzeit innerhalb der Ukraine beträgt 2–4 Werktage nach dem Versand." },
        { p: "Die Versandkosten werden nach den Tarifen des Transportdienstleisters berechnet und vom Käufer gesondert bezahlt (in der Regel bei Erhalt). Das Eigentum an der Ware und die Gefahr der zufälligen Beschädigung gehen mit dem Erhalt der Ware vom Transportdienstleister auf den Käufer über." },
        { p: "Bei Erhalt ist der Käufer verpflichtet, die Sendung auf Unversehrtheit der Verpackung und des Produkts zu prüfen. Bei Transportschäden ist ein Schadensprotokoll des Transportdienstleisters zu erstellen und der Verkäufer zu benachrichtigen — ein solcher Fall wird durch kostenlosen Neudruck oder Rückerstattung gelöst (siehe Abschnitt 8)." },
      ] },
      { h: "7. Qualität und Garantie", blocks: [
        { p: "Der Verkäufer garantiert die Übereinstimmung des Produkts mit den vom Käufer im Konfigurator bestätigten Parametern sowie eine ordnungsgemäße Druckqualität. Auf physische Produkte wird eine Garantie von 60 Tagen ab Erhalt gewährt, die Druckfehler und Schichtablösungen abdeckt, die nicht durch Verschulden des Käufers entstanden sind." },
        { p: "Natürliche Eigenschaften der FDM-Drucktechnologie (sichtbare Druckschichten, geringfügige Farbtonunterschiede des Kunststoffs zwischen Chargen, technologische Spuren an der Unterseite) stellen keine Mängel der Ware dar." },
        { p: "Der Verkäufer haftet nicht für das Druckergebnis auf der Ausrüstung des Käufers beim eigenständigen Druck des heruntergeladenen Digitalen Inhalts (die Qualität eines solchen Drucks hängt vom Drucker, Material und den Einstellungen des Käufers ab)." },
      ] },
      { h: "8. Rückerstattung und Umtausch", blocks: [
        { p: "Da die Ware auf individuelle Bestellung nach den einzigartigen Parametern des Käufers gefertigt wird, ist eine Ware von angemessener Qualität von Rückgabe und Umtausch ausgeschlossen (Gesetz der Ukraine „Über den Schutz der Verbraucherrechte“; Warenliste, genehmigt durch die Verordnung des Ministerkabinetts der Ukraine Nr. 172 vom 19.03.1994). Digitaler Inhalt ist nach Bereitstellung des Download-Zugangs als elektronischer Inhalt, dessen Leistung erbracht wurde, von der Rückerstattung ausgeschlossen." },
        { ul: [
          "Bis zur Übergabe der Bestellung in die Produktion kann der Käufer sie stornieren und eine vollständige Rückerstattung erhalten.",
          "Bei Mängeln, Transportschäden oder Abweichungen des Produkts von den bestätigten Parametern fertigt und versendet der Verkäufer nach Wahl des Käufers kostenlos ein neues Produkt oder erstattet den vollen Preis.",
          "Ist der Digitale Inhalt technisch beschädigt oder lässt er sich durch Verschulden des Verkäufers nicht herunterladen, wird die Datei kostenlos neu generiert oder der Betrag erstattet.",
        ] },
        { p: "Das Verfahren und die Fristen sind im Dokument [refund:„Rückgabe und Rückerstattung“] beschrieben, das ein untrennbarer Bestandteil dieses Vertrags ist. Die Rückerstattung erfolgt auf demselben Weg, auf dem die Zahlung getätigt wurde (auf die Karte über LiqPay), innerhalb der Fristen, die durch die Regeln des Zahlungssystems und der Bank vorgesehen sind." },
      ] },
      { h: "9. Rechte und Pflichten der Parteien", blocks: [
        { p: "Der Verkäufer ist verpflichtet: die Ware gemäß den vom Käufer bestätigten Parametern zu fertigen; die angegebenen Fristen einzuhalten; den Käufer über den Status der Bestellung zu informieren; die Vertraulichkeit der personenbezogenen Daten des Käufers zu gewährleisten." },
        { p: "Der Verkäufer ist berechtigt: Dritte zur Erfüllung seiner Verpflichtungen heranzuziehen (Transportdienstleister, Zahlungsdienste); die Ausführung der Bestellung bei Nichtzahlung auszusetzen; die Fertigung eines Produkts abzulehnen, dessen Inhalt gegen die Gesetzgebung der Ukraine verstößt (insbesondere verbotene Symbolik oder Hassrede enthält), mit vollständiger Rückerstattung." },
        { p: "Der Käufer ist verpflichtet: zutreffende Daten für die Ausführung der Bestellung anzugeben; die Bestellung zu bezahlen; die Ware innerhalb der Aufbewahrungsfrist der Sendung beim Transportdienstleister abzuholen." },
        { p: "Der Käufer ist berechtigt: eine Ware von angemessener Qualität innerhalb der angegebenen Frist zu erhalten; Informationen über den Status seiner Bestellung zu erhalten; eine Reklamation in dem in diesem Vertrag vorgesehenen Verfahren einzureichen." },
      ] },
      { h: "10. Rechte des geistigen Eigentums", blocks: [
        { p: "Die Kartendaten stammen von OpenStreetMap (© OpenStreetMap contributors, ODbL-Lizenz); die Höhendaten stammen aus offenen Quellen. Das generierte 3D-Modell wird dem Käufer zur persönlichen, nicht kommerziellen Nutzung und zum Druck bereitgestellt. Massenhafte kommerzielle Nutzung oder Weiterverkauf der Modelle bedarf einer gesonderten schriftlichen Vereinbarung mit dem Verkäufer." },
        { p: "Eine vom Käufer hochgeladene GPX-Route sind eigene Daten des Käufers und wird ausschließlich zur Erstellung seines Modells verarbeitet (siehe [privacy:Datenschutzerklärung]). Der Käufer garantiert, dass der von ihm bestellte Gravurtext keine Rechte Dritter verletzt." },
      ] },
      { h: "11. Personenbezogene Daten", blocks: [
        { p: "Mit der Aufgabe der Bestellung erteilt der Käufer seine Einwilligung zur Verarbeitung seiner personenbezogenen Daten (Name, Kontaktdaten, Lieferadresse) ausschließlich zum Zweck der Erfüllung dieses Vertrags gemäß dem Gesetz der Ukraine „Über den Schutz personenbezogener Daten“ und der [privacy:Datenschutzerklärung]. Die Daten werden nicht an Dritte weitergegeben, außer in Fällen, die für die Ausführung der Bestellung erforderlich sind (Transportdienstleister, Zahlungsdienst)." },
      ] },
      { h: "12. Haftung und höhere Gewalt", blocks: [
        { p: "Für die Nichterfüllung oder nicht ordnungsgemäße Erfüllung von Verpflichtungen haften die Parteien nach geltendem Recht der Ukraine. Die Gesamthaftung des Verkäufers für sämtliche Ansprüche ist auf den vom Käufer tatsächlich gezahlten Bestellbetrag begrenzt." },
        { p: "Die Parteien sind von der Haftung für die vollständige oder teilweise Nichterfüllung ihrer Verpflichtungen befreit, wenn diese auf Umständen höherer Gewalt beruht: Kriegshandlungen, Beschuss, Stromausfälle, Naturkatastrophen, Entscheidungen von Behörden, Störungen bei Transportdienstleistern usw. Die Erfüllungsfristen verlängern sich um die Dauer solcher Umstände." },
      ] },
      { h: "13. Reklamationen und Streitbeilegung", blocks: [
        { p: "Reklamationen zur Bestellung werden unter Angabe der Bestellnummer an {email} oder telefonisch unter {phone} entgegengenommen. Der Verkäufer bearbeitet Anfragen innerhalb von 2–4 Werktagen. Streitigkeiten werden durch Verhandlungen beigelegt; kommt keine Einigung zustande, erfolgt die Beilegung nach dem geltenden Recht der Ukraine." },
      ] },
      { h: "14. Laufzeit und Änderung der Bedingungen", blocks: [
        { p: "Der Vertrag tritt mit der Annahme des Angebots durch den Käufer in Kraft und gilt bis zur vollständigen Erfüllung der Verpflichtungen durch die Parteien. Der Verkäufer ist berechtigt, die Bedingungen dieses Angebots zu ändern, indem er eine neue Fassung auf dieser Seite veröffentlicht; die neue Fassung gilt für Bestellungen, die nach ihrer Veröffentlichung aufgegeben werden. Die aktuelle Fassung ist jederzeit unter {domain}/offer verfügbar." },
        { p: "Untrennbare Bestandteile dieses Vertrags sind die Dokumente: [refund:„Rückgabe und Rückerstattung“], [delivery:„Zahlung und Lieferung“], [privacy:„Datenschutzerklärung“] und [terms:„Nutzungsbedingungen“]." },
      ] },
      { h: "15. Angaben zum Verkäufer", blocks: [
        { kv: [
          { k: "Verkäufer", v: "{ownerFull}" },
          { k: "Steuernummer (IPN/RNOKPP)", v: "{taxId}" },
          { k: "Tätigkeitsart (KVED)", v: "{ved}" },
          { k: "IBAN", v: "{iban}" },
          { k: "Geschäft", v: "{storeName}, {storeAddress}" },
          { k: "E-Mail", v: "{email}" },
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
        { p: "Schreib an {email} oder ruf {phone} an und gib die Bestellnummer, den Grund und (bei einem Mangel) ein Foto des Produkts an. Wir bearbeiten dein Anliegen innerhalb von 2–4 Werktagen." },
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
        { p: "Die Produkte werden auf Bestellung aus dem Biokunststoff Eco PLA gedruckt. Die voraussichtliche Herstellungszeit beträgt 2–4 Werktage nach Bestätigung der Bestellung (je nach Auslastung und Komplexität)." },
      ] },
      { h: "Lieferung", blocks: [
        { p: "Ukraine:" },
        { ul: ["Nova Poshta — Filiale oder Paketautomat.", "Ukrposhta — Filiale."] },
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
    intro: [
      "Hier steht ehrlich und konkret: welche Daten wir erhalten, wo sie liegen, wie lange wir sie aufbewahren, an wen wir sie weitergeben und wie du sie löschen lassen kannst. Kurz gesagt: Wir speichern nur, was für die Erstellung des Modells und die Ausführung der Bestellung nötig ist, verkaufen nichts, und Modelldateien werden nach 90 Tagen automatisch gelöscht.",
    ],
    sections: [
      { h: "Wer für die Daten verantwortlich ist", blocks: [
        { p: "Verantwortlicher für personenbezogene Daten ist {ownerFull} ({storeName}, {storeAddress}). Wir richten uns nach dem Gesetz der Ukraine „Über den Schutz personenbezogener Daten“; für Besucher aus der EU gilt zusätzlich die DSGVO, soweit anwendbar. Bei jeder Frage zu deinen Daten schreib uns an {email}." },
      ] },
      { h: "Welche Daten wir erheben", blocks: [
        { ul: [
          "Konto: E-Mail-Adresse und Anmelde-Kennung über Google (Firebase Authentication). Dein Passwort sehen und speichern wir nicht.",
          "Bestellung: Name, Telefonnummer, Versandart, Stadt und Filiale oder Adresse, Kommentar, geschätzter Preis, Screenshots des Modells aus dem Konstruktor.",
          "Modell: Koordinaten des ausgewählten Kartenausschnitts, gewählte Einstellungen (Größe, Stil, Gravur, Markierung „mein Zuhause“), generierte Dateien (GLB für die Vorschau, 3MF/STL für den Druck) sowie die GPX-Route, falls du eine hochgeladen hast.",
          "Technische Daten während deines Besuchs — nur mit deiner Zustimmung zu Cookies (Abschnitt „Cookies und Analyse“).",
        ] },
      ] },
      { h: "Wofür wir sie verwenden", blocks: [
        { p: "Ausschließlich um: dein Modell zu erstellen und dir eine Vorschau zu zeigen; deine Bestellung auszuführen, zu liefern und dich diesbezüglich zu kontaktieren; als Einzelunternehmer (FOP) Buchführung zu betreiben; Besucherzahlen zu ermitteln und die Website zu verbessern (in aggregierter Form). Wir verkaufen deine Daten nicht und geben sie nicht zu Werbezwecken an Dritte weiter." },
      ] },
      { h: "Wie lange wir sie aufbewahren", blocks: [
        { ul: [
          "Modelldateien und Vorschauen (GLB, 3MF/STL, Hilfsdateien) — 90 Tage ab Erstellung, danach werden sie automatisch gelöscht. Der Eintrag im Kontoverlauf bleibt bestehen, die Datei selbst ist nach dieser Frist jedoch nicht mehr verfügbar — generiere das Modell in diesem Fall erneut.",
          "Modelle, zu denen eine Bestellung vorliegt — zusammen mit der Bestellung: bis zu 3 Jahre (Aufbewahrungsfrist für buchhalterische Primärdokumente).",
          "Bestelldaten (Name, Telefon, Lieferung) — aus demselben Grund bis zu 3 Jahre.",
          "Konto und Verlauf — bis du dein Konto löschst (Schaltfläche im Konto) oder uns darum bittest.",
          "Analyse — anonymisierte Datensätze begrenzten Umfangs (das Protokoll wird rotiert), nicht länger als 12 Monate.",
          "Sicherungskopien wichtiger Daten werden 7 Tage aufbewahrt.",
        ] },
      ] },
      { h: "An wen wir sie weitergeben (Auftragsverarbeiter)", blocks: [
        { p: "Damit die Website funktioniert, verarbeitet ein Teil der Daten Dienste, mit denen wir zusammenarbeiten. Jeder erhält nur das, was er für seine Funktion benötigt:" },
        { ul: [
          "Google Firebase Authentication — Konto-Anmeldung (E-Mail, Google-Kennung).",
          "Cloudflare — Schutz der Website und Content Delivery Network; wir sehen nur den Ländercode des Besuchers, den Cloudflare der Anfrage hinzufügt.",
          "LiqPay (PrivatBank) — Online-Zahlung. Kartendaten werden auf Seiten von LiqPay eingegeben, wir erhalten sie nicht.",
          "Nova Poshta / Ukrposhta — Lieferung: Name, Telefon, Filiale oder Adresse.",
          "Telegram — unser interner Nachrichtenkanal: Deine Bestellkarte (Name, Telefon, Lieferung, Screenshots) geht an den privaten Chat des Teams. Dritte haben keinen Zugriff.",
          "OpenStreetMap und Nominatim — Karte und Ortssuche: Dorthin gehen nur der Suchtext und Koordinaten, ohne deine Kontaktdaten.",
        ] },
      ] },
      { h: "Wo die Daten gespeichert werden", blocks: [
        { p: "Modelldateien, Bestellungen und Konten werden auf einem Server unter unserer Kontrolle in der Ukraine gespeichert; der Zugriff darauf erfolgt über Cloudflare. Zugriff auf Bestelldaten hat nur der Inhaber." },
      ] },
      { h: "Cookies und Analyse", blocks: [
        { p: "Ohne deine Zustimmung setzt die Website nur technische Cookies: Konto-Anmeldung, gewählte Sprache und die Aufzeichnung deiner Cookie-Entscheidung selbst. Nach dem Klick auf „Zustimmen“ im Banner werden aktiviert:" },
        { ul: [
          "Eigene Analyse auf unserem Server: Seitenaufrufe, Klicks und Schritte im Konstruktor (welches Szenario, welche Größe, welcher Ort gewählt wurde). Die IP-Adresse wird nicht gespeichert — nur ein Tages-Hash und der Ländercode.",
          "Google Analytics 4 und Google Ads (Conversion-Messung) sowie Meta Pixel — die Standard-Cookies dieser Dienste gemäß deren eigenen Richtlinien. Sie arbeiten im Consent Mode und werden nicht aktiviert, wenn du ablehnst.",
        ] },
        { p: "Du kannst deine Wahl jederzeit über die Schaltfläche „Cookie-Einstellungen“ in der Fußzeile der Website ändern." },
      ] },
      { h: "Der Link „In 3D teilen“", blocks: [
        { p: "Wenn du auf „In 3D teilen“ klickst, wird eine Seite mit deinem Modell erstellt, die für jeden zugänglich ist, der den Link hat. Sie enthält keine deiner persönlichen Daten — nur das 3D-Modell. Der Link funktioniert, solange die Modelldatei gespeichert ist (90 Tage)." },
      ] },
      { h: "Hochgeladene Routen (GPX) und Geodaten", blocks: [
        { p: "Wenn du eine GPX-Datei hochlädst (zum Beispiel den Export deiner eigenen Aktivität aus Strava oder einer anderen App), verarbeiten wir die Koordinaten der Route ausschließlich zur Erstellung deines 3D-Modells. Das sind deine eigenen Daten — wir veröffentlichen sie nicht, geben sie nicht an Dritte weiter und verwenden sie nicht für Werbung. Die Punkte der Route werden ausgedünnt und mit denselben Fristen gespeichert wie die Modelldateien." },
      ] },
      { h: "Deine Rechte", blocks: [
        { p: "Du hast das Recht zu erfahren, welche Daten wir über dich haben, sie zu berichtigen oder löschen zu lassen. In deinem Konto gibt es die Schaltfläche „Konto und alle Daten löschen“ — sie löscht Konto, Verlauf und Modelldateien sofort. Bestelldaten bleiben für die buchhalterische Aufbewahrungsfrist bestehen (bis zu 3 Jahre). Du kannst jede Anfrage auch an {email} senden — wir antworten innerhalb von 30 Tagen." },
      ] },
      { h: "Alter", blocks: [
        { p: "Der Dienst richtet sich an Erwachsene. Bestellungen dürfen nur von Personen ab 18 Jahren aufgegeben werden." },
      ] },
      { h: "Änderungen dieser Richtlinie", blocks: [
        { p: "Wenn wir unsere Praxis im Umgang mit Daten ändern, aktualisieren wir dieses Dokument und das Datum „Aktualisiert“ auf dieser Seite." },
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
      { h: "Speicherung von Modellen", blocks: [
        { p: "Generierte Dateien werden 90 Tage aufbewahrt und danach automatisch gelöscht (Modelle, zu denen eine Bestellung vorliegt, werden zusammen mit der Bestellung gelöscht). Der Eintrag im Kontoverlauf bleibt bestehen; das Modell kann erneut generiert werden. Der Link „In 3D teilen“ ist für jeden zugänglich, der ihn hat, und funktioniert, solange die Datei gespeichert ist. Du kannst dein Konto jederzeit zusammen mit allen Modellen im Konto löschen. Details — in der [privacy:Datenschutzerklärung]." },
      ] },
      { h: "Daten und Urheberrecht", blocks: [
        { p: "Kartendaten © OpenStreetMap contributors (ODbL). Die generierten Dateien darfst du für den persönlichen Druck verwenden. Der Weiterverkauf des Dienstes oder eine massenhafte kommerzielle Nutzung bedarf einer gesonderten Vereinbarung." },
      ] },
      { h: "Nutzungsregeln", blocks: [
        { ul: [
          "Lade nur GPX-Routen hoch, an denen du die Rechte hast.",
          "Verwende keine automatisierten Mittel zur massenhaften Modellerstellung und überlaste den Dienst nicht — die Generierung läuft auf unserer eigenen Ausrüstung, und wir können den Zugriff bei Missbrauch vorübergehend einschränken.",
          "Die Gravur auf dem Modell darf keinen beleidigenden oder rechtswidrigen Inhalt enthalten; wir können den Druck einer solchen Bestellung ablehnen und den vollen Betrag erstatten.",
        ] },
      ] },
      { h: "Bestellung und Zahlung", blocks: [
        { p: "Die Bestellung erfolgt über die Website; die Zahlung — online über LiqPay oder nach Absprache. Details zu Preisen und Lieferung findest du auf der Seite [delivery:„Zahlung und Lieferung“], die vollständigen Bedingungen im [offer:Vertrag des öffentlichen Angebots]." },
      ] },
      { h: "Haftung", blocks: [
        { p: "Der Dienst wird „wie besehen“ bereitgestellt. Wir streben höchste Genauigkeit der Modelle an, garantieren jedoch aufgrund der Beschränkungen der OSM-Ausgangsdaten keine vollständige Übereinstimmung mit realen Objekten. Die 3D-Vorschau basiert auf denselben Daten wie die Druckdatei." },
      ] },
      { h: "Kontakt", blocks: [
        { p: "Fragen: {email}." },
      ] },
    ],
  },
};
