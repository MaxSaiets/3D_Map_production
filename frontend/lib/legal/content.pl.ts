import type { LegalSet } from "./content";

export const pl: LegalSet = {
  offer: {
    title: "Umowa oferty publicznej",
    intro: [
      "Niniejszy dokument jest oficjalną ofertą publiczną {ownerFull} (dalej — „Sprzedawca”) zawarcia umowy sprzedaży towarów i świadczenia usług na warunkach przedstawionych poniżej. Opłacając zamówienie na stronie {domain} (dalej — „Strona”), Kupujący potwierdza, że w pełni przeczytał, zrozumiał i bezwarunkowo zaakceptował warunki niniejszej Umowy (akceptacja oferty zgodnie z art. 633, 641, 642 Kodeksu cywilnego Ukrainy).",
    ],
    sections: [
      { h: "1. Terminy", blocks: [
        { p: "Sprzedawca — {ownerShort}, numer podatkowy {taxId}." },
        { p: "Kupujący — każda osoba zdolna do czynności prawnych, która złożyła zamówienie na Stronie." },
        { p: "Towar / Usługa — cyfrowy model 3D (plik 3MF/STL) mapy miasta lub breloka-mapy oraz/lub wytworzenie (druk 3D) fizycznego wyrobu z bioplastiku Eco PLA na indywidualne zamówienie Kupującego." },
      ] },
      { h: "2. Przedmiot umowy", blocks: [
        { p: "Sprzedawca zobowiązuje się dostarczyć Kupującemu cyfrowy model 3D oraz/lub wytworzyć i przekazać fizyczny wyrób na zamówienie, a Kupujący — odebrać je i opłacić zgodnie z warunkami niniejszej Umowy." },
        { p: "Każdy wyrób jest wytwarzany indywidualnie według parametrów (miejsce na mapie, kształt, rozmiar, tekst), które Kupujący wybiera samodzielnie w kreatorze na Stronie, czyli jest towarem wytworzonym na zamówienie." },
      ] },
      { h: "3. Składanie zamówienia", blocks: [
        { p: "Kupujący tworzy zamówienie w kreatorze na Stronie i podaje dane kontaktowe (imię, telefon, sposób i adres dostawy). Zamówienie uznaje się za przyjęte po jego opłaceniu lub potwierdzeniu przez operatora." },
        { p: "Kupujący odpowiada za prawdziwość podanych danych. Sprzedawca nie ponosi odpowiedzialności za skutki błędów w danych podanych przez Kupującego." },
      ] },
      { h: "4. Cena i płatność", blocks: [
        { p: "Ceny towarów i usług podane są na Stronie w hrywnach (dla zamówień na terenie Ukrainy) oraz w euro (dla dostawy do UE) i mają charakter orientacyjny do momentu potwierdzenia zamówienia. Ostateczną cenę Kupujący widzi na etapie składania zamówienia." },
        { p: "Płatność odbywa się online za pośrednictwem serwisu płatniczego LiqPay (kartą bankową Visa/Mastercard) lub innym uzgodnionym sposobem. Pobranie gotowego pliku cyfrowego w ramach bezpłatnego limitu jest darmowe." },
        { p: "Szczegóły — na stronie [delivery:„Płatność i dostawa”]." },
      ] },
      { h: "5. Wytworzenie i dostawa", blocks: [
        { p: "Plik cyfrowy jest udostępniany w panelu / na adres e-mail od razu lub po potwierdzeniu zamówienia. Fizyczny wyrób jest wytwarzany i wysyłany w terminie wskazanym na stronie „Płatność i dostawa”, za pośrednictwem firm Nova Poshta lub Ukrposhta (Ukraina), Nova Post EU lub Meest (UE)." },
      ] },
      { h: "6. Zwrot środków", blocks: [
        { p: "Ponieważ towary są wytwarzane na indywidualne zamówienie, a pliki cyfrowe mają charakter treści elektronicznych, zwroty reguluje odrębny dokument — [refund:„Zwrot i wymiana środków”], który jest nieodłączną częścią niniejszej Umowy." },
      ] },
      { h: "7. Prawa własności intelektualnej", blocks: [
        { p: "Dane kartograficzne pochodzą z OpenStreetMap (ODbL), dane wysokościowe — ze źródeł otwartych. Wygenerowany model 3D jest udostępniany Kupującemu do osobistego, niekomercyjnego użytku i druku. Wgrana przez Kupującego trasa GPX jest jego własnymi danymi i jest przetwarzana wyłącznie w celu zbudowania modelu (zob. [privacy:Politykę prywatności])." },
      ] },
      { h: "8. Odpowiedzialność stron", blocks: [
        { p: "Sprzedawca nie odpowiada za jakość druku na sprzęcie Kupującego przy samodzielnym druku pobranego pliku. Łączna odpowiedzialność Sprzedawcy jest ograniczona do kwoty opłaconego zamówienia." },
        { p: "Strony są zwolnione z odpowiedzialności za niewykonanie zobowiązań wskutek okoliczności siły wyższej (force majeure)." },
      ] },
      { h: "9. Dane osobowe", blocks: [
        { p: "Składając zamówienie, Kupujący wyraża zgodę na przetwarzanie swoich danych osobowych w celu realizacji zamówienia zgodnie z ustawą Ukrainy o ochronie danych osobowych oraz [privacy:Polityką prywatności]." },
      ] },
      { h: "10. Rozstrzyganie sporów i okres obowiązywania", blocks: [
        { p: "Spory rozstrzygane są w drodze negocjacji, a w razie nieosiągnięcia porozumienia — zgodnie z obowiązującym prawem Ukrainy. Umowa obowiązuje od chwili akceptacji do pełnego wykonania zobowiązań przez strony. Sprzedawca ma prawo zmieniać warunki, publikując nową wersję na tej stronie." },
      ] },
      { h: "11. Dane Sprzedawcy", blocks: [
        { kv: [
          { k: "Sprzedawca", v: "{ownerFull}" },
          { k: "Numer podatkowy (IPN/RNOKPP)", v: "{taxId}" },
          { k: "IBAN", v: "{iban}" },
          { k: "Adres rejestracji", v: "{ownerRegAddress}" },
          { k: "Sklep", v: "{storeName}, {storeAddress}" },
          { k: "Email", v: "{email}" },
          { k: "Telefon", v: "{phone}" },
        ] },
      ] },
    ],
  },

  refund: {
    title: "Zwrot i wymiana środków",
    sections: [
      { h: "Charakter towaru", blocks: [
        { p: "Wszystkie wyroby {storeName} są wytwarzane indywidualnie według parametrów, które Kupujący wybiera sam (miejsce na mapie, kształt, rozmiar, tekst), a cyfrowe modele 3D są treścią elektroniczną. Wpływa to na warunki zwrotu zgodnie z ustawą Ukrainy o ochronie praw konsumentów (towar należytej jakości, wytworzony na indywidualne zamówienie, nie podlega zwrotowi ani wymianie)." },
      ] },
      { h: "Pliki cyfrowe (pobieranie 3MF/STL)", blocks: [
        { p: "Środki za plik cyfrowy nie podlegają zwrotowi po tym, jak plik został wygenerowany i udostępniony do pobrania, ponieważ usługa została już wykonana w pełnym zakresie. Jeśli plik technicznie się nie pobiera lub jest uszkodzony z naszej strony — bezpłatnie wygenerujemy go ponownie lub zwrócimy środki." },
      ] },
      { h: "Druk na zamówienie (fizyczny wyrób)", blocks: [
        { ul: [
          "Przed rozpoczęciem produkcji — możesz anulować zamówienie i otrzymać pełny zwrot środków, jeśli jeszcze nie rozpoczęliśmy druku.",
          "Po rozpoczęciu produkcji — środki nie podlegają zwrotowi, ponieważ wyrób jest wytwarzany personalnie pod Twoje zamówienie.",
          "Wada, uszkodzenie podczas dostawy lub niezgodność z zamówieniem — bezpłatnie wydrukujemy ponownie i wyślemy wyrób lub zwrócimy pełną wartość (według Twojego wyboru).",
        ] },
      ] },
      { h: "Jak złożyć zwrot", blocks: [
        { p: "Napisz na {email} lub zadzwoń pod {phone}, podając numer zamówienia, przyczynę oraz (w przypadku wady) zdjęcie wyrobu. Rozpatrzymy zgłoszenie w ciągu 1–3 dni roboczych." },
        { p: "Zwrot środków odbywa się tym samym sposobem, którym dokonano płatności (zwrot na kartę przez LiqPay), w terminach przewidzianych regulaminem systemu płatniczego i banku." },
      ] },
      { h: "Kontakt", blocks: [
        { p: "Dokument jest nieodłączną częścią [offer:Umowy oferty publicznej]. Pytania: {email}." },
      ] },
    ],
  },

  delivery: {
    title: "Płatność i dostawa",
    sections: [
      { h: "Towary i ceny", blocks: [
        { p: "Ceny wyrobów (dostawa płatna osobno wg taryfy przewoźnika):" },
        { ul: [
          "Brelok-mapa — od 120 ₴ (≈ 3 €).",
          "Mapa 3D dzielnicy: S 5,5 cm — 250 ₴, M 8 cm — 350 ₴, L 11 cm — 450 ₴, XL 15 cm — 550 ₴ (mapy — od 6 €).",
          "Magnes na lodówkę (mapa) — 150 ₴.",
          "Pobranie gotowego pliku 3MF/STL do samodzielnego druku — bezpłatnie w ramach limitu konta.",
        ] },
      ] },
      { h: "Płatność", blocks: [
        { p: "Płatność online kartą bankową Visa / Mastercard za pośrednictwem bezpiecznego serwisu LiqPay. Dane karty są przetwarzane po stronie systemu płatniczego — nie przechowujemy ich. Możliwa jest również płatność po uzgodnieniu z operatorem. Plik cyfrowy w ramach bezpłatnego limitu jest udostępniany bez opłaty." },
      ] },
      { h: "Wytworzenie", blocks: [
        { p: "Wyroby są drukowane na zamówienie z bioplastiku Eco PLA. Orientacyjny czas wytworzenia — 1–3 dni robocze po potwierdzeniu zamówienia (w zależności od obciążenia i złożoności)." },
      ] },
      { h: "Dostawa", blocks: [
        { p: "Ukraina:" },
        { ul: ["Nova Poshta — oddział lub paczkomat.", "Ukrposhta — oddział."] },
        { p: "Unia Europejska (15 krajów):" },
        { ul: ["Nova Post EU.", "Meest."] },
        { p: "Koszt dostawy jest obliczany według taryf przewoźnika i opłacany osobno (zazwyczaj przy odbiorze). Orientacyjny czas dostawy na terenie Ukrainy — 2–4 dni robocze po wysłaniu." },
      ] },
      { h: "Zwrot", blocks: [
        { p: "Warunki zwrotu środków opisano na stronie [refund:„Zwrot i wymiana środków”]. Warunki ogólne — w [offer:Umowie oferty publicznej]." },
      ] },
      { h: "Kontakt", blocks: [
        { p: "Pytania dotyczące płatności lub dostawy: {email}, {phone}." },
      ] },
    ],
  },

  contacts: {
    title: "Kontakt i dane",
    sections: [
      { h: "Skontaktuj się z nami", blocks: [
        { kv: [
          { k: "Email", v: "{email}" },
          { k: "Telefon", v: "{phone}" },
          { k: "Strona", v: "{domain}" },
        ] },
        { p: "Godziny obsługi zamówień: codziennie, odpowiadamy w ciągu doby." },
      ] },
      { h: "Sklep", blocks: [
        { kv: [
          { k: "Nazwa", v: "{storeName}" },
          { k: "Adres sklepu", v: "{storeAddress}" },
        ] },
      ] },
      { h: "Sprzedawca (FOP)", blocks: [
        { kv: [
          { k: "Nazwa", v: "{ownerFull}" },
          { k: "Numer podatkowy (IPN/RNOKPP)", v: "{taxId}" },
          { k: "Adres rejestracji", v: "{ownerRegAddress}" },
          { k: "Działalność (KVED)", v: "{ved}" },
          { k: "IBAN", v: "{iban}" },
        ] },
      ] },
      { h: "Co sprzedajemy", blocks: [
        { p: "{storeName} — to modele 3D map miast i breloków-map na zamówienie. Możesz pobrać gotowy plik do druku (3MF/STL) w ramach bezpłatnego limitu lub zamówić druk wyrobu z bioplastiku Eco PLA z dostawą. Ceny orientacyjne są wskazane w kreatorze oraz na stronie [delivery:„Płatność i dostawa”]." },
      ] },
      { h: "Dokumenty", blocks: [
        { ul: [
          "[offer:Umowa oferty publicznej]",
          "[refund:Zwrot i wymiana środków]",
          "[delivery:Płatność i dostawa]",
          "[privacy:Polityka prywatności]",
          "[terms:Warunki korzystania]",
        ] },
      ] },
    ],
  },

  privacy: {
    title: "Polityka prywatności",
    sections: [
      { h: "Jakie dane zbieramy", blocks: [
        { p: "Imię, email lub numer telefonu (przy logowaniu/rejestracji), a także dane kontaktowe i adres dostawy, które podajesz przy składaniu zamówienia. Dane techniczne (historia wygenerowanych modeli) są przechowywane w Twoim panelu." },
      ] },
      { h: "Jak wykorzystujemy dane", blocks: [
        { p: "Wyłącznie w celu świadczenia usługi: logowanie do konta, przechowywanie historii modeli, obsługa i dostawa zamówień, kontakt z Tobą w sprawie zamówienia. Nie sprzedajemy ani nie przekazujemy Twoich danych osobom trzecim w celach reklamowych." },
      ] },
      { h: "Wgrane trasy (GPX) i dane geolokalizacyjne", blocks: [
        { p: "Jeśli wgrasz plik GPX (na przykład eksport własnej aktywności ze Strava lub innej aplikacji), przetwarzamy współrzędne trasy wyłącznie w celu zbudowania Twojego modelu 3D. Są to Twoje własne dane — nie publikujemy ich, nie przekazujemy osobom trzecim ani nie wykorzystujemy do reklamy. Punkty trasy są upraszczane i przechowywane dokładnie tak długo, jak jest to potrzebne do wygenerowania modelu oraz (jeśli jesteś zalogowany) do prowadzenia historii modeli w panelu; możesz w każdej chwili poprosić o ich usunięcie." },
        { p: "Wyszukiwanie miejsca na mapie wysyła Twoje zapytanie do serwisu geokodowania OpenStreetMap (Nominatim), a same mapy są ładowane z kafelków OpenStreetMap — zgodnie z ich warunkami korzystania. Nie przekazujemy tym serwisom Twojego imienia ani danych kontaktowych." },
      ] },
      { h: "Przechowywanie i serwisy", blocks: [
        { p: "Autoryzacja działa za pośrednictwem Google Firebase Authentication. Strona jest chroniona przez Cloudflare. Zamówienia są obsługiwane ręcznie. Dane są przechowywane na zabezpieczonym serwerze dokładnie tak długo, jak jest to potrzebne do realizacji zamówienia i prowadzenia historii." },
      ] },
      { h: "Cookie i analityka", blocks: [
        { p: "Korzystamy z prywatnej analityki Cloudflare bez zewnętrznych plików cookie reklamowych. Pliki cookie są wykorzystywane wyłącznie do działania logowania do konta." },
      ] },
      { h: "Twoje prawa", blocks: [
        { p: "Możesz poprosić o usunięcie swojego konta i powiązanych danych. Napisz do nas na {email}." },
      ] },
      { h: "Kontakt", blocks: [
        { p: "W sprawach dotyczących prywatności: {email}." },
      ] },
    ],
  },

  terms: {
    title: "Warunki korzystania",
    sections: [
      { h: "O serwisie", blocks: [
        { p: "{storeName} pozwala stworzyć model 3D fragmentu miasta lub breloka z mapą na podstawie otwartych danych OpenStreetMap i pobrać gotowy plik do druku 3D (3MF/STL) lub zamówić druk." },
      ] },
      { h: "Konto i bezpłatne pobrania", blocks: [
        { p: "Do pobrania pełnego modelu wymagane jest konto. Każdy użytkownik ma dostęp do 5 bezpłatnych pobrań. Dalej — po uzgodnieniu (druk/płatność), kontakt przez stronę." },
      ] },
      { h: "Dane i prawa autorskie", blocks: [
        { p: "Dane kartograficzne © OpenStreetMap contributors (ODbL). Wygenerowane pliki możesz wykorzystywać do osobistego druku. Odsprzedaż usługi lub masowe wykorzystanie komercyjne wymaga odrębnego uzgodnienia." },
      ] },
      { h: "Zamówienia i płatność", blocks: [
        { p: "Zamówienie składa się przez stronę; płatność — online za pośrednictwem LiqPay lub po uzgodnieniu. Szczegóły cen i dostawy — na stronie [delivery:„Płatność i dostawa”], pełne warunki — w [offer:Umowie oferty publicznej]." },
      ] },
      { h: "Odpowiedzialność", blocks: [
        { p: "Usługa świadczona jest „tak jak jest”. Dążymy do maksymalnej dokładności modeli, ale nie gwarantujemy pełnej zgodności z rzeczywistymi obiektami ze względu na ograniczenia danych źródłowych OSM." },
      ] },
      { h: "Kontakt", blocks: [
        { p: "Pytania: {email}." },
      ] },
    ],
  },
};
