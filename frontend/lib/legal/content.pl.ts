import type { LegalSet } from "./content";

export const pl: LegalSet = {
  offer: {
    title: "Umowa oferty publicznej",
    intro: [
      "Niniejszy dokument stanowi oficjalną publiczną propozycję (ofertę) {ownerFull} (zwanego dalej „Sprzedawcą”) zawarcia umowy sprzedaży towarów i świadczenia usług na odległość na warunkach określonych poniżej, zgodnie z art. 633, 641, 642 Kodeksu cywilnego Ukrainy oraz ustawami Ukrainy „O handlu elektronicznym” i „O ochronie praw konsumentów”.",
      "Składając i/lub opłacając zamówienie w serwisie {domain} (zwanym dalej „Serwisem”), Kupujący potwierdza, że w pełni zapoznał się z warunkami niniejszej Umowy, zrozumiał je i bezwarunkowo zaakceptował (akceptacja oferty). Akceptacja niniejszej oferty jest równoznaczna z zawarciem umowy w formie pisemnej.",
    ],
    sections: [
      { h: "1. Terminy i definicje", blocks: [
        { kv: [
          { k: "Sprzedawca", v: "{ownerFull}, numer podatkowy (RNOKPP) {taxId}, płatnik podatku jednolitego." },
          { k: "Kupujący", v: "każda posiadająca zdolność do czynności prawnych osoba fizyczna lub prawna, która złożyła zamówienie w Serwisie i zaakceptowała niniejszą ofertę." },
          { k: "Serwis", v: "sklep internetowy {storeName} pod adresem {domain}, wraz z internetowym konfiguratorem modeli 3D." },
          { k: "Konfigurator", v: "usługa programowa Serwisu, w której Kupujący samodzielnie wybiera parametry przyszłego wyrobu: fragment mapy, kształt, rozmiar, format, tekst graweru itp." },
          { k: "Towar", v: "wyrób fizyczny (drukowana w 3D mapa miasta, panel, brelok-mapa, magnes itp.) z biotworzywa Eco PLA, wytwarzany na indywidualne zamówienie Kupującego." },
          { k: "Treść cyfrowa", v: "wygenerowany cyfrowy model 3D (plik w formacie 3MF/STL), nadający się do samodzielnego druku 3D." },
          { k: "Zamówienie", v: "złożone za pośrednictwem Serwisu zgłoszenie Kupującego dotyczące nabycia Towaru i/lub Treści cyfrowej." },
        ] },
      ] },
      { h: "2. Przedmiot umowy", blocks: [
        { p: "Sprzedawca zobowiązuje się, na zlecenie Kupującego, wygenerować cyfrowy model 3D i/lub wytworzyć wyrób fizyczny według indywidualnych parametrów Kupującego i przekazać go Kupującemu, a Kupujący zobowiązuje się przyjąć i opłacić Towar/Treść cyfrową na warunkach niniejszej Umowy." },
        { p: "Każdy wyrób jest wytwarzany indywidualnie według parametrów (miejsce na mapie, kształt, rozmiar, tekst, kolory), które Kupujący samodzielnie wybiera w Konfiguratorze. W związku z tym Towar stanowi produkt wykonany na indywidualne zamówienie Kupującego w rozumieniu ustawy Ukrainy „O ochronie praw konsumentów”." },
        { p: "Asortyment, właściwości i orientacyjne ceny Towarów podano w Serwisie. Zdjęcia i podglądy 3D mają charakter informacyjny: gotowy wyrób może wykazywać niewielkie różnice w odcieniu materiału i fakturze warstw, co jest naturalną cechą technologii druku 3D i nie stanowi wady Towaru." },
      ] },
      { h: "3. Procedura składania zamówienia", blocks: [
        { ul: [
          "Kupujący samodzielnie tworzy model w Konfiguratorze: wybiera miasto/fragment mapy, kształt, rozmiar, format (przestrzenny/płaski, magnes), a w razie potrzeby — tekst graweru lub trasę GPX.",
          "Przed złożeniem zamówienia Kupujący widzi podgląd 3D modelu oraz ostateczną wartość zamówienia.",
          "W celu złożenia zamówienia Kupujący podaje dane kontaktowe: imię, telefon, w razie potrzeby e-mail, sposób dostawy oraz adres/oddział przewoźnika.",
          "Zamówienie uważa się za przyjęte do realizacji po jego opłaceniu online lub po potwierdzeniu przez operatora (zgodnie z ustaleniami).",
        ] },
        { p: "Kupujący samodzielnie odpowiada za prawidłowość wybranych parametrów modelu (fragment mapy, tekst, rozmiar) oraz za poprawność danych kontaktowych. Sprzedawca nie ponosi odpowiedzialności za skutki błędów w danych podanych przez Kupującego, w szczególności za wytworzenie wyrobu z błędnym tekstem lub fragmentem mapy, które Kupujący sam zatwierdził w Konfiguratorze." },
      ] },
      { h: "4. Cena i płatność", blocks: [
        { p: "Ceny w Serwisie podane są w hrywnach (dla zamówień na terenie Ukrainy) oraz w euro (orientacyjnie). Ostateczną wartość zamówienia Kupujący widzi na etapie składania zamówienia, przed dokonaniem płatności. Koszt dostawy nie jest wliczony w cenę Towaru i jest opłacany osobno według taryf przewoźnika." },
        { p: "Płatność odbywa się online za pośrednictwem serwisu płatniczego LiqPay (JSC CB „PrivatBank”): kartą bankową Visa/Mastercard oraz innymi metodami dostępnymi w LiqPay. Dane karty płatniczej są przetwarzane po stronie systemu płatności; Sprzedawca ich nie otrzymuje i nie przechowuje." },
        { p: "Po uzgodnieniu z operatorem możliwa jest płatność w inny uzgodniony sposób. Pobranie gotowej Treści cyfrowej w ramach bezpłatnego limitu konta jest bezpłatne." },
        { p: "Towar jest opłacany w całości przed przekazaniem do produkcji, chyba że strony uzgodniły inaczej. Szczegóły — na stronie [delivery:„Płatność i dostawa”]." },
      ] },
      { h: "5. Terminy realizacji", blocks: [
        { p: "Treść cyfrowa jest generowana automatycznie i udostępniana na koncie Kupującego / e-mailem bezpośrednio po wygenerowaniu lub po potwierdzeniu zamówienia." },
        { p: "Orientacyjny czas wytworzenia wyrobu fizycznego wynosi 1–3 dni robocze od momentu opłacenia/potwierdzenia zamówienia. W przypadku zwiększonego obciążenia lub technicznej złożoności wyrobu termin może ulec wydłużeniu, o czym Sprzedawca informuje Kupującego." },
      ] },
      { h: "6. Dostawa", blocks: [
        { p: "Dostawa na terenie Ukrainy realizowana jest przez firmy „Nova Poshta” (oddział, paczkomat) lub „Ukrposhta”. Orientacyjny czas dostawy na terenie Ukrainy to 2–4 dni robocze od nadania." },
        { p: "Koszt dostawy jest obliczany według taryf przewoźnika i opłacany przez Kupującego osobno (zazwyczaj przy odbiorze). Prawo własności do Towaru oraz ryzyko przypadkowego uszkodzenia przechodzą na Kupującego z chwilą odbioru Towaru od przewoźnika." },
        { p: "Przy odbiorze Kupujący jest zobowiązany sprawdzić przesyłkę pod kątem nienaruszalności opakowania i wyrobu. W przypadku uszkodzenia w transporcie należy sporządzić protokół przewoźnika i powiadomić Sprzedawcę — taki przypadek jest rozwiązywany przez bezpłatny ponowny wydruk lub zwrot pieniędzy (zob. rozdział 8)." },
      ] },
      { h: "7. Jakość i gwarancja", blocks: [
        { p: "Sprzedawca gwarantuje zgodność wyrobu z parametrami zatwierdzonymi przez Kupującego w Konfiguratorze oraz należytą jakość druku. Na wyroby fizyczne udzielana jest 60-dniowa gwarancja od momentu odbioru, obejmująca wady druku i rozwarstwienia powstałe nie z winy Kupującego." },
        { p: "Naturalne cechy technologii druku FDM (widoczne warstwy druku, niewielkie różnice odcienia tworzywa między partiami, ślady technologiczne na dolnej powierzchni) nie stanowią wad Towaru." },
        { p: "Sprzedawca nie odpowiada za rezultat druku na sprzęcie Kupującego przy samodzielnym druku pobranej Treści cyfrowej (jakość takiego druku zależy od drukarki, materiału i ustawień Kupującego)." },
      ] },
      { h: "8. Zwrot pieniędzy i wymiana", blocks: [
        { p: "Ponieważ Towar jest wytwarzany na indywidualne zamówienie według unikalnych parametrów Kupującego, Towar należytej jakości nie podlega zwrotowi ani wymianie (ustawa Ukrainy „O ochronie praw konsumentów”; wykaz towarów zatwierdzony uchwałą Gabinetu Ministrów Ukrainy nr 172 z dnia 19.03.1994 r.). Treść cyfrowa po udostępnieniu do pobrania nie podlega zwrotowi jako treść elektroniczna, której usługa została w pełni wykonana." },
        { ul: [
          "Do momentu przekazania zamówienia do produkcji Kupujący może je anulować i otrzymać pełny zwrot pieniędzy.",
          "W przypadku wady, uszkodzenia podczas dostawy lub niezgodności wyrobu z zatwierdzonymi parametrami — Sprzedawca, według wyboru Kupującego, bezpłatnie wytwarza i wysyła nowy wyrób lub zwraca pełną wartość.",
          "Jeżeli Treść cyfrowa jest technicznie uszkodzona lub nie daje się pobrać z winy Sprzedawcy — plik jest bezpłatnie generowany ponownie lub pieniądze są zwracane.",
        ] },
        { p: "Procedura zgłoszenia i terminy zostały opisane w dokumencie [refund:„Zwrot i wymiana środków”], który stanowi integralną część niniejszej Umowy. Zwrot pieniędzy następuje tą samą metodą, którą dokonano płatności (na kartę za pośrednictwem LiqPay), w terminach przewidzianych regulaminem systemu płatności i banku." },
      ] },
      { h: "9. Prawa i obowiązki stron", blocks: [
        { p: "Sprzedawca jest zobowiązany: wytworzyć Towar zgodnie z parametrami zatwierdzonymi przez Kupującego; dotrzymywać deklarowanych terminów; informować Kupującego o statusie zamówienia; zapewnić poufność danych osobowych Kupującego." },
        { p: "Sprzedawca ma prawo: angażować osoby trzecie do wykonania swoich zobowiązań (przewoźnicy, serwisy płatnicze); wstrzymać realizację zamówienia w przypadku braku zapłaty; odmówić wytworzenia wyrobu, którego treść narusza ustawodawstwo Ukrainy (w szczególności zawiera zakazaną symbolikę lub mowę nienawiści), z pełnym zwrotem pieniędzy." },
        { p: "Kupujący jest zobowiązany: podać prawdziwe dane niezbędne do realizacji zamówienia; opłacić zamówienie; odebrać Towar od przewoźnika w terminie przechowywania przesyłki." },
        { p: "Kupujący ma prawo: otrzymać Towar należytej jakości w deklarowanym terminie; otrzymywać informacje o statusie swojego zamówienia; złożyć reklamację w trybie przewidzianym niniejszą Umową." },
      ] },
      { h: "10. Prawa własności intelektualnej", blocks: [
        { p: "Dane kartograficzne pochodzą z OpenStreetMap (© OpenStreetMap contributors, licencja ODbL); dane wysokościowe — ze źródeł otwartych. Wygenerowany model 3D jest udostępniany Kupującemu do osobistego, niekomercyjnego użytku i druku. Masowe komercyjne wykorzystanie lub odsprzedaż modeli wymaga odrębnego pisemnego porozumienia ze Sprzedawcą." },
        { p: "Przesłana przez Kupującego trasa GPX stanowi własne dane Kupującego i jest przetwarzana wyłącznie w celu zbudowania jego modelu (zob. [privacy:Polityka prywatności]). Kupujący gwarantuje, że zamówiony przez niego tekst graweru nie narusza praw osób trzecich." },
      ] },
      { h: "11. Dane osobowe", blocks: [
        { p: "Składając zamówienie, Kupujący wyraża zgodę na przetwarzanie swoich danych osobowych (imię, dane kontaktowe, adres dostawy) wyłącznie w celu wykonania niniejszej Umowy, zgodnie z ustawą Ukrainy „O ochronie danych osobowych” oraz [privacy:Polityką prywatności]. Dane nie są przekazywane osobom trzecim, z wyjątkiem przypadków niezbędnych do realizacji zamówienia (przewoźnik, serwis płatniczy)." },
      ] },
      { h: "12. Odpowiedzialność i siła wyższa", blocks: [
        { p: "Za niewykonanie lub nienależyte wykonanie zobowiązań strony ponoszą odpowiedzialność zgodnie z obowiązującym ustawodawstwem Ukrainy. Łączna odpowiedzialność Sprzedawcy z tytułu jakichkolwiek roszczeń jest ograniczona do kwoty zamówienia faktycznie zapłaconej przez Kupującego." },
        { p: "Strony są zwolnione z odpowiedzialności za całkowite lub częściowe niewykonanie zobowiązań, jeżeli było ono skutkiem okoliczności siły wyższej: działań wojennych, ostrzałów, przerw w dostawie energii elektrycznej, klęsk żywiołowych, decyzji organów władzy, zakłóceń w pracy przewoźników itp. Terminy wykonania zobowiązań ulegają przedłużeniu o czas trwania takich okoliczności." },
      ] },
      { h: "13. Reklamacje i rozstrzyganie sporów", blocks: [
        { p: "Reklamacje dotyczące zamówienia przyjmowane są pod adresem {email} lub telefonicznie pod numerem {phone}, z podaniem numeru zamówienia. Sprzedawca rozpatruje zgłoszenia w ciągu 1–3 dni roboczych. Spory rozstrzygane są w drodze negocjacji, a w przypadku braku porozumienia — w trybie określonym obowiązującym ustawodawstwem Ukrainy." },
      ] },
      { h: "14. Okres obowiązywania i zmiana warunków", blocks: [
        { p: "Umowa wchodzi w życie z chwilą akceptacji oferty przez Kupującego i obowiązuje do pełnego wykonania zobowiązań przez strony. Sprzedawca ma prawo zmieniać warunki niniejszej oferty, publikując nową wersję na tej stronie; nowa wersja ma zastosowanie do zamówień złożonych po jej opublikowaniu. Aktualna wersja jest stale dostępna pod adresem {domain}/offer." },
        { p: "Integralną częścią niniejszej Umowy są dokumenty: [refund:„Zwrot i wymiana środków”], [delivery:„Płatność i dostawa”], [privacy:„Polityka prywatności”] oraz [terms:„Warunki korzystania”]." },
      ] },
      { h: "15. Dane Sprzedawcy", blocks: [
        { kv: [
          { k: "Sprzedawca", v: "{ownerFull}" },
          { k: "Numer podatkowy (RNOKPP)", v: "{taxId}" },
          { k: "Rodzaj działalności (KVED)", v: "{ved}" },
          { k: "IBAN", v: "{iban}" },
          { k: "Sklep", v: "{storeName}, {storeAddress}" },
          { k: "E-mail", v: "{email}" },
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
