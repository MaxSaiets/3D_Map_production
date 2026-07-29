# -*- coding: utf-8 -*-
"""Генератор lib/worldCities.ts — міста Європи з реальними фактами.

Кожен запис: slug, координати, цільова локаль, назви 6 мовами, факти
(населення/рік/заснування/площа/річка/регіон/орієнтир). Дані — загальновідомі
довідкові (Eurostat/нацстат ~2023, засновано/перша згадка з енциклопедій).
Мета: сторінки /maps/{slug} з УНІКАЛЬНИМ контентом, а не шаблон.
"""
import io

# slug, key, lat, lon, locale, [uk,en,de,pl,fr,es], pop, popYear, founded, firstMention, area, river(uk/lat), region(uk/lat), landmark(uk/lat)
C = [
 ("warsaw","Warsaw",52.2297,21.0122,"pl",["Варшава","Warsaw","Warschau","Warszawa","Varsovie","Varsovia"],1861975,2023,1300,1,517.2,("Вісла","Vistula"),("Мазовецьке воєводство","Masovian Voivodeship"),("Палац культури і науки","Palace of Culture and Science")),
 ("krakow","Krakow",50.0647,19.9450,"pl",["Краків","Kraków","Krakau","Kraków","Cracovie","Cracovia"],804237,2023,966,1,326.8,("Вісла","Vistula"),("Малопольське воєводство","Lesser Poland"),("Вавельський замок","Wawel Castle")),
 ("wroclaw","Wroclaw",51.1079,17.0385,"pl",["Вроцлав","Wrocław","Breslau","Wrocław","Wrocław","Breslavia"],674312,2023,1000,1,292.8,("Одра","Oder"),("Нижньосілезьке воєводство","Lower Silesia"),("Ринок Вроцлава","Wrocław Market Square")),
 ("gdansk","Gdansk",54.3520,18.6466,"pl",["Ґданськ","Gdańsk","Danzig","Gdańsk","Gdańsk","Gdansk"],486492,2023,997,1,262.0,("Мотлава","Motława"),("Поморське воєводство","Pomerania"),("Журав","Gdańsk Crane")),
 ("poznan","Poznan",52.4064,16.9252,"pl",["Познань","Poznań","Posen","Poznań","Poznań","Poznan"],541316,2023,968,1,261.9,("Варта","Warta"),("Великопольське воєводство","Greater Poland"),("Познанська ратуша","Poznań Town Hall")),
 ("lodz","Lodz",51.7592,19.4560,"pl",["Лодзь","Łódź","Lodz","Łódź","Łódź","Lodz"],658444,2023,1332,1,293.3,("Лодка","Łódka"),("Лодзьке воєводство","Łódź Voivodeship"),("Вулиця Пьотрковська","Piotrkowska Street")),
 ("szczecin","Szczecin",53.4285,14.5528,"pl",["Щецин","Szczecin","Stettin","Szczecin","Szczecin","Szczecin"],391566,2023,1243,0,300.6,("Одра","Oder"),("Західнопоморське воєводство","West Pomerania"),("Замок Померанських князів","Pomeranian Dukes' Castle")),
 ("lublin","Lublin",51.2465,22.5684,"pl",["Люблін","Lublin","Lublin","Lublin","Lublin","Lublin"],331243,2023,1317,0,147.5,("Бистриця","Bystrzyca"),("Люблінське воєводство","Lublin Voivodeship"),("Люблінський замок","Lublin Castle")),
 ("berlin","Berlin",52.5200,13.4050,"de",["Берлін","Berlin","Berlin","Berlin","Berlin","Berlín"],3878100,2023,1237,1,891.7,("Шпрее","Spree"),("Земля Берлін","State of Berlin"),("Бранденбурзькі ворота","Brandenburg Gate")),
 ("munich","Munich",48.1351,11.5820,"de",["Мюнхен","Munich","München","Monachium","Munich","Múnich"],1512491,2023,1158,0,310.7,("Ізар","Isar"),("Баварія","Bavaria"),("Марієнплац","Marienplatz")),
 ("hamburg","Hamburg",53.5511,9.9937,"de",["Гамбург","Hamburg","Hamburg","Hamburg","Hambourg","Hamburgo"],1892122,2023,808,0,755.2,("Ельба","Elbe"),("Земля Гамбург","State of Hamburg"),("Ельбська філармонія","Elbphilharmonie")),
 ("cologne","Cologne",50.9375,6.9603,"de",["Кельн","Cologne","Köln","Kolonia","Cologne","Colonia"],1084831,2023,-38,0,405.0,("Рейн","Rhine"),("Північний Рейн-Вестфалія","North Rhine-Westphalia"),("Кельнський собор","Cologne Cathedral")),
 ("frankfurt","Frankfurt",50.1109,8.6821,"de",["Франкфурт","Frankfurt","Frankfurt am Main","Frankfurt","Francfort","Fráncfort"],773068,2023,794,1,248.3,("Майн","Main"),("Гессен","Hesse"),("Ремер","Römer")),
 ("stuttgart","Stuttgart",48.7758,9.1829,"de",["Штутгарт","Stuttgart","Stuttgart","Stuttgart","Stuttgart","Stuttgart"],632865,2023,950,0,207.4,("Неккар","Neckar"),("Баден-Вюртемберг","Baden-Württemberg"),("Палацова площа","Schlossplatz")),
 ("dusseldorf","Dusseldorf",51.2277,6.7735,"de",["Дюссельдорф","Düsseldorf","Düsseldorf","Düsseldorf","Düsseldorf","Düsseldorf"],629047,2023,1288,0,217.4,("Рейн","Rhine"),("Північний Рейн-Вестфалія","North Rhine-Westphalia"),("Райнтурм","Rheinturm")),
 ("leipzig","Leipzig",51.3397,12.3731,"de",["Лейпциг","Leipzig","Leipzig","Lipsk","Leipzig","Leipzig"],628718,2023,1015,1,297.8,("Вайсе-Ельстер","White Elster"),("Саксонія","Saxony"),("Пам'ятник битві народів","Monument to the Battle of the Nations")),
 ("dresden","Dresden",51.0504,13.7373,"de",["Дрезден","Dresden","Dresden","Drezno","Dresde","Dresde"],566222,2023,1206,1,328.5,("Ельба","Elbe"),("Саксонія","Saxony"),("Фрауенкірхе","Frauenkirche")),
 ("nuremberg","Nuremberg",49.4521,11.0767,"de",["Нюрнберг","Nuremberg","Nürnberg","Norymberga","Nuremberg","Núremberg"],523026,2023,1050,1,186.4,("Пегніц","Pegnitz"),("Баварія","Bavaria"),("Нюрнберзький замок","Nuremberg Castle")),
 ("vienna","Vienna",48.2082,16.3738,"de",["Відень","Vienna","Wien","Wiedeń","Vienne","Viena"],2005760,2023,1147,1,414.9,("Дунай","Danube"),("Земля Відень","State of Vienna"),("Собор Святого Стефана","St. Stephen's Cathedral")),
 ("salzburg","Salzburg",47.8095,13.0550,"de",["Зальцбург","Salzburg","Salzburg","Salzburg","Salzbourg","Salzburgo"],157479,2023,696,0,65.7,("Зальцах","Salzach"),("Земля Зальцбург","State of Salzburg"),("Фортеця Гогензальцбург","Hohensalzburg Fortress")),
 ("zurich","Zurich",47.3769,8.5417,"de",["Цюрих","Zurich","Zürich","Zurych","Zurich","Zúrich"],443037,2023,-15,0,87.9,("Ліммат","Limmat"),("Кантон Цюрих","Canton of Zurich"),("Гросмюнстер","Grossmünster")),
 ("prague","Prague",50.0755,14.4378,"de",["Прага","Prague","Prag","Praga","Prague","Praga"],1384732,2023,885,1,496.2,("Влтава","Vltava"),("Столичний край","Prague Capital"),("Карлів міст","Charles Bridge")),
 ("brno","Brno",49.1951,16.6068,"de",["Брно","Brno","Brünn","Brno","Brno","Brno"],400566,2023,1243,0,230.2,("Свратка","Svratka"),("Південноморавський край","South Moravia"),("Вілла Тугендгат","Villa Tugendhat")),
 ("bratislava","Bratislava",48.1486,17.1077,"de",["Братислава","Bratislava","Pressburg","Bratysława","Bratislava","Bratislava"],475503,2023,907,1,367.6,("Дунай","Danube"),("Братиславський край","Bratislava Region"),("Братиславський град","Bratislava Castle")),
 ("budapest","Budapest",47.4979,19.0402,"de",["Будапешт","Budapest","Budapest","Budapeszt","Budapest","Budapest"],1682308,2023,1873,0,525.2,("Дунай","Danube"),("Центральна Угорщина","Central Hungary"),("Будівля парламенту","Hungarian Parliament Building")),
 ("paris","Paris",48.8566,2.3522,"fr",["Париж","Paris","Paris","Paryż","Paris","París"],2102650,2023,-52,0,105.4,("Сена","Seine"),("Іль-де-Франс","Île-de-France"),("Ейфелева вежа","Eiffel Tower")),
 ("lyon","Lyon",45.7640,4.8357,"fr",["Ліон","Lyon","Lyon","Lyon","Lyon","Lyon"],522250,2023,-43,0,47.9,("Рона","Rhône"),("Овернь-Рона-Альпи","Auvergne-Rhône-Alpes"),("Базиліка Фурв'єр","Fourvière Basilica")),
 ("marseille","Marseille",43.2965,5.3698,"fr",["Марсель","Marseille","Marseille","Marsylia","Marseille","Marsella"],873076,2023,-600,0,240.6,("Ювон","Huveaune"),("Прованс-Альпи-Лазурний Берег","Provence-Alpes-Côte d'Azur"),("Нотр-Дам-де-ла-Гард","Notre-Dame de la Garde")),
 ("toulouse","Toulouse",43.6047,1.4442,"fr",["Тулуза","Toulouse","Toulouse","Tuluza","Toulouse","Toulouse"],504078,2023,-100,0,118.3,("Гаронна","Garonne"),("Окситанія","Occitania"),("Базиліка Сен-Сернен","Basilica of Saint-Sernin")),
 ("nice","nice",43.7102,7.2620,"fr",["Ніцца","Nice","Nizza","Nicea","Nice","Niza"],348085,2023,-350,0,71.9,("Пайон","Paillon"),("Прованс-Альпи-Лазурний Берег","Provence-Alpes-Côte d'Azur"),("Англійська набережна","Promenade des Anglais")),
 ("bordeaux","Bordeaux",44.8378,-0.5792,"fr",["Бордо","Bordeaux","Bordeaux","Bordeaux","Bordeaux","Burdeos"],261804,2023,-300,0,49.4,("Гаронна","Garonne"),("Нова Аквітанія","Nouvelle-Aquitaine"),("Площа Біржі","Place de la Bourse")),
 ("nantes","Nantes",47.2184,-1.5536,"fr",["Нант","Nantes","Nantes","Nantes","Nantes","Nantes"],320732,2023,-70,0,65.2,("Луара","Loire"),("Пеї-де-ла-Луар","Pays de la Loire"),("Замок герцогів Бретані","Château des ducs de Bretagne")),
 ("strasbourg","Strasbourg",48.5734,7.7521,"fr",["Страсбург","Strasbourg","Straßburg","Strasburg","Strasbourg","Estrasburgo"],291313,2023,-12,0,78.3,("Іль","Ill"),("Гранд-Ест","Grand Est"),("Страсбурзький собор","Strasbourg Cathedral")),
 ("brussels","Brussels",50.8503,4.3517,"fr",["Брюссель","Brussels","Brüssel","Bruksela","Bruxelles","Bruselas"],1249597,2023,979,0,162.4,("Сенна","Senne"),("Брюссельський столичний регіон","Brussels-Capital Region"),("Гран-Плас","Grand-Place")),
 ("geneva","Geneva",46.2044,6.1432,"fr",["Женева","Geneva","Genf","Genewa","Genève","Ginebra"],203856,2023,-121,0,15.9,("Рона","Rhône"),("Кантон Женева","Canton of Geneva"),("Же-до","Jet d'Eau")),
 ("madrid","Madrid",40.4168,-3.7038,"es",["Мадрид","Madrid","Madrid","Madryt","Madrid","Madrid"],3223334,2023,865,1,604.3,("Мансанарес","Manzanares"),("Автономна спільнота Мадрид","Community of Madrid"),("Королівський палац","Royal Palace")),
 ("barcelona","Barcelona",41.3874,2.1686,"es",["Барселона","Barcelona","Barcelona","Barcelona","Barcelone","Barcelona"],1620343,2023,-218,0,101.4,("Бесос","Besòs"),("Каталонія","Catalonia"),("Саграда Фамілія","Sagrada Família")),
 ("valencia","Valencia",39.4699,-0.3763,"es",["Валенсія","Valencia","Valencia","Walencja","Valence","Valencia"],807693,2023,-138,0,134.6,("Турія","Turia"),("Валенсійська спільнота","Valencian Community"),("Місто мистецтв і наук","City of Arts and Sciences")),
 ("seville","Seville",37.3891,-5.9845,"es",["Севілья","Seville","Sevilla","Sewilla","Séville","Sevilla"],681998,2023,-206,0,140.8,("Гвадалквівір","Guadalquivir"),("Андалусія","Andalusia"),("Хіральда","La Giralda")),
 ("zaragoza","Zaragoza",41.6488,-0.8891,"es",["Сарагоса","Zaragoza","Saragossa","Saragossa","Saragosse","Zaragoza"],681877,2023,-14,0,973.8,("Ебро","Ebro"),("Арагон","Aragon"),("Базиліка дель Пілар","Basilica del Pilar")),
 ("malaga","Malaga",36.7213,-4.4214,"es",["Малага","Malaga","Málaga","Malaga","Malaga","Málaga"],586384,2023,-770,0,395.1,("Гвадальмедіна","Guadalmedina"),("Андалусія","Andalusia"),("Алькасаба","Alcazaba")),
 ("bilbao","Bilbao",43.2630,-2.9350,"es",["Більбао","Bilbao","Bilbao","Bilbao","Bilbao","Bilbao"],346843,2023,1300,0,41.3,("Нервйон","Nervión"),("Країна Басків","Basque Country"),("Музей Гуггенхайма","Guggenheim Museum")),
 ("lisbon","Lisbon",38.7223,-9.1393,"es",["Лісабон","Lisbon","Lissabon","Lizbona","Lisbonne","Lisboa"],548703,2023,-1200,0,100.1,("Тежу","Tagus"),("Лісабонський регіон","Lisbon Region"),("Вежа Белен","Belém Tower")),
 ("porto","porto",41.1579,-8.6291,"es",["Порту","Porto","Porto","Porto","Porto","Oporto"],231962,2023,-300,0,41.4,("Дору","Douro"),("Північний регіон","Norte Region"),("Міст Луїша I","Dom Luís I Bridge")),
 ("rome","Rome",41.9028,12.4964,"es",["Рим","Rome","Rom","Rzym","Rome","Roma"],2748109,2023,-753,0,1287.4,("Тибр","Tiber"),("Лаціо","Lazio"),("Колізей","Colosseum")),
 ("milan","Milan",45.4642,9.1900,"es",["Мілан","Milan","Mailand","Mediolan","Milan","Milán"],1371498,2023,-590,0,181.8,("Олона","Olona"),("Ломбардія","Lombardy"),("Міланський собор","Duomo di Milano")),
 ("naples","Naples",40.8518,14.2681,"es",["Неаполь","Naples","Neapel","Neapol","Naples","Nápoles"],913462,2023,-470,0,119.0,("Себето","Sebeto"),("Кампанія","Campania"),("Везувій","Mount Vesuvius")),
 ("florence","Florence",43.7696,11.2558,"es",["Флоренція","Florence","Florenz","Florencja","Florence","Florencia"],361619,2023,-59,0,102.4,("Арно","Arno"),("Тоскана","Tuscany"),("Санта-Марія-дель-Фйоре","Florence Cathedral")),
 ("venice","Venice",45.4408,12.3155,"es",["Венеція","Venice","Venedig","Wenecja","Venise","Venecia"],250369,2023,421,0,414.6,("Гранд-канал","Grand Canal"),("Венето","Veneto"),("Площа Сан-Марко","St Mark's Square")),
 ("turin","Turin",45.0703,7.6869,"es",["Турин","Turin","Turin","Turyn","Turin","Turín"],847287,2023,-28,0,130.2,("По","Po"),("П'ємонт","Piedmont"),("Моле Антонелліана","Mole Antonelliana")),
 ("london","London",51.5074,-0.1278,"en",["Лондон","London","London","Londyn","Londres","Londres"],8866180,2022,47,0,1572.0,("Темза","Thames"),("Великий Лондон","Greater London"),("Тауерський міст","Tower Bridge")),
 ("manchester","Manchester",53.4808,-2.2426,"en",["Манчестер","Manchester","Manchester","Manchester","Manchester","Mánchester"],568996,2022,79,0,115.6,("Ірвелл","Irwell"),("Великий Манчестер","Greater Manchester"),("Стадіон Олд Траффорд","Old Trafford")),
 ("edinburgh","Edinburgh",55.9533,-3.1883,"en",["Единбург","Edinburgh","Edinburgh","Edynburg","Édimbourg","Edimburgo"],526470,2022,1130,0,264.0,("Вотер-оф-Літ","Water of Leith"),("Шотландія","Scotland"),("Единбурзький замок","Edinburgh Castle")),
 ("dublin","Dublin",53.3498,-6.2603,"en",["Дублін","Dublin","Dublin","Dublin","Dublin","Dublín"],592713,2022,841,0,117.8,("Ліффі","Liffey"),("Ленстер","Leinster"),("Трініті-коледж","Trinity College")),
 ("amsterdam","Amsterdam",52.3676,4.9041,"en",["Амстердам","Amsterdam","Amsterdam","Amsterdam","Amsterdam","Ámsterdam"],921402,2023,1275,1,219.3,("Амстел","Amstel"),("Північна Голландія","North Holland"),("Канали Амстердама","Amsterdam Canals")),
 ("rotterdam","Rotterdam",51.9244,4.4777,"en",["Роттердам","Rotterdam","Rotterdam","Rotterdam","Rotterdam","Róterdam"],664311,2023,1270,0,324.1,("Ньїве-Маас","Nieuwe Maas"),("Південна Голландія","South Holland"),("Міст Еразма","Erasmus Bridge")),
 ("copenhagen","Copenhagen",55.6761,12.5683,"en",["Копенгаген","Copenhagen","Kopenhagen","Kopenhaga","Copenhague","Copenhague"],653664,2023,1043,1,88.3,("Ересунн","Øresund"),("Столичний регіон","Capital Region"),("Русалонька","The Little Mermaid")),
 ("stockholm","Stockholm",59.3293,18.0686,"en",["Стокгольм","Stockholm","Stockholm","Sztokholm","Stockholm","Estocolmo"],984748,2023,1252,1,188.0,("Норрстрем","Norrström"),("Стокгольмський лен","Stockholm County"),("Гамла Стан","Gamla Stan")),
 ("oslo","Oslo",59.9139,10.7522,"en",["Осло","Oslo","Oslo","Oslo","Oslo","Oslo"],709037,2023,1040,0,454.0,("Акерсельва","Akerselva"),("Осло","Oslo County"),("Оперний театр","Oslo Opera House")),
 ("helsinki","Helsinki",60.1699,24.9384,"en",["Гельсінкі","Helsinki","Helsinki","Helsinki","Helsinki","Helsinki"],664000,2023,1550,0,214.3,("Вантаа","Vantaa"),("Уусімаа","Uusimaa"),("Кафедральний собор","Helsinki Cathedral")),
 ("athens","Athens",37.9838,23.7275,"en",["Афіни","Athens","Athen","Ateny","Athènes","Atenas"],643452,2023,-1400,0,38.96,("Кіфісос","Kifisos"),("Аттика","Attica"),("Акрополь","Acropolis")),
 ("riga","riga",56.9496,24.1052,"en",["Рига","Riga","Riga","Ryga","Riga","Riga"],605802,2023,1201,0,304.0,("Даугава","Daugava"),("Ризький регіон","Riga Region"),("Будинок Чорноголових","House of the Blackheads")),
 ("vilnius","Vilnius",54.6872,25.2797,"en",["Вільнюс","Vilnius","Wilna","Wilno","Vilnius","Vilna"],581475,2023,1323,1,401.0,("Нярис","Neris"),("Вільнюський повіт","Vilnius County"),("Вежа Гедиміна","Gediminas' Tower")),
 ("tallinn","Tallinn",59.4370,24.7536,"en",["Таллінн","Tallinn","Tallinn","Tallin","Tallinn","Tallin"],461000,2023,1219,1,159.2,("Пирита","Pirita"),("Харьюмаа","Harju County"),("Старе місто","Tallinn Old Town")),
]

HEAD = '''import type { AppLocale } from "@/i18n/routing";
import type { CityFacts } from "@/lib/cityFacts";

/**
 * SEO-РОЗШИРЕННЯ НА ЄС (29.07.2026): сторінки міст ЄВРОПИ для /maps/[slug].
 *
 * Навіщо: локалі de/pl/fr/es досі мали контент лише про українські міста —
 * нульовий локальний інтерес для їхньої аудиторії. Німець шукає
 * «3D-Karte Berlin», поляк «mapa 3D Warszawy». Конструктор і так працює для
 * всього світу, тож це НЕ doorway — сторінки під реальний продукт.
 *
 * АНТИ-DOORWAY ПРАВИЛО: місто потрапляє сюди ЛИШЕ з реальними унікальними
 * даними (населення/рік/заснування/площа/річка/регіон/орієнтир) — той самий
 * контракт CityFacts, що й для українських міст. Без даних — без сторінки.
 *
 * ФАЙЛ ЗГЕНЕРОВАНО: scripts/_gen_world_cities.py (правити там, не тут).
 */
export interface WorldCity {
  slug: string;
  /** Ключ пресету конструктора (CITIES у app/[locale]/create/page.tsx). */
  key: string;
  center: [number, number];
  /** Мова-таргет: для внутрішньої перелінковки й пріоритету в sitemap. */
  primaryLocale: AppLocale;
  names: Record<AppLocale, string>;
  facts: CityFacts;
}

export const WORLD_CITIES: WorldCity[] = [
'''

TAIL = '''];

export const WORLD_CITY_BY_SLUG: Record<string, WorldCity> = Object.fromEntries(
  WORLD_CITIES.map((c) => [c.slug, c]),
);

/** Міста, для яких ця локаль — «домашня» (перелінковка «поблизу»). */
export function worldCitiesForLocale(locale: AppLocale): WorldCity[] {
  const home = WORLD_CITIES.filter((c) => c.primaryLocale === locale);
  return home.length ? home : WORLD_CITIES.slice(0, 6);
}
'''

def esc(s):
    return s.replace('"', '\\"')

rows = []
for (slug, key, lat, lon, loc, names, pop, py, founded, fm, area, river, region, lm) in C:
    uk, en, de, pl, fr, es = names
    rows.append(
        f'  {{ slug: "{slug}", key: "World_{key.capitalize() if key.islower() else key}", center: [{lat}, {lon}], primaryLocale: "{loc}",\n'
        f'    names: {{ uk: "{esc(uk)}", en: "{esc(en)}", de: "{esc(de)}", pl: "{esc(pl)}", fr: "{esc(fr)}", es: "{esc(es)}" }},\n'
        f'    facts: {{ population: {pop}, populationYear: {py}, founded: {founded}, firstMention: {"true" if fm else "false"}, area_km2: {area},\n'
        f'      river: {{ uk: "{esc(river[0])}", latin: "{esc(river[1])}" }}, oblast: {{ uk: "{esc(region[0])}", latin: "{esc(region[1])}" }},\n'
        f'      landmark: {{ uk: "{esc(lm[0])}", latin: "{esc(lm[1])}" }} }} }},'
    )

out = HEAD + "\n".join(rows) + "\n" + TAIL
io.open("lib/worldCities.ts", "w", encoding="utf-8").write(out)
print("cities:", len(C))
