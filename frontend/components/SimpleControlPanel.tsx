"use client";

import { useCallback, useEffect, useState } from "react";
import { Loader2, Play, Download, MapPin, Check, Sparkles, ShoppingBag, ChevronDown, Sliders } from "lucide-react";
import { useTranslations } from "next-intl";
import { useGenerationStore } from "@/store/generation-store";
import { MAP_TEMPLATES, MAP_STYLE_PRESETS } from "@/lib/templates";
import { buildMapRequest, SIMPLE_SIZES, GPX_MAX_M_PER_MM } from "@/lib/generation";
import { OrderDialog } from "@/components/OrderDialog";
import { StickyActionBar } from "@/components/StickyActionBar";
import { useAuth } from "@/components/AuthProvider";
import { gatedDownload } from "@/lib/download";
import { fetchQuote, type Quote } from "@/lib/pricing";
import { MAP_MAGNET_PRICE_UAH, MAP_RELIEF_ADDON_UAH } from "@/lib/mapPrices";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

/**
 * Simple, preset-first map builder shown by default.
 * Three decisions: location (featured district card or draw) → style → size → Generate.
 * All fine-grained sliders live in the full ControlPanel ("Про" mode).
 */
export function SimpleControlPanel({
  availableCities,
  selectedCityKey,
  onCityChange,
  onAdvanced,
  showStickyBar = true,
}: {
  availableCities?: Record<string, { center: [number, number]; bounds: any }>;
  selectedCityKey?: string;
  onCityChange?: (key: string) => void;
  onAdvanced?: () => void;
  // Панель монтується ДВІЧІ (desktop aside + mobile section). StickyActionBar
  // — портал у <body>, тож обидві копії малювали його → ДВА бари на мобільному
  // (+ inline-кнопка = «3 кнопки генерації»). Малюємо лише з мобільної копії.
  showStickyBar?: boolean;
}) {
  const t = useTranslations("simple");
  const tOrder = useTranslations("order");
  const s = useGenerationStore();
  const {
    selectedArea, setSelectedArea,
    isGenerating, downloadUrl, progress, status, printQuality,
    taskGroupId, setTaskGroup, setActiveTaskId, setGenerating,
    setDownloadUrl, setTaskStatuses, updateProgress,
    modelSizeMm, setModelSizeMm, setGpxFocus,
    setTerrainEnabled,
    setPreviewIncludeBuildings, setPreviewIncludeRoads,
    setPreviewIncludeWater, setPreviewIncludeParks,
  } = s;

  // styleId/activeTemplate — теж у store (панель монтується двічі: desktop+mobile);
  // локальний стан розсинхронізовувався → при ресайзі/генерації з іншої копії
  // застосовувався старий стиль/шаблон. Той самий клас багу, що й магніт/панно.
  const styleId = s.simpleStyleId;
  const setStyleId = s.setSimpleStyleId;
  // МАГНІТ/ПАННО/GPX — у zustand store, НЕ useState: панель змонтована двічі
  // (desktop + mobile), локальний стан розсинхронізовувався між копіями і
  // вибір губився при генерації з іншої копії.
  const magnetMode = s.simpleMagnetMode;
  const setMagnetMode = s.setSimpleMagnetMode;
  const mapLabel = s.simpleMapLabel;
  const setMapLabel = s.setSimpleMapLabel;
  const panelMode = s.simplePanelMode;
  const setPanelMode = s.setSimplePanelMode;
  const flatAmsMode = s.simpleFlatAms;
  const setFlatAmsMode = s.setSimpleFlatAms;
  // З'ЄДНУВАЧ-ПАЗИ (метелик): стикує дві плоскі карти; стан у store (панель ×2).
  const connectorMode = s.simpleConnector;
  const setConnectorMode = s.setSimpleConnector;
  // ПРЕМІУМ-РАМКА: компас + масштабна лінійка + координати поверх плоскої карти.
  const frameMode = s.simpleFrame;
  const setFrameMode = s.setSimpleFrame;
  // РЕЛЬЄФ (висоти землі): окремий перемикач для усіх режимів (3D-карта).
  const reliefMode = s.simpleRelief;
  const setReliefMode = s.setSimpleRelief;
  // ПЛАСКІ БУДИНКИ у плоских режимах (тонкі footprint-плити).
  const flatBuildingsMode = s.simpleFlatBuildings;
  const setFlatBuildingsMode = s.setSimpleFlatBuildings;
  // D4 GPX-трек: точки живуть у gpxFocus (їх же використовує карта-оверлей)
  const gpxTrack = s.gpxFocus?.points ?? null;
  const gpxName = s.gpxName;
  const setGpxName = s.setGpxName;
  const gpxNote = s.gpxNote;
  const setGpxNote = s.setGpxNote;
  // E4 ШЕРИНГ: рендер моделі → /share/{task} з og:image
  const [shareBusy, setShareBusy] = useState(false);
  const [shareCopied, setShareCopied] = useState(false);
  // «Більше опцій» (магніт/GPX/панно) сховані за замовчанням — Просто-режим має
  // бути коротким: Місто → Район → Стиль → Розмір → Створити. Авто-розкривається,
  // якщо одна з опцій уже активна (відновлена зі store), щоб вибір не «зник».
  const advancedActive = magnetMode || !!gpxTrack || panelMode > 0 || flatAmsMode || connectorMode || frameMode;
  const [moreOpen, setMoreOpen] = useState(advancedActive);
  useEffect(() => { if (advancedActive) setMoreOpen(true); }, [advancedActive]);

  const doShare = async () => {
    if (!taskGroupId) return;
    setShareBusy(true);
    try {
      // Завантаження прев'ю-картинки = best-effort: якщо впаде (мережа/таймаут),
      // НЕ блокуємо шеринг — посилання все одно валідне (share-сторінка рендериться без og:image).
      try {
        const { capturePreviewImages } = await import("@/lib/capturePreview");
        const shots = await capturePreviewImages();
        const png = shots.find((s) => s.startsWith("data:image/png"));
        if (png) {
          await fetch(`${API_BASE}/api/share/preview`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ task_id: taskGroupId, image: png }),
          });
        }
      } catch { /* прев'ю опційне */ }
      const url = `${window.location.origin}/share/${taskGroupId}`;
      if (typeof navigator.share === "function") {
        await navigator.share({ url, title: "Monadruk" }).catch(() => {});
      } else {
        await navigator.clipboard.writeText(url);
        setShareCopied(true);
        setTimeout(() => setShareCopied(false), 2500);
      }
    } catch { /* ignore */ }
    setShareBusy(false);
  };
  const [quote, setQuote] = useState<Quote | null>(null);
  const [error, setError] = useState<string | null>(null);
  const activeTemplate = s.simpleTemplate;
  const setActiveTemplate = s.setSimpleTemplate;
  const [orderOpen, setOrderOpen] = useState(false);
  const [dlBusy, setDlBusy] = useState(false);
  // Прогрес фонової генерації друкарського 3MF при завантаженні стандартної карти
  // (на екрані — швидкий GLB-прев'ю; друкарський файл готуємо на вимогу). null = не йде.
  const [printPrep, setPrintPrep] = useState<number | null>(null);
  const { getIdToken, openLogin } = useAuth();
  const [quota, setQuota] = useState<{ remaining: number; limit: number; isAdmin?: boolean } | null>(null);

  // Лічильник безкоштовних завантажень (вхід + 5 безкоштовних, далі замовлення).
  const refreshQuota = useCallback(async () => {
    try {
      const token = await getIdToken();
      if (!token) { setQuota(null); return; }
      const r = await fetch(`${API_BASE}/api/account/quota`, { headers: { Authorization: `Bearer ${token}` } });
      if (!r.ok) return;
      const d = await r.json();
      const q = d?.quota || d;
      if (q) setQuota({ remaining: Number(q.remaining ?? 0), limit: Number(q.limit ?? 5), isAdmin: Boolean(q.is_admin) });
    } catch { /* ignore */ }
  }, [getIdToken]);
  useEffect(() => { refreshQuota(); }, [refreshQuota]);

  // Воронка конверсії (1 раз/сесію кожен крок): перегляд конструктора + вибір зони.
  // Решта кроків — generate (нижче) / order_open / order_submit (OrderDialog).
  useEffect(() => { import("@/lib/analytics").then((m) => m.trackFunnel("view")).catch(() => {}); }, []);
  useEffect(() => { if (selectedArea) import("@/lib/analytics").then((m) => m.trackFunnel("area")).catch(() => {}); }, [selectedArea]);

  // Жива орієнтовна ціна — оновлюється при зміні розміру/стилю (relief = +надбавка).
  useEffect(() => {
    let alive = true;
    const relief = reliefMode;  // окремий перемикач «Рельєф» — джерело правди
    // Магніт — окремий продукт із фіксованою ціною (ключ розміру 60 = 180₴), а
    // НЕ звичайна мапа за вибраним S/M/L/XL. Без цього у формі показувалась ціна
    // мапи (напр. 250₴ замість 180₴). Генерація теж форсує modelSizeMm=60.
    fetchQuote("map", magnetMode ? 60 : modelSizeMm, magnetMode ? false : relief).then((q) => { if (alive) setQuote(q); });
    return () => { alive = false; };
  }, [modelSizeMm, styleId, magnetMode, reliefMode]);

  // Ціна для форми замовлення. КРИТИЧНО: для панно множимо на кількість плиток —
  // 3×3 = 9 окремих мап, раніше коштувало як 1 плитка → ~9× недозбір (і LiqPay
  // брав суму однієї). Магніт-fallback = 180₴ (не ціна мапи). Quote вже per-tile.
  const orderTiles = panelMode > 0 ? panelMode * panelMode : 1;
  const fmtPrice = (n: number, currency: string) =>
    currency === "EUR" ? `€${n}` : `${n} ₴`;
  const orderPriceText = (() => {
    if (quote) {
      return orderTiles > 1 ? fmtPrice(quote.price * orderTiles, quote.currency) : quote.formatted;
    }
    const near = SIMPLE_SIZES.reduce((best, z) =>
      Math.abs(z.mm - modelSizeMm) < Math.abs(best.mm - modelSizeMm) ? z : best, SIMPLE_SIZES[0]);
    // Рельєф додає надбавку (як у бекенд-quote) — інакше fallback недооцінює.
    // Ціни з єдиного джерела mapPrices.ts (не хардкод) — щоб fallback не розходився з quote.
    const reliefAddon = (reliefMode && !magnetMode) ? MAP_RELIEF_ADDON_UAH : 0;
    const unit = magnetMode ? MAP_MAGNET_PRICE_UAH : near.price + reliefAddon; // магніт = окремий продукт
    return fmtPrice(unit * orderTiles, "UAH");
  })();

  const doGatedDownload = async () => {
    setDlBusy(true);
    setError(null);
    try {
      let dlTaskId = taskGroupId;
      let dlUrl = downloadUrl;
      // Стандартна карта на екрані = ШВИДКИЙ GLB-прев'ю. Завантаження мусить дати
      // ДРУКАРСЬКИЙ 3MF (вимога: адміни + юзери з безкоштовними/докупленими качають 3MF).
      // Магніт/панно/брелки вже завжди 3MF — їх віддаємо напряму.
      if (dlUrl && /\.glb(\?|$)/i.test(dlUrl)) {
        // Не запускаємо дорогу 3MF-генерацію (1-3 хв), якщо юзер не залогінений або
        // вичерпав ліміт — перевіряємо ПЕРЕД генерацією, щоб не змусити чекати намарно.
        const token = await getIdToken().catch(() => null);
        if (!token) { openLogin(); setDlBusy(false); return; }
        if (quota && !quota.isAdmin && quota.remaining <= 0) {
          window.dispatchEvent(new CustomEvent("monadruk:open-contact", { detail: { message: t("limitMsg") } }));
          setDlBusy(false); return;
        }
        setPrintPrep(0);
        const print = await generatePrint3mf((p) => setPrintPrep(p));
        setPrintPrep(null);
        if (!print) { setError(t("errGen")); setDlBusy(false); return; }
        dlTaskId = print.taskId;
        dlUrl = print.url;
      }
      const res = await gatedDownload({
        taskId: dlTaskId, downloadUrl: dlUrl,
        meta: { title: selectedCityKey, city: selectedCityKey, product_type: "map" },
        getIdToken, openLogin,
        onLimit: () => window.dispatchEvent(new CustomEvent("monadruk:open-contact", {
          detail: { message: t("limitMsg") },
        })),
      });
      if (res.quota && typeof res.quota.remaining === "number") {
        setQuota((q) => ({ remaining: res.quota!.remaining as number, limit: q?.limit ?? 5, isAdmin: q?.isAdmin }));
      } else if (res.status === "ok") {
        refreshQuota();
      }
    } finally {
      setPrintPrep(null);
      setDlBusy(false);
    }
  };

  const cityKeys = availableCities ? Object.keys(availableCities) : [];
  const featured = MAP_TEMPLATES.filter((t) => t.cityKey === selectedCityKey);

  // Poll task status (this panel can be the only one mounted).
  // UX-FIX: термінальні стани (cancelled/зниклий task після рестарту сервера)
  // ОБОВʼЯЗКОВО розблоковують UI — інакше «Генерація N%» висіла назавжди
  // і переживала перезавантаження сторінки (taskGroupId у localStorage).
  useEffect(() => {
    if (!taskGroupId) return;
    let stop = false;
    let pollFails = 0;
    const iv = setInterval(async () => {
      try {
        const { api } = await import("@/lib/api");
        const r: any = await api.getStatus(taskGroupId);
        if (stop) return;
        pollFails = 0;
        setTaskStatuses({ [r.task_id]: r });
        // D3 ПАННО: batch-статус — агрегуємо прогрес плиток; коли всі готові,
        // даємо посилання на zip-архів з усіма плитками + layout.png
        if (r.status === "multiple") {
          const total = Number(r.total || 0);
          const done = Number(r.completed || 0);
          const subTasks: any[] = r.tasks || [];
          const avg = subTasks.length
            ? Math.round(subTasks.reduce((acc, st) => acc + Number(st.progress || 0), 0) / subTasks.length)
            : 0;
          updateProgress(avg, `${t("panelTiles")}: ${done}/${total}`);
          const allTerminal = subTasks.length > 0 && subTasks.every(
            (st) => st.status === "completed" || st.status === "failed" || st.status === "cancelled",
          );
          if (subTasks.some((st) => st.status === "failed")) {
            setGenerating(false);
            setError(t("errGen"));
            clearInterval(iv);
          } else if (total > 0 && done === total) {
            setDownloadUrl(`/api/zones/${taskGroupId}/download_all`);
            setGenerating(false);
            clearInterval(iv);
          } else if (allTerminal) {
            // Скасований batch: тихо розблоковуємо — юзер може почати заново
            setGenerating(false);
            clearInterval(iv);
          }
          return;
        }
        updateProgress(r.progress, r.message);
        if (r.status === "completed") {
          setDownloadUrl(r.download_url);
          (s as any).setPrintQuality?.(r.print_quality ?? null);
          setGenerating(false);
          clearInterval(iv);
        } else if (r.status === "failed" || r.status === "cancelled") {
          setGenerating(false);
          if (r.status === "failed") setError(r.message || t("errGen"));
          clearInterval(iv);
        }
      } catch {
        // 404/мережа: 4 поспіль (~10с) = задача зникла (рестарт сервера) —
        // розблоковуємо UI замість вічного спінера
        pollFails += 1;
        if (pollFails >= 4 && !stop) {
          setGenerating(false);
          setError(t("errStale"));
          clearInterval(iv);
        }
      }
    }, 2500);
    return () => { stop = true; clearInterval(iv); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [taskGroupId]);

  // UX-FIX: «Замовити друк» зі сторонніх місць (Швидкий статус на мобілці)
  // відкриває OrderDialog цієї панелі через глобальну подію.
  useEffect(() => {
    const open = () => setOrderOpen(true);
    window.addEventListener("monadruk:open-order", open);
    return () => window.removeEventListener("monadruk:open-order", open);
  }, []);

  // UX-FIX: явне скасування генерації — DELETE на бек + миттєве розблокування
  const cancelGeneration = async () => {
    try {
      const { api } = await import("@/lib/api");
      if (taskGroupId) await api.cancelTask(taskGroupId).catch(() => {});
    } finally {
      setGenerating(false);
      updateProgress(0, "");
    }
  };

  const applyStyle = (id: string) => {
    const preset = MAP_STYLE_PRESETS.find((p) => p.id === id);
    if (!preset) return;
    setStyleId(id);
    setPreviewIncludeBuildings(preset.layers.buildings);
    setPreviewIncludeRoads(preset.layers.roads);
    setPreviewIncludeWater(preset.layers.water);
    setPreviewIncludeParks(preset.layers.parks);
    setTerrainEnabled(preset.layers.terrain);
    // Стиль задає ДЕФОЛТ рельєфу; далі окремий перемикач «Рельєф» — джерело правди.
    setReliefMode(preset.layers.terrain);
  };

  // Чернетка конструктора: зона/стиль/розмір переживають перезавантаження.
  // Відновлюємо РАЗ при маунті; зберігаємо з debounce при змінах.
  useEffect(() => {
    try {
      const raw = localStorage.getItem("monadruk:draft:create");
      if (!raw) return;
      const d = JSON.parse(raw);
      if (d.styleId && MAP_STYLE_PRESETS.some((p) => p.id === d.styleId)) applyStyle(d.styleId);
      if (typeof d.modelSizeMm === "number" && d.modelSizeMm >= 40) setModelSizeMm(d.modelSizeMm);
      // КРИТИЧНО: JSON.parse повертає plain object, НЕ Leaflet LatLngBounds —
      // прямий setSelectedArea(d.selectedArea) валив /create і /keychains
      // («getNorth/getCenter is not a function», store спільний між роутами).
      // Реконструюємо справжній L.LatLngBounds з координат чернетки.
      if (!selectedArea && d.selectedArea && typeof d.selectedArea === "object") {
        const sw = d.selectedArea._southWest;
        const ne = d.selectedArea._northEast;
        if (
          sw && ne &&
          typeof sw.lat === "number" && typeof sw.lng === "number" &&
          typeof ne.lat === "number" && typeof ne.lng === "number"
        ) {
          import("leaflet")
            .then((L) => {
              setSelectedArea(new L.LatLngBounds([sw.lat, sw.lng], [ne.lat, ne.lng]) as any);
            })
            .catch(() => { /* ignore */ });
        }
      }
    } catch { /* ignore */ }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => {
    const timer = setTimeout(() => {
      try {
        localStorage.setItem("monadruk:draft:create", JSON.stringify({ selectedArea, styleId, modelSizeMm }));
      } catch { /* ignore */ }
    }, 800);
    return () => clearTimeout(timer);
  }, [selectedArea, styleId, modelSizeMm]);

  const pickTemplate = (id: string) => {
    const tpl = MAP_TEMPLATES.find((t) => t.id === id);
    if (!tpl) return;
    setActiveTemplate(id);
    setError(null);
    // ІДЕАЛЬНИЙ ПРИКЛАД одним кліком: ставимо куровані СТИЛЬ + РОЗМІР, ЛЕТИМО картою
    // до району і ставимо зону точно під розмір моделі (1:10000 → мм×10 м). Раніше
    // pickTemplate робив лише setSelectedArea → карта НЕ рухалась і налаштування НЕ
    // мінялись («не працює»). Тепер клік готує сцену до генерації повністю.
    if (tpl.style) applyStyle(tpl.style);
    const sizeMm = tpl.sizeMm ?? modelSizeMm ?? 80;
    if (tpl.sizeMm) setModelSizeMm(tpl.sizeMm);
    const [lat, lon] = tpl.center;
    // Затримка — щоб зміна розміру встигла перебудувати оверлей зони ДО map-goto,
    // інакше ребілд міг би перетерти щойно поставлену зону. map-goto з явним widthM
    // летить до району + ставить зону рівно sizeMm×10 м (handler ігнорує clamp).
    window.setTimeout(() => {
      window.dispatchEvent(new CustomEvent("monadruk:map-goto", {
        detail: { lat, lon, widthM: sizeMm * 10 },
      }));
    }, 180);
  };

  // Будує запит карти. forPrint=true → ПОВНА друкарська якість (preview_mode=false,
  // export 3mf) для завантаження/замовлення; інакше — швидке GLB-прев'ю на екрані.
  // Стандартна карта на екрані рендериться як легкий GLB; друкарський 3MF
  // генерується НА ВИМОГУ при download/order (магніт/панно вже завжди 3MF).
  const buildSingleMapReq = (forPrint: boolean) => {
    const preset = MAP_STYLE_PRESETS.find((p) => p.id === styleId);
    const layerBuildings = preset ? preset.layers.buildings : s.previewIncludeBuildings;
    const layerRoads = preset ? preset.layers.roads : s.previewIncludeRoads;
    const layerWater = preset ? preset.layers.water : s.previewIncludeWater;
    const layerParks = preset ? preset.layers.parks : s.previewIncludeParks;
    // «Плоска кольорова (AMS)»: пласка багатокольорова плитка-карта (terrain off,
    // окремі кольорові шари, основа 3мм). Лише для одиночної карти (не панно/магніт).
    const flatAms = panelMode === 0 && !magnetMode && s.simpleFlatAms;
    // З'ЄДНУВАЧ-ПАЗИ: вимагає плоского режиму (3мм основа, паз у дні). Сумісний
    // з flatAms (кольорова плитка з пазами), несумісний з магнітом/панно.
    const connector = panelMode === 0 && !magnetMode && s.simpleConnector;
    // ПРЕМІУМ-РАМКА: компас+лінійка+координати поверх плоскої карти. Будується у
    // flat_plate (тож вимагає плоского режиму). Сумісна з flatAms/connector/магнітом
    // (магніт уже плоский); несумісна лише з панно (3D-плитки, інший пайплайн).
    const frame = panelMode === 0 && s.simpleFrame;
    const flatPlate = flatAms || connector || frame;
    // РЕЛЬЄФ (висоти землі) — окремий перемикач, джерело правди для terrain. Працює
    // на 3D-карті (стандарт + панно); плоскі режими/магніт фізично без рельєфу.
    const relief = !magnetMode && !flatPlate && reliefMode;
    // ПЛАСКІ БУДИНКИ — лише у плоских режимах (footprint-плити одної низької висоти).
    const flatBuildings = (flatPlate || magnetMode) && flatBuildingsMode;
    // ПОВЕРНУТА мапа: розширюємо fetch-bbox до AABB повернутого полігона (як у брелках).
    let fN = selectedArea!.getNorth(), fS = selectedArea!.getSouth();
    let fE = selectedArea!.getEast(), fW = selectedArea!.getWest();
    const zpoly = s.zonePolygonCoords;
    if (panelMode === 0 && zpoly && zpoly.length >= 3) {
      for (const [lon, lat] of zpoly) {
        fN = Math.max(fN, lat); fS = Math.min(fS, lat);
        fE = Math.max(fE, lon); fW = Math.min(fW, lon);
      }
    }
    return buildMapRequest({
      north: fN, south: fS,
      east: fE, west: fW,
      roadWidthMultiplier: s.roadWidthMultiplier, roadHeightMm: s.roadHeightMm, roadEmbedMm: s.roadEmbedMm,
      buildingMinHeight: s.buildingMinHeight, buildingHeightMultiplier: s.buildingHeightMultiplier,
      buildingFoundationMm: s.buildingFoundationMm, buildingEmbedMm: s.buildingEmbedMm,
      waterDepth: s.waterDepth, terrainEnabled: relief, terrainZScale: s.terrainZScale,
      terrainBaseThicknessMm: (magnetMode || flatPlate) ? 3.0 : s.terrainBaseThicknessMm, terrainResolution: s.terrainResolution,
      terrariumZoom: s.terrariumZoom,
      flatUniformBuildingHeight: flatBuildings,
      flatMaxBuildingHeightMm: flatBuildings ? 0.8 : undefined,
      // forPrint → друкарський 3MF (не GLB-прев'ю).
      exportFormat: forPrint ? "3mf" : s.exportFormat,
      modelSizeMm: magnetMode ? 60 : s.modelSizeMm,
      isAmsMode: flatAms ? true : s.isAmsMode,
      // Панно = повні 3D-плитки: магніт/превʼю вимикаються примусово.
      // flatAms/connector → пласка плитка (flat_plate колірні шари + пази у дні).
      flatPlateMode: panelMode > 0 ? false : (magnetMode || flatPlate) ? true : s.flatPlateMode,
      // forPrint → ЗАВЖДИ повна якість; інакше швидке прев'ю лише для стандартної карти.
      // flatAms/connector = повний кольоровий 3MF у прев'ю (flat_plate сам і є прев'ю, не GLB).
      previewMode: forPrint ? false : (panelMode > 0 || magnetMode || flatPlate ? false : s.previewMode),
      magnetPocket: panelMode > 0 ? false : magnetMode,
      mapConnector: connector,
      mapFrame: frame,
      mapLabel: magnetMode && panelMode === 0 ? mapLabel : "",
      gpxTrack,
      previewIncludeBase: s.previewIncludeBase, previewIncludeRoads: layerRoads,
      previewIncludeBuildings: layerBuildings, previewIncludeWater: layerWater,
      previewIncludeParks: layerParks,
      // Панно: плитки ріжуться строго по своїх bbox — фігурний полігон
      // (rounded-rect/коло/серце з /create) сюди потрапляти НЕ повинен.
      zonePolygonCoords: panelMode > 0 ? null : s.zonePolygonCoords,
    });
  };

  // Генерує ДРУКАРСЬКИЙ 3MF стандартної карти НА ВИМОГУ (download/order), не чіпаючи
  // GLB-прев'ю на екрані. Повертає {taskId, url} 3MF або null. onProg(0..100) для UI.
  const generatePrint3mf = async (onProg?: (p: number) => void): Promise<{ taskId: string; url: string } | null> => {
    try {
      const { api } = await import("@/lib/api");
      const req = buildSingleMapReq(true);
      const r = await api.generateModel(req as any);
      for (let i = 0; i < 90; i++) {
        await new Promise((res) => setTimeout(res, 4000));
        let st: any;
        try { st = await api.getStatus(r.task_id); } catch { continue; }
        if (typeof st?.progress === "number") onProg?.(st.progress);
        if (st?.status === "completed") {
          const url = st.download_url_3mf || st.download_url || "";
          return url ? { taskId: r.task_id, url } : null;
        }
        if (st?.status === "failed" || st?.status === "error") return null;
      }
      return null;
    } catch { return null; }
  };

  const handleGenerate = async (opts?: { forPrint?: boolean }) => {
    if (!selectedArea) { setError(t("errSelectArea")); return; }
    setError(null);
    setGenerating(true);
    // Ads/GA4: генерація = сильний сигнал наміру (ремаркетинг-аудиторія).
    import("@/lib/analytics").then((m) => { m.trackConversion("generate", { props: { product: "map" } }); m.trackFunnel("generate"); }).catch(() => {});
    try {
      const req = buildSingleMapReq(opts?.forPrint ?? false);
      const { api } = await import("@/lib/api");
      // D3 ПАННО: ділимо зону на R×C плиток (row 0 = ПІВНІЧ, col 0 = захід)
      // і шлемо batch у /api/generate-zones — бек гарантує ідеальні шви
      // (preserve_global_xy + спільний elevation_ref + grid_step).
      if (panelMode > 0) {
        const N = selectedArea.getNorth(), S = selectedArea.getSouth();
        const E = selectedArea.getEast(), W = selectedArea.getWest();
        const G = panelMode;
        const zones: any[] = [];
        for (let r = 0; r < G; r++) {
          for (let c = 0; c < G; c++) {
            const zn = N - (r * (N - S)) / G;
            const zs = N - ((r + 1) * (N - S)) / G;
            const zw = W + (c * (E - W)) / G;
            const ze = W + ((c + 1) * (E - W)) / G;
            zones.push({
              id: `tile_${r}_${c}`,
              geometry: { type: "Polygon", coordinates: [[[zw, zs], [ze, zs], [ze, zn], [zw, zn], [zw, zs]]] },
              properties: { row: r, col: c },
            });
          }
        }
        const batch = await api.generateZones(zones, req as any);
        const ids = batch.all_task_ids?.length ? batch.all_task_ids : [batch.task_id];
        setTaskGroup(batch.task_id, ids);
        setActiveTaskId(ids[0] ?? null);
        // СКЛАДЕНЕ ПРЕВʼЮ: показуємо всі плитки разом (та сама механіка, що
        // «Показати всі зони» у Профі) + row/col, щоб кожна стала на місце.
        s.setShowAllZones(true);
        const meta: Record<string, { zoneId: string; row?: number; col?: number }> = {};
        for (let i = 0; i < ids.length && i < zones.length; i += 1) {
          meta[String(ids[i])] = {
            zoneId: String(zones[i].id),
            row: zones[i].properties?.row,
            col: zones[i].properties?.col,
          };
        }
        s.setBatchZoneMetaByTaskId(meta);
        return;
      }
      const r = await api.generateModel(req as any);
      setTaskGroup(r.task_id, [r.task_id]);
      setActiveTaskId(r.task_id);
      s.setShowAllZones(false);
      s.setBatchZoneMetaByTaskId({});
    } catch (e: any) {
      setError(e?.message || t("errGen"));
      setGenerating(false);
    }
  };

  // ЗАМОВИТИ ОДРАЗУ: користувач не мусить чекати 1-3 хв перед замовленням.
  // Клік стартує генерацію у фоні (set taskGroupId) і ВІДРАЗУ відкриває форму.
  // Поки юзер заповнює контакти — модель готується; замовлення несе task_id,
  // тож оператор отримає і конфіг, і готову модель (бек приймає order і без
  // завершеної генерації).
  const orderNow = () => {
    if (!selectedArea) { setError(t("errSelectArea")); return; }
    // Замовлення = ПОВНА друкарська якість: оператор має отримати готовий 3MF, а не
    // GLB-прев'ю. Якщо на екрані лише прев'ю (GLB) або нічого — стартуємо повну
    // генерацію у фоні (покупець заповнює контакти, поки 3MF готується; order несе task).
    const needPrint = !downloadUrl || /\.glb(\?|$)/i.test(downloadUrl);
    if (needPrint && !isGenerating) handleGenerate({ forPrint: true });
    setOrderOpen(true);
  };

  const downloadHref = downloadUrl
    ? (downloadUrl.startsWith("http") ? downloadUrl : `${API_BASE}${downloadUrl}`)
    : null;

  return (
    <div className="h-full overflow-y-auto px-4 py-4 sm:px-5">
      <div className="space-y-5 pb-8">
        {/* 1. City */}
        {cityKeys.length > 0 && onCityChange && (
          <div>
            <div className="mb-2 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--text-secondary)]">
              <MapPin size={14} /> {t("step1city")}
            </div>
            <select
              value={selectedCityKey}
              onChange={(e) => { onCityChange(e.target.value); setActiveTemplate(null); }}
              aria-label={t("step1city")}
              title={t("step1city")}
              className="w-full rounded-2xl border border-[var(--surface-border)] bg-white/90 px-4 py-3 text-sm font-semibold text-[var(--text-primary)] outline-none transition focus:border-[rgba(11,92,87,0.35)]"
            >
              {cityKeys.map((k) => (
                <option key={k} value={k}>{k === "Kyiv" ? t("kyiv") : k === "Khmelnytskyi" ? t("khmel") : k}</option>
              ))}
            </select>
          </div>
        )}

        {/* 2. Featured districts */}
        <div>
          <div className="mb-2 flex items-center justify-between">
            <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--text-secondary)]">
              <Sparkles size={14} /> {t("step2districts")}
            </div>
            <span className="text-[11px] text-[var(--text-secondary)]">{t("orDraw")}</span>
          </div>
          {featured.length > 0 ? (
            <div className="grid grid-cols-1 gap-2">
              {featured.map((t) => {
                const active = activeTemplate === t.id;
                return (
                  <button
                    key={t.id}
                    type="button"
                    onClick={() => pickTemplate(t.id)}
                    title={t.blurb}
                    className={`flex items-center gap-2.5 rounded-[16px] border px-2.5 py-2 text-left transition ${
                      active
                        ? "border-[rgba(11,92,87,0.4)] bg-[rgba(15,118,110,0.1)] shadow-[0_10px_24px_rgba(11,92,87,0.14)]"
                        : "border-[var(--surface-border)] bg-white/80 hover:border-[rgba(11,92,87,0.25)]"
                    }`}
                  >
                    <span className={`grid h-9 w-9 shrink-0 place-items-center rounded-[11px] transition ${
                      active ? "bg-[var(--accent-strong)] text-white" : "bg-[rgba(46,74,58,0.08)] text-[var(--accent-strong)]"
                    }`}>
                      <MapPin size={16} className={active ? "" : "opacity-70"} />
                    </span>
                    <span className="flex min-w-0 flex-1 items-center gap-2">
                      <span className="truncate text-sm font-semibold text-[var(--text-primary)]">{t.district}</span>
                      {t.tag && <span className="shrink-0 rounded-full bg-[var(--accent-strong)] px-2 py-0.5 text-[9px] font-bold uppercase tracking-wide text-white">{t.tag}</span>}
                    </span>
                    {active && <Check size={16} className="shrink-0 text-[var(--accent-strong)]" />}
                  </button>
                );
              })}
            </div>
          ) : (
            <div className="rounded-[18px] border border-dashed border-[var(--surface-border)] bg-white/60 px-4 py-4 text-center text-xs text-[var(--text-secondary)]">
              {t("noDistricts")}
            </div>
          )}
          <div className="mt-2 text-[11px] text-[var(--text-secondary)]">
            {selectedArea ? t("areaSelected") : t("areaNotSelected")}
          </div>
        </div>

        {/* 3. Style */}
        <div>
          <div id="simple-style-label" className="mb-2 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--text-secondary)]">
            {t("step3style")}
          </div>
          <div className="grid grid-cols-2 gap-2" role="radiogroup" aria-labelledby="simple-style-label">
            {MAP_STYLE_PRESETS.map((p) => {
              const active = styleId === p.id;
              return (
                <button
                  key={p.id}
                  type="button"
                  role="radio"
                  aria-checked={active}
                  onClick={() => applyStyle(p.id)}
                  title={p.blurb}
                  className={`rounded-[16px] border px-3 py-2.5 text-center text-sm font-semibold transition ${
                    active
                      ? "border-[rgba(11,92,87,0.4)] bg-[rgba(15,118,110,0.1)] text-[var(--text-primary)]"
                      : "border-[var(--surface-border)] bg-white/80 text-[var(--text-primary)] hover:border-[rgba(11,92,87,0.25)]"
                  }`}
                >
                  {p.label}
                </button>
              );
            })}
          </div>
        </div>

        {/* 4. Size */}
        <div>
          <div id="simple-size-label" className="mb-2 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--text-secondary)]">
            {t("step4size")}
          </div>
          <div className="grid grid-cols-4 gap-2" role="radiogroup" aria-labelledby="simple-size-label">
            {SIMPLE_SIZES.map((sz) => {
              const active = Math.abs(modelSizeMm - sz.mm) < 1;
              return (
                <button
                  key={sz.key}
                  type="button"
                  role="radio"
                  aria-checked={active}
                  aria-label={`${sz.label} · ${sz.cm}`}
                  onClick={() => setModelSizeMm(sz.mm)}
                  className={`rounded-[16px] border px-2 py-3 text-center transition ${
                    active
                      ? "border-[rgba(11,92,87,0.4)] bg-[rgba(15,118,110,0.1)]"
                      : "border-[var(--surface-border)] bg-white/80 hover:border-[rgba(11,92,87,0.25)]"
                  }`}
                >
                  <span className="block text-base font-bold text-[var(--text-primary)]">{sz.label}</span>
                  <span className="block text-[11px] text-[var(--text-secondary)]">{sz.cm}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* РЕЛЬЄФ (висоти землі) — окремий перемикач для УСІХ режимів карт. Завжди
            видимий (не ховається під «Більше опцій»). Рельєф = 3D-карта (повний
            пайплайн); тому взаємовиключний з плоскими режимами (AMS/магніт/конектор/
            рамка). Працює зі стандартною картою та панно. */}
        <button
          type="button"
          aria-pressed={reliefMode}
          data-testid="relief-toggle"
          onClick={() => {
            const next = !reliefMode;
            setReliefMode(next);
            if (next) {
              if (flatAmsMode) setFlatAmsMode(false);
              if (magnetMode) setMagnetMode(false);
              if (connectorMode) setConnectorMode(false);
              if (frameMode) setFrameMode(false);
            }
          }}
          className={`w-full rounded-[18px] border px-4 py-3 text-left transition ${
            reliefMode
              ? "border-[rgba(11,92,87,0.4)] bg-[rgba(15,118,110,0.1)]"
              : "border-[var(--surface-border)] bg-white/80 hover:border-[rgba(11,92,87,0.25)]"
          }`}
        >
          <span className="flex items-center justify-between text-sm font-semibold text-[var(--text-primary)]">
            🏔 {t("reliefToggle")}
            {reliefMode && <Check size={16} className="text-[var(--accent-strong)]" />}
          </span>
          <span className="mt-0.5 block text-[11px] leading-4 text-[var(--text-secondary)]">{t("reliefHint")}</span>
        </button>

        {/* Більше опцій — магніт/GPX/панно сховані за замовчанням, щоб Просто-режим
            лишався коротким. Розкривається кліком або авто (якщо щось уже активне). */}
        <div>
          <button
            type="button"
            onClick={() => setMoreOpen((v) => !v)}
            aria-expanded={moreOpen}
            data-testid="more-options"
            className="flex w-full items-center justify-between rounded-[16px] border border-[var(--surface-border)] bg-white/70 px-4 py-2.5 text-[13px] font-semibold transition hover:border-[rgba(11,92,87,0.25)]"
          >
            <span className="flex items-center gap-2 text-[var(--text-secondary)]">
              <Sliders size={14} /> {t("moreOptions")}
              {advancedActive && <span className="h-1.5 w-1.5 rounded-full bg-[var(--accent-strong)]" aria-hidden />}
            </span>
            <ChevronDown size={16} className={`text-[var(--text-secondary)] transition ${moreOpen ? "rotate-180" : ""}`} />
          </button>
        </div>
        {moreOpen && (
        <>
        {/* Плоска кольорова (AMS): пласка багатокольорова плитка-карта — кожен шар
            окремий колір-філамент (Base/Вода/Парки/Дороги/Будинки), міцна основа 3мм.
            БЕЗ рельєфу, БЕЗ з'єднувачів-пазів. Друк плоско = ідеально для Bambu AMS. */}
        <button
          type="button"
          aria-pressed={flatAmsMode}
          data-testid="flat-ams-toggle"
          onClick={() => {
            const next = !flatAmsMode;
            setFlatAmsMode(next);
            // Взаємовиключно з магнітом/панно (різні плоскі формати) і з рельєфом
            // (рельєф = 3D-карта, flat-AMS = плоска).
            if (next) {
              if (magnetMode) setMagnetMode(false);
              if (panelMode > 0) setPanelMode(0);
              if (reliefMode) setReliefMode(false);
            }
          }}
          className={`w-full rounded-[18px] border px-4 py-3 text-left transition ${
            flatAmsMode
              ? "border-[rgba(11,92,87,0.4)] bg-[rgba(15,118,110,0.1)]"
              : "border-[var(--surface-border)] bg-white/80 hover:border-[rgba(11,92,87,0.25)]"
          }`}
        >
          <span className="flex items-center justify-between text-sm font-semibold text-[var(--text-primary)]">
            🎨 {t("flatAmsToggle")}
            {flatAmsMode && <Check size={16} className="text-[var(--accent-strong)]" />}
          </span>
          <span className="mt-0.5 block text-[11px] leading-4 text-[var(--text-secondary)]">{t("flatAmsHint")}</span>
        </button>

        {/* ПЛАСКІ БУДИНКИ (#6): у плоских режимах будинки = тонкі footprint-плити
            одної низької висоти (чистіший AMS-друк) замість лог-блоків. Показуємо
            лише коли активний плоский режим (де це має сенс). Opt-in. */}
        {(flatAmsMode || connectorMode || frameMode || magnetMode) && (
          <label className="flex cursor-pointer items-center justify-between gap-3 rounded-[14px] border border-[var(--surface-border)] bg-white/60 px-4 py-2.5">
            <span className="text-[13px] font-medium text-[var(--text-secondary)]">🏢 {t("flatBuildingsToggle")}</span>
            <input
              type="checkbox"
              data-testid="flat-buildings-toggle"
              checked={flatBuildingsMode}
              onChange={(e) => setFlatBuildingsMode(e.target.checked)}
              className="h-4 w-4 shrink-0 accent-[var(--accent-strong)]"
            />
          </label>
        )}

        {/* З'єднувач-пази (метелик): «ластівчин-хвіст» пази на гранях + окрема
            деталь-ключ, щоб стикувати дві плоскі карти у диптих/панно. Паз у ДНІ
            3мм основи → спереду шов непомітний. Сумісний з flat-AMS (кольорова
            плитка з пазами), несумісний з магнітом/панно (інший формат дна). */}
        <button
          type="button"
          aria-pressed={connectorMode}
          data-testid="connector-toggle"
          onClick={() => {
            const next = !connectorMode;
            setConnectorMode(next);
            if (next) {
              if (magnetMode) setMagnetMode(false);
              if (panelMode > 0) setPanelMode(0);
              if (reliefMode) setReliefMode(false);
            }
          }}
          className={`w-full rounded-[18px] border px-4 py-3 text-left transition ${
            connectorMode
              ? "border-[rgba(11,92,87,0.4)] bg-[rgba(15,118,110,0.1)]"
              : "border-[var(--surface-border)] bg-white/80 hover:border-[rgba(11,92,87,0.25)]"
          }`}
        >
          <span className="flex items-center justify-between text-sm font-semibold text-[var(--text-primary)]">
            🧩 {t("connectorToggle")}
            {connectorMode && <Check size={16} className="text-[var(--accent-strong)]" />}
          </span>
          <span className="mt-0.5 block text-[11px] leading-4 text-[var(--text-secondary)]">{t("connectorHint")}</span>
        </button>

        {/* Преміум-рамка: компас + масштабна лінійка + координати центру окремою
            чорною деталлю поверх плоскої карти. Сумісна з усіма плоскими режимами
            (flat-AMS / з'єднувач / магніт), несумісна з панно (3D-плитки). */}
        <button
          type="button"
          aria-pressed={frameMode}
          data-testid="frame-toggle"
          onClick={() => {
            const next = !frameMode;
            setFrameMode(next);
            if (next && panelMode > 0) setPanelMode(0);
            if (next && reliefMode) setReliefMode(false);
          }}
          className={`w-full rounded-[18px] border px-4 py-3 text-left transition ${
            frameMode
              ? "border-[rgba(11,92,87,0.4)] bg-[rgba(15,118,110,0.1)]"
              : "border-[var(--surface-border)] bg-white/80 hover:border-[rgba(11,92,87,0.25)]"
          }`}
        >
          <span className="flex items-center justify-between text-sm font-semibold text-[var(--text-primary)]">
            🧭 {t("frameToggle")}
            {frameMode && <Check size={16} className="text-[var(--accent-strong)]" />}
          </span>
          <span className="mt-0.5 block text-[11px] leading-4 text-[var(--text-secondary)]">{t("frameHint")}</span>
        </button>

        {/* Магніт: плаский формат 6 см з кишенею під магніт у дні */}
        <button
          type="button"
          aria-pressed={magnetMode}
          onClick={() => {
            const next = !magnetMode;
            setMagnetMode(next);
            if (next && flatAmsMode) setFlatAmsMode(false);
            if (next && connectorMode) setConnectorMode(false);
            if (next && reliefMode) setReliefMode(false);
            // Магніт і панно — взаємовиключні (панно = багато плиток, магніт = одна
            // плитка з кишенею). Раніше можна було лишити обидва ON → генерувалось
            // панно, але ціна показувала магніт (180₴) — мовчазна підміна продукту.
            if (next && panelMode > 0) {
              setPanelMode(0);
              window.dispatchEvent(new CustomEvent("monadruk:toast", { detail: { type: "info", ns: "simple", key: "panelOffForMagnet" } }));
            }
          }}
          className={`w-full rounded-[18px] border px-4 py-3 text-left transition ${
            magnetMode
              ? "border-[rgba(11,92,87,0.4)] bg-[rgba(15,118,110,0.1)]"
              : "border-[var(--surface-border)] bg-white/80 hover:border-[rgba(11,92,87,0.25)]"
          }`}
        >
          <span className="flex items-center justify-between text-sm font-semibold text-[var(--text-primary)]">
            🧲 {t("magnetToggle")}
            {magnetMode && <Check size={16} className="text-[var(--accent-strong)]" />}
          </span>
          <span className="mt-0.5 block text-[11px] leading-4 text-[var(--text-secondary)]">{t("magnetHint")}</span>
        </button>
        {magnetMode && (
          <input
            value={mapLabel}
            onChange={(e) => setMapLabel(e.target.value.toUpperCase().slice(0, 24))}
            placeholder={t("mapLabelPh")}
            className="w-full rounded-[18px] border border-[var(--surface-border)] bg-white/90 px-4 py-3 text-sm font-semibold uppercase tracking-[0.08em] text-[var(--text-primary)] outline-none transition focus:border-[rgba(11,92,87,0.4)]"
          />
        )}

        {/* D4 GPX-трек: маршрут (біг/похід/вело) як підвищений шар поверх мапи */}
        <div className="rounded-[18px] border border-[var(--surface-border)] bg-white/80 px-4 py-3">
          <label className="flex cursor-pointer items-center justify-between gap-3 text-sm font-semibold text-[var(--text-primary)]">
            <span>🏃 {t("gpxUpload")}</span>
            <input
              type="file"
              accept=".gpx,application/gpx+xml"
              data-testid="gpx-input"
              className="hidden"
              onChange={async (e) => {
                const file = e.target.files?.[0];
                e.target.value = "";
                if (!file) return;
                try {
                  const { parseGpx, gpxBounds } = await import("@/lib/gpx");
                  const parsed = parseGpx(await file.text());
                  if (!parsed) { setGpxName(null); setGpxNote(null); setGpxFocus(null); setError(t("gpxErr")); return; }
                  setError(null);
                  // GPX несумісний з панно (трек на одній мапі, не на наборі плиток).
                  if (panelMode > 0) setPanelMode(0);
                  setGpxName(parsed.name || file.name.replace(/\.gpx$/i, ""));
                  // Авто-фокус: зона і карта їдуть до треку (раніше трек з іншого
                  // міста мовчки обрізався по чужій зоні → у моделі його не було).
                  const bb = gpxBounds(parsed.points);
                  if (bb) {
                    const [w, s_, e_, n] = bb;
                    const latC = (s_ + n) / 2;
                    const wM = (e_ - w) * 111320 * Math.max(Math.cos((latC * Math.PI) / 180), 0.2);
                    const hM = (n - s_) * 111320;
                    const spanM = Math.max(wM, hM) * 1.1;
                    // Авто-розмір: найменший пресет, чия зона (мм × 10 м, 1:10000)
                    // покриває трек. Якщо трек більший — НЕ ріжемо його: зона
                    // розширюється до GPX_MAX_M_PER_MM (плоска мапа друкується
                    // шарами, точний масштаб не критичний — дрібніші деталі).
                    const fit = SIMPLE_SIZES.find((sz) => sz.mm * 10 >= spanM);
                    const target = fit ?? SIMPLE_SIZES[SIMPLE_SIZES.length - 1];
                    if (target.mm > (modelSizeMm || 0)) setModelSizeMm(target.mm);
                    if (fit) setGpxNote(t("gpxMoved"));
                    else if (spanM <= target.mm * GPX_MAX_M_PER_MM) {
                      const scale = Math.round((spanM / target.mm) * 1000);
                      setGpxNote(t("gpxScaled", { scale: `1:${scale}` }));
                    } else setGpxNote(t("gpxPartial"));
                    setGpxFocus({ west: w, south: s_, east: e_, north: n, points: parsed.points });
                  }
                } catch { setError(t("gpxErr")); }
              }}
            />
            <span className="rounded-full border border-[var(--surface-border)] bg-white px-3 py-1 text-[12px] font-semibold text-[var(--accent-strong)]">
              {gpxTrack ? t("gpxReplace") : t("gpxChoose")}
            </span>
          </label>
          {gpxTrack ? (
            <div className="mt-2 space-y-1">
              <div className="flex items-center justify-between gap-2 text-[12px] text-[var(--text-secondary)]">
                <span className="truncate">✓ {gpxName} · {gpxTrack.length} {t("gpxPoints")}</span>
                <button type="button" onClick={() => { setGpxName(null); setGpxNote(null); setGpxFocus(null); }} className="shrink-0 font-semibold text-red-700 hover:underline">
                  {t("gpxClear")}
                </button>
              </div>
              {gpxNote && <p data-testid="gpx-note" className="text-[11px] leading-4 font-semibold text-[var(--accent-strong)]">{gpxNote}</p>}
              <p className="text-[11px] leading-4 text-[var(--text-secondary)]">{t("gpxPrivacy")}</p>
            </div>
          ) : (
            <p className="mt-1 text-[11px] leading-4 text-[var(--text-secondary)]">{t("gpxHint")}</p>
          )}
        </div>

        {/* D3 ПАННО: серія зшитих плиток 2×2/3×3 + zip зі схемою розкладки */}
        <div className="rounded-[18px] border border-[var(--surface-border)] bg-white/80 px-4 py-3">
          <div className="flex items-center justify-between gap-2 text-sm font-semibold text-[var(--text-primary)]">
            <span>🖼 {t("panelToggle")}</span>
            <div className="flex gap-1.5" data-testid="panel-chips" role="radiogroup" aria-label={t("panelToggle")}>
              {([[0, t("panelOff")], [2, "2×2"], [3, "3×3"]] as Array<[0 | 2 | 3, string]>).map(([mode, label]) => (
                <button
                  key={`panel-${mode}`}
                  type="button"
                  role="radio"
                  aria-checked={panelMode === mode}
                  onClick={() => {
                    setPanelMode(mode);
                    if (mode > 0 && flatAmsMode) setFlatAmsMode(false);
                    if (mode > 0 && connectorMode) setConnectorMode(false);
                    if (mode > 0 && frameMode) setFrameMode(false);
                    // Панно вимикає магніт + GPX (несумісні: панно = набір повних плиток).
                    if (mode > 0 && (magnetMode || gpxTrack)) {
                      setMagnetMode(false);
                      setGpxName(null); setGpxNote(null); setGpxFocus(null);
                      window.dispatchEvent(new CustomEvent("monadruk:toast", { detail: { type: "info", ns: "simple", key: "magnetOffForPanel" } }));
                    }
                  }}
                  className={`min-h-[36px] rounded-full border px-3.5 py-1.5 text-[12px] font-semibold transition ${
                    panelMode === mode
                      ? "border-[rgba(11,92,87,0.4)] bg-[rgba(15,118,110,0.12)] text-[var(--accent-strong)]"
                      : "border-[var(--surface-border)] bg-white text-[var(--text-secondary)] hover:border-[rgba(11,92,87,0.25)]"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
          <p className="mt-1 text-[11px] leading-4 text-[var(--text-secondary)]">
            {panelMode > 0 ? t("panelHintOn", { tiles: panelMode * panelMode }) : t("panelHint")}
          </p>
        </div>
        </>
        )}

        {/* Generate */}
        <div className="space-y-3">
          {/* Перемикач «Швидке прев'ю / Для друку» прибрано: на екрані завжди
              швидке прев'ю (previewMode=true за замовчанням), а друкарську якість
              оператор генерує при оформленні замовлення. Менше технічних рішень
              для покупця. */}
          <button
            type="button"
            onClick={() => handleGenerate()}
            disabled={!selectedArea || isGenerating}
            className="inline-flex min-h-[52px] w-full items-center justify-center gap-2 rounded-full bg-[var(--accent-strong)] px-5 py-3.5 text-sm font-bold text-white shadow-[0_16px_32px_rgba(11,92,87,0.24)] transition hover:bg-[var(--accent)] disabled:cursor-not-allowed disabled:bg-slate-400"
          >
            {isGenerating ? (
              <><Loader2 className="h-4 w-4 animate-spin" /> {t("generating")} {progress}%</>
            ) : panelMode > 0 ? (
              <><Play className="h-4 w-4" /> {t("generateTiles", { tiles: panelMode * panelMode })}</>
            ) : (
              <><Play className="h-4 w-4" /> {t("generate")}</>
            )}
          </button>
          {/* UX: чесне очікування — час генерації відомий заздалегідь */}
          {!isGenerating && (
            <p className="-mt-1 text-center text-[11px] text-[var(--text-secondary)]">
              {panelMode > 0
                ? t("etaTiles", { tiles: panelMode * panelMode })
                : (flatAmsMode || connectorMode || frameMode || magnetMode)
                  ? t("etaFlat")
                  : t("etaSingle")}
            </p>
          )}
          {isGenerating && (
            <button
              type="button"
              onClick={cancelGeneration}
              className="-mt-1 inline-flex w-full items-center justify-center gap-1 text-[12px] font-semibold text-red-700 underline-offset-2 hover:underline"
            >
              {t("cancelGen")}
            </button>
          )}

          {/* «Замовити друк» — одразу після «Створити», завжди на видному місці */}
          <button
            type="button"
            onClick={() => setOrderOpen(true)}
            className="inline-flex min-h-[52px] w-full items-center justify-center gap-2 rounded-full bg-[var(--bronze,#8E6B3D)] px-5 py-3.5 text-[15px] font-extrabold text-white shadow-[0_16px_34px_rgba(142,107,61,0.32)] transition hover:opacity-90"
          >
            <ShoppingBag className="h-5 w-5" /> {t("orderPrint")} · {orderPriceText}
          </button>

          {error && (
            <div className="rounded-[16px] border border-red-200 bg-red-50 px-4 py-2.5 text-xs text-red-700">
              <p>{error}</p>
              {selectedArea && !isGenerating && (
                <button
                  type="button"
                  onClick={() => handleGenerate()}
                  className="mt-2 inline-flex items-center gap-1 font-semibold text-red-800 underline-offset-2 hover:underline"
                >
                  ↻ {t("retry")}
                </button>
              )}
            </div>
          )}

          {downloadUrl && printQuality && printQuality.status !== "ok" && (printQuality.warnings?.length ?? 0) > 0 && (
            <div className="rounded-[16px] border border-amber-200 bg-amber-50 px-4 py-2.5 text-xs text-amber-900">
              {t("qualityWarn")}
            </div>
          )}

          {downloadUrl && downloadUrl.includes("/download_all") && (
            <div data-testid="panel-howto" className="rounded-[16px] border border-[rgba(11,92,87,0.25)] bg-[rgba(15,118,110,0.07)] px-4 py-2.5 text-xs leading-5 text-[var(--text-primary)]">
              {t("panelHowto", { count: s.taskIds?.length ?? 0 })}
            </div>
          )}

          {downloadUrl && (
            <button
              type="button"
              onClick={doGatedDownload}
              disabled={dlBusy}
              className="inline-flex min-h-12 w-full items-center justify-center gap-2 rounded-full border border-[var(--surface-border)] bg-white px-5 py-3 text-sm font-semibold text-[var(--text-primary)] transition hover:bg-white/70 disabled:opacity-60"
            >
              {dlBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}{" "}
              {printPrep !== null
                ? `${t("generating")} 3MF ${printPrep}%`
                : downloadUrl?.includes("/download_all") ? t("panelZip") : t("downloadFile")}
            </button>
          )}

          {downloadUrl && !downloadUrl.includes("/download_all") && (
            <button
              type="button"
              onClick={doShare}
              disabled={shareBusy}
              className="inline-flex min-h-12 w-full items-center justify-center gap-2 rounded-full border border-[var(--surface-border)] bg-white/70 px-5 py-3 text-sm font-semibold text-[var(--text-primary)] transition hover:bg-white disabled:opacity-60"
            >
              {shareBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <span aria-hidden>🔗</span>}{" "}
              {shareCopied ? t("shareCopied") : t("shareBtn")}
            </button>
          )}

          {downloadUrl && quota && !quota.isAdmin && (
            <div className={`-mt-1 text-center text-[12px] font-medium ${quota.remaining > 0 ? "text-[var(--text-secondary)]" : "text-amber-700"}`}>
              {quota.remaining > 0
                ? t("quotaLeft", { n: quota.remaining, limit: quota.limit })
                : t("quotaExhausted")}
            </div>
          )}

          {onAdvanced && (
            <button
              type="button"
              onClick={onAdvanced}
              className="w-full text-center text-[12px] text-[var(--text-secondary)] underline-offset-2 hover:underline"
            >
              {t("advancedHint")}
            </button>
          )}

        </div>
      </div>

      <OrderDialog
        open={orderOpen}
        onClose={() => setOrderOpen(false)}
        taskId={taskGroupId}
        productType="map"
        priceText={orderPriceText}
        modelPending={!downloadUrl}
        summary={{
          city: selectedCityKey,
          district: MAP_TEMPLATES.find((t) => t.id === activeTemplate)?.district,
          size: panelMode > 0
            ? `${t("panelToggle")} ${panelMode}×${panelMode} (${orderTiles}×)`
            : magnetMode
              ? t("magnetToggle")
              : SIMPLE_SIZES.find((z) => Math.abs(modelSizeMm - z.mm) < 1)?.cm,
        }}
      />

      {/* Мобільний sticky-бар: ціна завжди на екрані + головна дія стану.
          Лише з ОДНІЄЇ копії панелі (showStickyBar) — інакше дубль порталів. */}
      {showStickyBar && (
        <>
          <div className="h-20 lg:hidden" aria-hidden="true" />
          <StickyActionBar
            // Орієнтовна ціна завжди на видноті (фолбек з SIMPLE_SIZES, ніколи «—»)
            // — щоб покупець не тиснув «Замовити» наосліп (головна втрата конверсії).
            priceLabel={t("estPrice")}
            price={orderPriceText}
            actionLabel={
              downloadUrl
                ? t("orderShort")
                : isGenerating
                  ? `${t("generating")} ${progress}%`
                  : t("generateShort") /* короткі лейбли для sticky — щоб поряд із ціною не обрізались */
            }
            busy={isGenerating}
            // НЕ блокуємо коли зона не вибрана — інакше на мобільному тап по єдиній
            // видимій кнопці нічого не робить (глухий кут). Замість цього даємо фідбек.
            disabled={isGenerating}
            onAction={() => {
              // «Замовити» → orderNow (а не голий setOrderOpen): якщо на екрані лише
              // GLB-прев'ю, orderNow стартує ПОВНУ 3MF-генерацію у фоні, щоб оператор
              // отримав друкарський файл, а не чорновик.
              if (downloadUrl) { orderNow(); return; }
              if (!selectedArea) {
                setError(t("errSelectArea"));
                window.dispatchEvent(new CustomEvent("monadruk:toast", { detail: { type: "warn", ns: "simple", key: "errSelectArea" } }));
                return;
              }
              handleGenerate();
            }}
            // Друга дія залежить від стану:
            //  • готово → «Завантажити» поряд із «Замовити»;
            //  • до/під час генерації → «Замовити» (order-now: фонова генерація
            //    + форма одразу, без очікування 1-3 хв).
            secondaryLabel={downloadUrl ? t("downloadShort") : (selectedArea ? t("orderShort") : undefined)}
            onSecondary={downloadUrl ? doGatedDownload : (selectedArea ? orderNow : undefined)}
          />
        </>
      )}
    </div>
  );
}
