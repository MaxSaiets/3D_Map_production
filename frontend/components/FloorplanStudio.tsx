"use client";

/**
 * Студія «макет квартири»: план → перевірка → друкована 3D-модель.
 *
 * Редактор тут — не оздоблення, а суть продукту. Автодетекція стін і масштабу
 * ніколи не буває стовідсотковою, а помилка масштабу невидима на екрані й
 * виявляється лише після того, як виріб надрукували й відправили. Тому між
 * аналізом і генерацією ЗАВЖДИ стоїть людина: вона бачить стіни поверх свого
 * креслення, підтверджує масштаб лінійкою і читає розмір словами.
 *
 * Уся геометрія в цьому файлі — в ПІКСЕЛЯХ прев'ю-зображення (див. lib/floorplan.ts).
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslations } from "next-intl";
import {
  Check, Download, Loader2, MousePointer2, Move3d, Pencil, Redo2,
  Ruler, ShoppingBag, Trash2, Upload, Undo2, DoorOpen, AlertTriangle,
} from "lucide-react";
import {
  distanceToWall, fileToDataUrl, floorplanApi, medianThickness, MODEL_SIZES_MM,
  planBounds, planMetrics, pointAt, snapToAxis, wallLength, WALL_HEIGHTS_M,
  type FpAnalyzeResponse, type FpCapabilities, type FpOpening, type FpPlan, type FpWall,
} from "@/lib/floorplan";
import { api, API_BASE_URL, type TaskStatus } from "@/lib/api";
import { track } from "@/lib/analytics";
import { OrderDialog } from "@/components/OrderDialog";
import { floorplanPriceUah } from "@/lib/mapPrices";

type Tool = "select" | "draw" | "ruler" | "opening";
type Stage = "upload" | "edit" | "result";

interface DragState {
  tool: Tool;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/** Що саме тягнемо в режимі «Обрати»: кінець стіни, всю стіну або отвір. */
type Grab =
  | { kind: "end"; wall: number; which: 0 | 1 }
  | { kind: "wall"; wall: number; startX: number; startY: number; origin: FpWall }
  | { kind: "opening"; index: number };

const WALL_COLOR = "#2f6b46";
const WALL_SELECTED = "#c62626";
const RULER_COLOR = "#c9902f";
const DOOR_COLOR = "#c96a2f";
const WINDOW_COLOR = "#2f7fc9";
const HIT_RADIUS_PX = 12;

export default function FloorplanStudio() {
  const t = useTranslations("maket");

  const [stage, setStage] = useState<Stage>("upload");
  const [caps, setCaps] = useState<FpCapabilities | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [analysis, setAnalysis] = useState<FpAnalyzeResponse | null>(null);
  const [plan, setPlan] = useState<FpPlan | null>(null);
  const [history, setHistory] = useState<FpPlan[]>([]);
  const [future, setFuture] = useState<FpPlan[]>([]);

  const [mPerPx, setMPerPx] = useState(0);
  const [scaleSource, setScaleSource] = useState<string>("assumed");
  const [scaleConfirmed, setScaleConfirmed] = useState(false);

  const [tool, setTool] = useState<Tool>("select");
  const [selected, setSelected] = useState<number | null>(null);
  const [drag, setDrag] = useState<DragState | null>(null);
  const [grab, setGrab] = useState<Grab | null>(null);
  const [rulerLine, setRulerLine] = useState<DragState | null>(null);
  const [rulerInput, setRulerInput] = useState("");
  const [areaInput, setAreaInput] = useState("");
  const [areaError, setAreaError] = useState<string | null>(null);

  const [sizeMm, setSizeMm] = useState<number>(150);
  const [wallHeightM, setWallHeightM] = useState<number>(2.7);
  const [trueScaleHeight, setTrueScaleHeight] = useState(false);

  const [taskId, setTaskId] = useState<string | null>(null);
  const [status, setStatus] = useState<TaskStatus | null>(null);
  const [orderOpen, setOrderOpen] = useState(false);

  const imageRef = useRef<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [viewScale, setViewScale] = useState(1);

  useEffect(() => {
    floorplanApi.capabilities().then(setCaps).catch(() => setCaps(null));
  }, []);

  // ── Історія правок ─────────────────────────────────────────────────────────
  const pushHistory = useCallback((next: FpPlan) => {
    setPlan((current) => {
      if (current) setHistory((h) => [...h.slice(-29), current]);
      setFuture([]);
      return next;
    });
  }, []);

  /** Знімок ПЕРЕД початком перетягування — щоб Ctrl+Z відкочував рух цілком,
   *  а не по одному пікселю. */
  const snapshot = useCallback(() => {
    setPlan((current) => {
      if (current) setHistory((h) => [...h.slice(-29), current]);
      setFuture([]);
      return current;
    });
  }, []);

  const undo = useCallback(() => {
    setHistory((h) => {
      if (!h.length) return h;
      const previous = h[h.length - 1];
      setPlan((current) => {
        if (current) setFuture((f) => [current, ...f.slice(0, 29)]);
        return previous;
      });
      return h.slice(0, -1);
    });
    setSelected(null);
  }, []);

  const redo = useCallback(() => {
    setFuture((f) => {
      if (!f.length) return f;
      const next = f[0];
      setPlan((current) => {
        if (current) setHistory((h) => [...h, current]);
        return next;
      });
      return f.slice(1);
    });
  }, []);

  // ── Аналіз файлу ───────────────────────────────────────────────────────────
  const handleFile = useCallback(async (file: File) => {
    setError(null);
    setBusy(true);
    track("maket_upload", { size_kb: Math.round(file.size / 1024) });
    try {
      const dataUrl = await fileToDataUrl(file);
      const result = await floorplanApi.analyze({ image: dataUrl, filename: file.name });
      setAnalysis(result);
      setPlan(result.plan);
      setHistory([]);
      setFuture([]);
      setMPerPx(result.scale.m_per_px);
      setScaleSource(result.scale.source);
      // «reference» і «pdf» приходять із точного джерела — їх можна не чіпати.
      setScaleConfirmed(result.scale.source === "reference" || result.scale.source === "pdf");
      setStage("edit");
      setTool(result.scale.source === "reference" ? "select" : "ruler");
      track("maket_analyzed", {
        detector: result.detector,
        walls: result.plan.walls.length,
        scale_source: result.scale.source,
      });
    } catch (exception) {
      setError(exception instanceof Error ? exception.message : String(exception));
      track("maket_analyze_failed", {});
    } finally {
      setBusy(false);
    }
  }, []);

  // ── Полотно ────────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!analysis) return;
    const image = new Image();
    image.onload = () => {
      imageRef.current = image;
      draw();
    };
    image.src = analysis.preview;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analysis]);

  useEffect(() => {
    const resize = () => {
      if (!wrapRef.current || !analysis) return;
      const available = wrapRef.current.clientWidth;
      setViewScale(Math.min(1, available / analysis.image_size_px[0]));
    };
    resize();
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, [analysis]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    if (!canvas || !image || !analysis || !plan) return;
    const [width, height] = analysis.image_size_px;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.round(width * viewScale * dpr);
    canvas.height = Math.round(height * viewScale * dpr);
    canvas.style.width = `${width * viewScale}px`;
    canvas.style.height = `${height * viewScale}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr * viewScale, 0, 0, dpr * viewScale, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.globalAlpha = 0.45;                    // креслення просвічує ПІД стінами
    ctx.drawImage(image, 0, 0, width, height);
    ctx.globalAlpha = 1;

    plan.walls.forEach((wall, index) => {
      ctx.strokeStyle = index === selected ? WALL_SELECTED : WALL_COLOR;
      ctx.lineWidth = Math.max(1.5, wall.thickness_m);
      ctx.lineCap = "butt";
      ctx.globalAlpha = index === selected ? 0.95 : 0.72;
      ctx.beginPath();
      ctx.moveTo(wall.x1, wall.y1);
      ctx.lineTo(wall.x2, wall.y2);
      ctx.stroke();
    });
    ctx.globalAlpha = 1;

    plan.openings.forEach((opening) => {
      const wall = plan.walls[opening.wall];
      if (!wall) return;
      const length = wallLength(wall);
      if (length < 1e-6) return;
      const half = opening.width_m / 2 / length;
      const a = pointAt(wall, Math.max(0, opening.center_t - half));
      const b = pointAt(wall, Math.min(1, opening.center_t + half));
      ctx.strokeStyle = opening.kind === "window" ? WINDOW_COLOR : DOOR_COLOR;
      ctx.lineWidth = Math.max(2.5, wall.thickness_m + 2);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    });

    // Маркери на кінцях обраної стіни: без них не видно, що її можна тягнути.
    if (tool === "select" && selected !== null && plan.walls[selected]) {
      const wall = plan.walls[selected];
      const handle = Math.max(4, 7 / viewScale);
      ([[wall.x1, wall.y1], [wall.x2, wall.y2]] as const).forEach(([hx, hy]) => {
        ctx.beginPath();
        ctx.arc(hx, hy, handle, 0, Math.PI * 2);
        ctx.fillStyle = "#ffffff";
        ctx.fill();
        ctx.lineWidth = Math.max(1.5, 2 / viewScale);
        ctx.strokeStyle = WALL_SELECTED;
        ctx.stroke();
      });
    }

    if (drag && drag.tool === "draw") {
      ctx.strokeStyle = WALL_COLOR;
      ctx.setLineDash([6, 4]);
      ctx.lineWidth = Math.max(2, medianThickness(plan));
      ctx.beginPath();
      ctx.moveTo(drag.x1, drag.y1);
      ctx.lineTo(drag.x2, drag.y2);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    const ruler = drag?.tool === "ruler" ? drag : rulerLine;
    if (ruler) {
      ctx.strokeStyle = RULER_COLOR;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(ruler.x1, ruler.y1);
      ctx.lineTo(ruler.x2, ruler.y2);
      ctx.stroke();
      [[ruler.x1, ruler.y1], [ruler.x2, ruler.y2]].forEach(([x, y]) => {
        ctx.fillStyle = RULER_COLOR;
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fill();
      });
    }
  }, [analysis, plan, selected, drag, rulerLine, viewScale, tool]);

  useEffect(() => { draw(); }, [draw]);

  const toImageCoords = useCallback((event: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left) / viewScale,
      y: (event.clientY - rect.top) / viewScale,
    };
  }, [viewScale]);

  const hitWall = useCallback((x: number, y: number): number | null => {
    if (!plan) return null;
    let best: number | null = null;
    let bestDistance = HIT_RADIUS_PX / viewScale;
    plan.walls.forEach((wall, index) => {
      const { distance } = distanceToWall(wall, x, y);
      const tolerance = Math.max(bestDistance, wall.thickness_m * 0.75);
      if (distance <= tolerance && distance < bestDistance + wall.thickness_m) {
        bestDistance = distance;
        best = index;
      }
    });
    return best;
  }, [plan, viewScale]);

  const onPointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!plan) return;
    const { x, y } = toImageCoords(event);
    // setPointerCapture кидає NotFoundError, якщо вказівник уже неактивний
    // (буває на дотикових екранах і в автотестах). Без try/catch виняток
    // обривав увесь обробник — стіна не бралась, і клік просто «не працював».
    try {
      event.currentTarget.setPointerCapture(event.pointerId);
    } catch {
      /* захоплення необов'язкове: рух ловимо через pointermove на самому canvas */
    }

    if (tool === "select") {
      // Пріоритет захоплення: отвір → кінець стіни → тіло стіни. Кінці й отвори
      // дрібніші за стіну, тож якби стіна перехоплювала клік першою, підправити
      // їх було б неможливо.
      const grabRadius = HIT_RADIUS_PX / viewScale;
      const openingIndex = plan.openings.findIndex((opening) => {
        const wall = plan.walls[opening.wall];
        if (!wall) return false;
        const point = pointAt(wall, opening.center_t);
        return Math.hypot(point.x - x, point.y - y) <= grabRadius;
      });
      if (openingIndex >= 0) {
        snapshot();
        setSelected(plan.openings[openingIndex].wall);
        setGrab({ kind: "opening", index: openingIndex });
        return;
      }
      let endGrab: Grab | null = null;
      plan.walls.forEach((wall, index) => {
        ([[wall.x1, wall.y1, 0], [wall.x2, wall.y2, 1]] as const).forEach(([wx, wy, which]) => {
          if (!endGrab && Math.hypot(wx - x, wy - y) <= grabRadius) {
            endGrab = { kind: "end", wall: index, which: which as 0 | 1 };
          }
        });
      });
      if (endGrab) {
        snapshot();
        setSelected((endGrab as { wall: number }).wall);
        setGrab(endGrab);
        return;
      }
      const index = hitWall(x, y);
      setSelected(index);
      if (index !== null) {
        snapshot();
        setGrab({ kind: "wall", wall: index, startX: x, startY: y, origin: { ...plan.walls[index] } });
      }
      return;
    }
    if (tool === "opening") {
      const index = hitWall(x, y);
      if (index === null) return;
      const wall = plan.walls[index];
      const { t: position } = distanceToWall(wall, x, y);
      const existing = plan.openings.findIndex((opening) => {
        if (opening.wall !== index) return false;
        const halfT = opening.width_m / 2 / Math.max(1, wallLength(wall));
        return Math.abs(opening.center_t - position) <= halfT + 0.02;
      });
      const next: FpPlan = { ...plan, openings: [...plan.openings] };
      if (existing >= 0) {
        // цикл: двері → вікно → прибрати
        const current = next.openings[existing];
        if (current.kind === "door") {
          next.openings[existing] = { ...current, kind: "window", sill_m: 0.85, height_m: 1.45 };
        } else {
          next.openings.splice(existing, 1);
        }
      } else {
        const defaultWidth = Math.min(
          wallLength(wall) * 0.5,
          Math.max(8, 0.85 / Math.max(mPerPx, 1e-6)),
        );
        next.openings.push({
          wall: index, center_t: position, width_m: defaultWidth,
          kind: "door", sill_m: 0, height_m: 2.1,
        });
      }
      pushHistory(next);
      return;
    }
    setDrag({ tool, x1: x, y1: y, x2: x, y2: y });
  };

  const onPointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const { x, y } = toImageCoords(event);

    if (grab && plan) {
      // Правки застосовуємо БЕЗ запису в історію на кожен піксель — інакше одне
      // перетягування з'їдало б увесь стек скасувань. Знімок кладемо на
      // pointerDown, а тут лише оновлюємо геометрію.
      const walls = plan.walls.map((w) => ({ ...w }));
      const openings = plan.openings.map((o) => ({ ...o }));
      if (grab.kind === "end") {
        const wall = walls[grab.wall];
        if (!wall) return;
        const anchor = grab.which === 0 ? { x: wall.x2, y: wall.y2 } : { x: wall.x1, y: wall.y1 };
        const snapped = snapToAxis(anchor.x, anchor.y, x, y, !event.shiftKey);
        if (grab.which === 0) { wall.x1 = snapped.x2; wall.y1 = snapped.y2; }
        else { wall.x2 = snapped.x2; wall.y2 = snapped.y2; }
      } else if (grab.kind === "wall") {
        const wall = walls[grab.wall];
        if (!wall) return;
        const dx = x - grab.startX;
        const dy = y - grab.startY;
        wall.x1 = grab.origin.x1 + dx;
        wall.y1 = grab.origin.y1 + dy;
        wall.x2 = grab.origin.x2 + dx;
        wall.y2 = grab.origin.y2 + dy;
      } else {
        const opening = openings[grab.index];
        const wall = walls[opening?.wall];
        if (!opening || !wall) return;
        const { t } = distanceToWall(wall, x, y);
        const half = opening.width_m / 2 / Math.max(1, wallLength(wall));
        opening.center_t = Math.min(Math.max(t, half), 1 - half);
      }
      setPlan({ ...plan, walls, openings });
      return;
    }

    if (!drag) return;
    const snapped = drag.tool === "draw"
      ? snapToAxis(drag.x1, drag.y1, x, y, !event.shiftKey)
      : { x2: x, y2: y };
    setDrag({ ...drag, x2: snapped.x2, y2: snapped.y2 });
  };

  const onPointerUp = () => {
    if (grab) {
      setGrab(null);
      track("maket_wall_moved", { kind: grab.kind });
      return;
    }
    if (!drag || !plan) { setDrag(null); return; }
    const length = Math.hypot(drag.x2 - drag.x1, drag.y2 - drag.y1);
    if (drag.tool === "draw") {
      if (length > 6) {
        const wall: FpWall = {
          x1: drag.x1, y1: drag.y1, x2: drag.x2, y2: drag.y2,
          thickness_m: medianThickness(plan), bearing: false,
        };
        pushHistory({ ...plan, walls: [...plan.walls, wall] });
        track("maket_wall_added", {});
      }
    } else if (drag.tool === "ruler" && length > 8) {
      setRulerLine({ ...drag });
      setRulerInput("");
    }
    setDrag(null);
  };

  const deleteSelected = useCallback(() => {
    if (!plan || selected === null) return;
    const walls = plan.walls.filter((_, index) => index !== selected);
    const openings = plan.openings
      .filter((opening) => opening.wall !== selected)
      .map((opening) => ({
        ...opening,
        wall: opening.wall > selected ? opening.wall - 1 : opening.wall,
      }));
    pushHistory({ ...plan, walls, openings });
    setSelected(null);
    track("maket_wall_deleted", {});
  }, [plan, selected, pushHistory]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) return;
      if (event.key === "Delete" || event.key === "Backspace") { deleteSelected(); }
      else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
        event.preventDefault();
        event.shiftKey ? redo() : undo();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [deleteSelected, undo, redo]);

  /** Масштаб із загальної площі: m/px = √(площа_м² / площа_приміщень_px²).
   *  Заміряно на синтетиці — 2.2% середньої похибки, тобто вдвічі точніше за
   *  оцінку по ширині дверей. */
  /** Повернути план у стан «як розпізнала автоматика». Правки бувають невдалі,
   *  і без цієї кнопки єдиний вихід — перезавантажувати файл наново. */
  const resetToAuto = useCallback(() => {
    if (!analysis) return;
    pushHistory(JSON.parse(JSON.stringify(analysis.plan)) as FpPlan);
    setSelected(null);
    track("maket_reset_auto", {});
  }, [analysis, pushHistory]);

  const applyArea = useCallback(() => {
    const area = parseFloat(areaInput.replace(",", "."));
    const interior = analysis?.estimate.interior_px2 ?? 0;
    // Мовчазна відмова тут була найгіршим станом: користувач вписує площу з
    // договору, тисне «Застосувати» — і НІЧОГО не змінюється, без пояснення.
    if (!Number.isFinite(area) || area < 5 || area > 600) {
      setAreaError(t("areaOutOfRange"));
      return;
    }
    if (interior < 1) {
      setAreaError(t("areaUnavailable"));
      return;
    }
    setAreaError(null);
    setMPerPx(Math.sqrt(area / interior));
    setScaleSource("area");
    setScaleConfirmed(true);
    setAreaInput("");
    track("maket_scale_set", { source: "area" });
  }, [areaInput, analysis, t]);

  const applyRuler = useCallback(() => {
    if (!rulerLine) return;
    const meters = parseFloat(rulerInput.replace(",", "."));
    if (!Number.isFinite(meters) || meters <= 0.05 || meters > 200) return;
    const pixels = Math.hypot(rulerLine.x2 - rulerLine.x1, rulerLine.y2 - rulerLine.y1);
    if (pixels < 4) return;
    setMPerPx(meters / pixels);
    setScaleSource("reference");
    setScaleConfirmed(true);
    setRulerLine(null);
    setTool("select");
    track("maket_scale_set", { source: "reference" });
  }, [rulerLine, rulerInput]);

  // ── Генерація ──────────────────────────────────────────────────────────────
  const metrics = useMemo(() => {
    if (!plan || mPerPx <= 0) return null;
    const base = planMetrics(plan, mPerPx);
    // Показуємо площу ПРИМІЩЕНЬ, а не габаритного прямокутника. Користувач
    // звіряє її з договором: побачити «124 м²» після того, як сам вписав 68,
    // означає вирішити, що сервіс помилився. Габарит лишається окремо — він
    // потрібен, щоб оцінити, чи влізе виріб на стіл.
    const interior = analysis?.estimate.interior_px2 ?? 0;
    const area = interior > 1 ? interior * mPerPx * mPerPx : base.area;
    return { ...base, area, bboxArea: base.area };
  }, [plan, mPerPx, analysis]);

  const scaleDenominator = useMemo(() => {
    if (!plan || !metrics || metrics.width <= 0) return 0;
    const { minX, minY, maxX, maxY } = planBounds(plan);
    const spanM = Math.max(maxX - minX, maxY - minY) * mPerPx;
    if (spanM <= 0) return 0;
    return (spanM * 1000) / sizeMm;
  }, [plan, metrics, mPerPx, sizeMm]);

  const areaProblem = useMemo(() => {
    if (!metrics) return null;
    if (metrics.area < 8) return t("areaTooSmall");
    if (metrics.area > 400) return t("areaTooBig");
    return null;
  }, [metrics, t]);

  const canGenerate = Boolean(
    plan && plan.walls.length && mPerPx > 0 && scaleConfirmed && !areaProblem && !busy,
  );

  const generate = useCallback(async () => {
    if (!plan || !canGenerate) return;
    setBusy(true);
    setError(null);
    try {
      const response = await floorplanApi.generate({
        plan, m_per_px: mPerPx, model_size_mm: sizeMm,
        wall_height_mode: trueScaleHeight ? "true_scale" : "maquette",
        wall_height_m: wallHeightM,
      });
      setTaskId(response.task_id);
      setStage("result");
      track("maket_generate", {
        size_mm: sizeMm, walls: plan.walls.length,
        area_m2: metrics ? Math.round(metrics.area) : 0,
      });
    } catch (exception) {
      setError(exception instanceof Error ? exception.message : String(exception));
    } finally {
      setBusy(false);
    }
  }, [plan, canGenerate, mPerPx, sizeMm, trueScaleHeight, wallHeightM, metrics]);

  useEffect(() => {
    if (!taskId) return;
    let cancelled = false;
    let failures = 0;
    const tick = async () => {
      try {
        const result = (await api.getStatus(taskId)) as TaskStatus;
        if (cancelled) return;
        setStatus(result);
        failures = 0;
        if (result.status === "completed") {
          track("maket_ready", { task: taskId });
          return;
        }
        if (result.status === "failed") {
          setError(result.message || t("genFailed"));
          return;
        }
      } catch {
        failures += 1;
        if (failures >= 4) {
          setError(t("genFailed"));
          return;
        }
      }
      if (!cancelled) window.setTimeout(tick, 2500);
    };
    void tick();
    return () => { cancelled = true; };
  }, [taskId, t]);

  const glbUrl = useMemo(() => {
    const raw = (status as unknown as { download_url_glb?: string })?.download_url_glb;
    return raw ? `${API_BASE_URL}${raw}` : null;
  }, [status]);

  const download = useCallback(async (format: "3mf" | "stl") => {
    if (!taskId) return;
    const blob = await api.downloadModel(taskId, format);
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `maket.${format}`;
    anchor.click();
    URL.revokeObjectURL(url);
    track("maket_download", { format });
  }, [taskId]);

  // ═════════════════════════════════════════════════════════════════════════
  const scaleLabel = t(`scaleSource.${scaleSource}` as never, {}) as string;

  return (
    <div className="mx-auto w-full max-w-[1180px] px-4 pb-24 pt-6">
      {error && (
        <div role="alert" className="mb-4 flex items-start gap-2 rounded-2xl border border-[#e0b4b4] bg-[#fdf3f3] px-4 py-3 text-[14px] text-[#8a2b2b]">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {stage === "upload" && (
        <section className="rounded-[28px] border border-[var(--line-soft,#e3e0d5)] bg-white/70 p-8 text-center">
          <h2 className="text-[22px] font-semibold text-[var(--text-primary,#1c2320)]">{t("uploadTitle")}</h2>
          <p className="mx-auto mt-2 max-w-[560px] text-[15px] leading-relaxed text-[var(--text-secondary,#5a655a)]">
            {t("uploadHint")}
          </p>
          <label className="mt-6 inline-flex cursor-pointer items-center gap-2 rounded-full bg-[var(--forest,#2f6b46)] px-7 py-3 text-[15px] font-medium text-white transition hover:opacity-90">
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
            {busy ? t("analyzing") : t("chooseFile")}
            <input
              type="file"
              accept="image/png,image/jpeg,image/webp,application/pdf"
              className="hidden"
              disabled={busy}
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) void handleFile(file);
                event.target.value = "";
              }}
            />
          </label>
          <ul className="mx-auto mt-6 max-w-[520px] space-y-1 text-left text-[13px] text-[var(--text-secondary,#5a655a)]">
            <li>• {t("tipStraight")}</li>
            <li>• {t("tipWhole")}</li>
            <li>• {t("tipDimension")}</li>
          </ul>
          {caps && !caps.neural_detector && (
            <p className="mt-4 text-[12px] text-[var(--text-secondary,#5a655a)]">{t("cvOnly")}</p>
          )}
        </section>
      )}

      {stage === "edit" && plan && analysis && (
        <div className="grid gap-6 lg:grid-cols-[1fr_340px]">
          <div>
            <div className="mb-3 flex flex-wrap items-center gap-2">
              {([
                ["select", MousePointer2, t("toolSelect")],
                ["draw", Pencil, t("toolDraw")],
                ["opening", DoorOpen, t("toolOpening")],
                ["ruler", Ruler, t("toolRuler")],
              ] as const).map(([id, Icon, label]) => (
                <button
                  key={id}
                  type="button"
                  role="radio"
                  aria-checked={tool === id}
                  onClick={() => { setTool(id as Tool); setSelected(null); }}
                  className={`inline-flex items-center gap-1.5 rounded-full border px-3.5 py-2 text-[13px] transition ${
                    tool === id
                      ? "border-[var(--forest,#2f6b46)] bg-[var(--forest,#2f6b46)] text-white"
                      : "border-[var(--line-soft,#e3e0d5)] bg-white/70 text-[var(--text-secondary,#5a655a)]"
                  }`}
                >
                  <Icon className="h-3.5 w-3.5" />
                  {label}
                </button>
              ))}
              <span className="mx-1 h-5 w-px bg-[var(--line-soft,#e3e0d5)]" />
              <button type="button" onClick={undo} disabled={!history.length}
                className="rounded-full border border-[var(--line-soft,#e3e0d5)] bg-white/70 p-2 disabled:opacity-40"
                aria-label={t("undo")}>
                <Undo2 className="h-4 w-4" />
              </button>
              <button type="button" onClick={redo} disabled={!future.length}
                className="rounded-full border border-[var(--line-soft,#e3e0d5)] bg-white/70 p-2 disabled:opacity-40"
                aria-label={t("redo")}>
                <Redo2 className="h-4 w-4" />
              </button>
              <button type="button" onClick={deleteSelected} disabled={selected === null}
                className="rounded-full border border-[var(--line-soft,#e3e0d5)] bg-white/70 p-2 disabled:opacity-40"
                aria-label={t("deleteWall")}>
                <Trash2 className="h-4 w-4" />
              </button>
              <button type="button" onClick={resetToAuto} disabled={!history.length}
                className="rounded-full border border-[var(--line-soft,#e3e0d5)] bg-white/70 px-3 py-2 text-[12px] disabled:opacity-40">
                {t("resetAuto")}
              </button>
            </div>

            <div ref={wrapRef} className="overflow-hidden rounded-[20px] border border-[var(--line-soft,#e3e0d5)] bg-white">
              <canvas
                ref={canvasRef}
                onPointerDown={onPointerDown}
                onPointerMove={onPointerMove}
                onPointerUp={onPointerUp}
                className="block touch-none"
                style={{ cursor: tool === "select" ? "pointer" : "crosshair" }}
              />
            </div>
            <p className="mt-2 text-[12px] text-[var(--text-secondary,#5a655a)]">{t(`toolHint.${tool}` as never)}</p>

            {rulerLine && (
              <div className="mt-3 flex flex-wrap items-center gap-2 rounded-2xl border border-[#e4d2a8] bg-[#fdf8ec] px-4 py-3">
                <span className="text-[14px] text-[var(--text-primary,#1c2320)]">{t("rulerAsk")}</span>
                <input
                  autoFocus
                  inputMode="decimal"
                  value={rulerInput}
                  onChange={(event) => setRulerInput(event.target.value)}
                  onKeyDown={(event) => { if (event.key === "Enter") applyRuler(); }}
                  placeholder="5.2"
                  aria-label={t("rulerAsk")}
                  className="w-24 rounded-lg border border-[var(--line-soft,#e3e0d5)] px-3 py-1.5 text-[14px]"
                />
                <span className="text-[14px] text-[var(--text-secondary,#5a655a)]">{t("meters")}</span>
                <button type="button" onClick={applyRuler}
                  className="rounded-full bg-[var(--forest,#2f6b46)] px-4 py-1.5 text-[13px] font-medium text-white">
                  {t("apply")}
                </button>
                <button type="button" onClick={() => setRulerLine(null)}
                  className="rounded-full border border-[var(--line-soft,#e3e0d5)] px-4 py-1.5 text-[13px]">
                  {t("cancel")}
                </button>
              </div>
            )}
          </div>

          <aside className="space-y-5">
            <div className="rounded-[20px] border border-[var(--line-soft,#e3e0d5)] bg-white/70 p-4">
              <h3 className="text-[15px] font-semibold text-[var(--text-primary,#1c2320)]">{t("scaleTitle")}</h3>
              <p className="mt-1 text-[12px] text-[var(--text-secondary,#5a655a)]">{scaleLabel}</p>
              {metrics && (
                <p className="mt-3 text-[14px] leading-relaxed text-[var(--text-primary,#1c2320)]">
                  {t("sizeSentence", {
                    w: metrics.width.toFixed(1),
                    h: metrics.height.toFixed(1),
                    a: Math.round(metrics.area),
                  })}
                </p>
              )}
              {areaProblem && (
                <p className="mt-2 text-[13px] font-medium text-[#8a2b2b]">{areaProblem}</p>
              )}
              {!scaleConfirmed ? (
                <div className="mt-3 space-y-2">
                  <button type="button" onClick={() => setTool("ruler")}
                    className="w-full rounded-full border border-[var(--forest,#2f6b46)] px-4 py-2 text-[13px] font-medium text-[var(--forest,#2f6b46)]">
                    {t("setScaleWithRuler")}
                  </button>
                  {/* Площа з договору — найзручніший спосіб для українця: розміри
                      на скані можуть не читатись, а «61,4 м²» знає кожен. */}
                  <div className="flex items-center gap-2">
                    <input
                      inputMode="decimal"
                      value={areaInput}
                      onChange={(event) => { setAreaInput(event.target.value); setAreaError(null); }}
                      onKeyDown={(event) => { if (event.key === "Enter") applyArea(); }}
                      placeholder={t("areaPlaceholder")}
                      aria-label={t("areaLabel")}
                      className="w-full rounded-lg border border-[var(--line-soft,#e3e0d5)] px-3 py-2 text-[13px]"
                    />
                    <button type="button" onClick={applyArea} disabled={!areaInput.trim()}
                      className="shrink-0 rounded-full border border-[var(--line-soft,#e3e0d5)] px-3 py-2 text-[13px] disabled:opacity-40">
                      {t("apply")}
                    </button>
                  </div>
                  {areaError && (
                    <p className="text-[12px] font-medium text-[#8a2b2b]">{areaError}</p>
                  )}
                  <button type="button" onClick={() => { setScaleConfirmed(true); track("maket_scale_accepted", { source: scaleSource }); }}
                    className="w-full rounded-full bg-[var(--forest,#2f6b46)] px-4 py-2 text-[13px] font-medium text-white">
                    {t("confirmScale")}
                  </button>
                </div>
              ) : (
                <p className="mt-3 inline-flex items-center gap-1.5 text-[13px] font-medium text-[var(--forest,#2f6b46)]">
                  <Check className="h-4 w-4" /> {t("scaleConfirmed")}
                </p>
              )}
            </div>

            <div className="rounded-[20px] border border-[var(--line-soft,#e3e0d5)] bg-white/70 p-4">
              <h3 className="text-[15px] font-semibold text-[var(--text-primary,#1c2320)]">{t("sizeTitle")}</h3>
              <div className="mt-3 grid grid-cols-2 gap-2" role="radiogroup" aria-label={t("sizeTitle")}>
                {MODEL_SIZES_MM.map((size) => (
                  <button
                    key={size}
                    type="button"
                    role="radio"
                    aria-checked={sizeMm === size}
                    onClick={() => setSizeMm(size)}
                    className={`rounded-xl border px-3 py-2 text-[13px] transition ${
                      sizeMm === size
                        ? "border-[var(--forest,#2f6b46)] bg-[var(--forest,#2f6b46)]/10 font-medium text-[var(--forest,#2f6b46)]"
                        : "border-[var(--line-soft,#e3e0d5)] text-[var(--text-secondary,#5a655a)]"
                    }`}
                  >
                    {size / 10} {t("cm")}
                  </button>
                ))}
              </div>
              {scaleDenominator > 0 && (
                <p className="mt-2 text-[12px] text-[var(--text-secondary,#5a655a)]">
                  {t("approxScale", { n: Math.round(scaleDenominator) })}
                </p>
              )}
            </div>

            <div className="rounded-[20px] border border-[var(--line-soft,#e3e0d5)] bg-white/70 p-4">
              <h3 className="text-[15px] font-semibold text-[var(--text-primary,#1c2320)]">{t("heightTitle")}</h3>
              <div className="mt-3 flex flex-wrap gap-2" role="radiogroup" aria-label={t("heightTitle")}>
                {WALL_HEIGHTS_M.map((height) => (
                  <button
                    key={height}
                    type="button"
                    role="radio"
                    aria-checked={wallHeightM === height}
                    onClick={() => setWallHeightM(height)}
                    className={`rounded-full border px-3 py-1.5 text-[13px] ${
                      wallHeightM === height
                        ? "border-[var(--forest,#2f6b46)] bg-[var(--forest,#2f6b46)]/10 font-medium text-[var(--forest,#2f6b46)]"
                        : "border-[var(--line-soft,#e3e0d5)] text-[var(--text-secondary,#5a655a)]"
                    }`}
                  >
                    {height.toFixed(1)} {t("meters")}
                  </button>
                ))}
              </div>
              <label className="mt-3 flex items-start gap-2 text-[12px] text-[var(--text-secondary,#5a655a)]">
                <input type="checkbox" checked={trueScaleHeight}
                  onChange={(event) => setTrueScaleHeight(event.target.checked)} className="mt-0.5" />
                <span>{t("trueScaleHeight")}</span>
              </label>
            </div>

            <button
              type="button"
              onClick={generate}
              disabled={!canGenerate}
              className="flex w-full items-center justify-center gap-2 rounded-full bg-[var(--forest,#2f6b46)] px-6 py-3.5 text-[15px] font-medium text-white transition hover:opacity-90 disabled:opacity-40"
            >
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Move3d className="h-4 w-4" />}
              {t("generate")}
            </button>
            {!scaleConfirmed && (
              <p className="text-center text-[12px] text-[var(--text-secondary,#5a655a)]">{t("confirmScaleFirst")}</p>
            )}
          </aside>
        </div>
      )}

      {stage === "result" && (
        <section className="rounded-[28px] border border-[var(--line-soft,#e3e0d5)] bg-white/70 p-6">
          {status?.status !== "completed" ? (
            <div className="py-16 text-center">
              <Loader2 className="mx-auto h-8 w-8 animate-spin text-[var(--forest,#2f6b46)]" />
              <p className="mt-4 text-[15px] text-[var(--text-primary,#1c2320)]">
                {status?.message || t("building")}
              </p>
              <div className="mx-auto mt-4 h-1.5 w-64 overflow-hidden rounded-full bg-[var(--line-soft,#e3e0d5)]">
                <div className="h-full rounded-full bg-[var(--forest,#2f6b46)] transition-all"
                  style={{ width: `${status?.progress ?? 5}%` }} />
              </div>
            </div>
          ) : (
            <div className="space-y-5">
              <h2 className="text-center text-[20px] font-semibold text-[var(--text-primary,#1c2320)]">
                {t("readyTitle")}
              </h2>
              <p className="text-center text-[14px] text-[var(--text-secondary,#5a655a)]">{status.message}</p>
              {glbUrl && <ModelPreview url={glbUrl} />}
              {status.print_quality?.warnings?.length ? (
                <ul className="mx-auto max-w-[640px] space-y-1 rounded-2xl bg-[#fdf8ec] px-4 py-3 text-[13px] text-[#7a5a1e]">
                  {status.print_quality.warnings.map((warning, index) => (
                    <li key={index}>• {warning}</li>
                  ))}
                </ul>
              ) : null}
              <div className="flex flex-wrap items-center justify-center gap-3">
                <button type="button" onClick={() => download("3mf")}
                  className="inline-flex items-center gap-2 rounded-full bg-[var(--forest,#2f6b46)] px-6 py-3 text-[15px] font-medium text-white">
                  <Download className="h-4 w-4" /> {t("download3mf")}
                </button>
                <button type="button" onClick={() => download("stl")}
                  className="inline-flex items-center gap-2 rounded-full border border-[var(--line-soft,#e3e0d5)] px-6 py-3 text-[15px]">
                  <Download className="h-4 w-4" /> STL
                </button>
                <button type="button"
                  onClick={() => { setOrderOpen(true); track("maket_order_click", { size_mm: sizeMm }); }}
                  className="inline-flex items-center gap-2 rounded-full border border-[var(--bronze,#c9902f)] px-6 py-3 text-[15px] font-medium text-[var(--bronze,#c9902f)]">
                  <ShoppingBag className="h-4 w-4" /> {t("orderPrint")}
                </button>
              </div>
              <div className="text-center">
                <button type="button"
                  onClick={() => { setStage("edit"); setTaskId(null); setStatus(null); }}
                  className="text-[13px] underline text-[var(--text-secondary,#5a655a)]">
                  {t("backToEdit")}
                </button>
              </div>
            </div>
          )}
        </section>
      )}

      <OrderDialog
        open={orderOpen}
        onClose={() => setOrderOpen(false)}
        taskId={taskId}
        productType="floorplan"
        priceText={`${floorplanPriceUah(sizeMm)} ₴`}
        summary={{
          size: `${sizeMm / 10} ${t("cm")}`,
          city: metrics ? `${metrics.width.toFixed(1)}×${metrics.height.toFixed(1)} м` : undefined,
        }}
        modelPending={status?.status !== "completed"}
      />
    </div>
  );
}

/** Three.js важкий — вантажимо лише коли модель справді готова. */
function ModelPreview({ url }: { url: string }) {
  const [Viewer, setViewer] = useState<React.ComponentType<{ url: string; height?: number; allowZoom?: boolean }> | null>(null);
  useEffect(() => {
    let alive = true;
    import("@/components/Model3DViewer")
      .then((module) => { if (alive) setViewer(() => module.default); })
      .catch(() => undefined);
    return () => { alive = false; };
  }, []);
  if (!Viewer) return <div className="h-[360px] rounded-[20px] bg-[#0f172a]" />;
  return <Viewer url={url} height={380} allowZoom />;
}
