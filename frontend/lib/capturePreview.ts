// Capture the on-screen product preview as PNG data-URLs.
//
// Used for (a) order screenshots sent to the Telegram CRM so the operator sees
// EXACTLY what the customer designed (text, layout, rotation) before printing,
// and (b) saving a thumbnail with the model in the account history.
//
// The keychain designer is a self-contained inline <svg> (no <foreignObject>,
// no external images — the live city is native SVG paths), so it serializes to
// PNG cleanly across browsers. The 3D model preview is a <canvas> we also grab.

/** Serialize one SVG element to a PNG data-URL. Returns null on failure. */
export async function svgToPngDataUrl(
  svg: SVGSVGElement,
  opts: { scale?: number; background?: string; maxWidth?: number } = {},
): Promise<string | null> {
  const background = opts.background ?? "#050a18";
  try {
    const rect = svg.getBoundingClientRect();
    const rw = Math.max(Math.round(rect.width) || 0, 1);
    const rh = Math.max(Math.round(rect.height) || 0, 1);
    // Effective scale: honour `scale`, but if a maxWidth is given (thumbnails),
    // downscale so the output never exceeds it — keeps the data-URL small enough
    // for the account-history store (backend caps the field length).
    let scale = opts.scale ?? 2;
    if (opts.maxWidth && rw * scale > opts.maxWidth) scale = opts.maxWidth / rw;
    const w = rw;
    const h = rh;
    // Clone so we can pin explicit width/height (some browsers need it to raster).
    const clone = svg.cloneNode(true) as SVGSVGElement;
    clone.setAttribute("width", String(w));
    clone.setAttribute("height", String(h));
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    const xml = new XMLSerializer().serializeToString(clone);
    // unicode-safe base64 (label may contain Cyrillic)
    const svg64 = "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(xml)));

    return await new Promise<string | null>((resolve) => {
      const img = new Image();
      img.onload = () => {
        try {
          const canvas = document.createElement("canvas");
          canvas.width = w * scale;
          canvas.height = h * scale;
          const ctx = canvas.getContext("2d");
          if (!ctx) return resolve(null);
          ctx.fillStyle = background;
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
          resolve(canvas.toDataURL("image/png"));
        } catch {
          resolve(null);
        }
      };
      img.onerror = () => resolve(null);
      img.src = svg64;
    });
  } catch {
    return null;
  }
}

/** Grab the largest <canvas> elements (3D previews) as PNG data-URLs. */
function captureCanvases(max = 2): string[] {
  const shots: string[] = [];
  try {
    const canvases = Array.from(document.querySelectorAll("canvas")) as HTMLCanvasElement[];
    canvases.sort((a, b) => b.width * b.height - a.width * a.height);
    for (const c of canvases.slice(0, max)) {
      try {
        const url = c.toDataURL("image/png");
        if (url && url.length > 5000) shots.push(url);
      } catch {
        /* tainted canvas (cross-origin map tiles) — skip */
      }
    }
  } catch {
    /* ignore */
  }
  return shots;
}

/** Find the keychain designer SVG (front preview). */
export function getKeychainDesignerSvg(): SVGSVGElement | null {
  return document.querySelector<SVGSVGElement>('svg[data-testid="keychain-designer-svg"]');
}

/**
 * Capture preview images for an order: the keychain designer SVG first (it shows
 * the exact text/layout), then any 3D canvases. Returns PNG data-URLs.
 */
export async function capturePreviewImages(): Promise<string[]> {
  const shots: string[] = [];
  const svg = getKeychainDesignerSvg();
  if (svg) {
    const png = await svgToPngDataUrl(svg, { maxWidth: 1200 });
    if (png && png.length > 5000) shots.push(png);
  }
  shots.push(...captureCanvases(2));
  return shots.slice(0, 4);
}
