// Quota-gated download: requires login, enforces the free-download limit, and
// signals when the limit is hit so the UI can show the contact/pay popup.

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

export interface GatedDownloadOpts {
  taskId: string | null;
  downloadUrl?: string | null;
  meta?: { title?: string; city?: string; product_type?: "map" | "keychain" };
  /** Optional small PNG data-URL thumbnail saved with the model in account history. */
  preview?: string | null;
  /** Optional generation params (lat/lon/size_mm/scenario/...) saved alongside the
   *  model — lets the account page offer "regenerate" with the same place/size later. */
  params?: Record<string, unknown> | null;
  getIdToken: () => Promise<string | null>;
  openLogin: () => void;
  onLimit?: () => void;
}

export type GatedResult = { status: "ok" | "login" | "limit" | "error"; quota?: any; message?: string };

export async function gatedDownload(opts: GatedDownloadOpts): Promise<GatedResult> {
  const token = await opts.getIdToken();
  if (!token) { opts.openLogin(); return { status: "login" }; }
  try {
    const res = await fetch(`${API_BASE}/api/account/download`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
      body: JSON.stringify({
        task_id: opts.taskId,
        download_url: opts.downloadUrl || "",
        title: opts.meta?.title || "",
        city: opts.meta?.city || "",
        product_type: opts.meta?.product_type || "map",
        preview: opts.preview || "",
        ...(opts.params ? { params: opts.params } : {}),
      }),
    });
    if (res.status === 402) { opts.onLimit?.(); return { status: "limit" }; }
    if (res.status === 401) { opts.openLogin(); return { status: "login" }; }
    if (!res.ok) return { status: "error", message: `HTTP ${res.status}` };
    // The endpoint streams the full model file directly (auth-gated delivery).
    const blob = await res.blob();
    const remaining = res.headers.get("X-Quota-Remaining");
    const cd = res.headers.get("Content-Disposition") || "";
    const nameMatch = cd.match(/filename="?([^"]+)"?/);
    const fname = nameMatch?.[1] || `monadruk_${opts.taskId || "model"}.3mf`;
    const blobUrl = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = blobUrl; a.download = fname; document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(blobUrl);
    try { const { track } = await import("./analytics"); track("download_model"); } catch { /* ignore */ }
    return { status: "ok", quota: remaining != null ? { remaining: Number(remaining) } : undefined };
  } catch (e: any) {
    return { status: "error", message: e?.message };
  }
}
