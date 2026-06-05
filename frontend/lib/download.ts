// Quota-gated download: requires login, enforces the free-download limit, and
// signals when the limit is hit so the UI can show the contact/pay popup.

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

export interface GatedDownloadOpts {
  taskId: string | null;
  downloadUrl?: string | null;
  meta?: { title?: string; city?: string; product_type?: "map" | "keychain" };
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
      }),
    });
    if (res.status === 402) { opts.onLimit?.(); return { status: "limit" }; }
    if (!res.ok) return { status: "error", message: `HTTP ${res.status}` };
    const data = await res.json();
    const url = data.url ? (data.url.startsWith("http") ? data.url : `${API_BASE}${data.url}`) : null;
    if (url) {
      const a = document.createElement("a");
      a.href = url; a.download = ""; document.body.appendChild(a); a.click(); a.remove();
    }
    return { status: "ok", quota: data.quota };
  } catch (e: any) {
    return { status: "error", message: e?.message };
  }
}
