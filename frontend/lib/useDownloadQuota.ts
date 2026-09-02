"use client";

import { useEffect, useState } from "react";
import { useAuth } from "@/components/AuthProvider";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

export type DownloadQuota = { remaining: number; limit: number; isAdmin?: boolean };

/**
 * Залишок безкоштовних завантажень для guided-флоу (T-D.5). Той самий endpoint,
 * що й у панелях (`/api/account/quota`), але без дублювання їхнього стану:
 * повертає null, поки користувач не залогінений або квота ще не завантажилась.
 * Оновлюється при зміні user і на подію `monadruk:quota-changed` (після завантаження).
 */
export function useDownloadQuota(): DownloadQuota | null {
  const { user, getIdToken } = useAuth();
  const [quota, setQuota] = useState<DownloadQuota | null>(null);

  useEffect(() => {
    let alive = true;
    const load = async () => {
      if (!user) { setQuota(null); return; }
      try {
        const token = await getIdToken();
        if (!token) { setQuota(null); return; }
        const r = await fetch(`${API_BASE}/api/account/quota`, { headers: { Authorization: `Bearer ${token}` } });
        if (!r.ok) return;
        const d = await r.json();
        const q = d?.quota || d;
        if (alive && q && typeof q.remaining === "number") {
          setQuota({ remaining: q.remaining, limit: typeof q.limit === "number" ? q.limit : 5, isAdmin: !!q.isAdmin });
        }
      } catch { /* ignore */ }
    };
    load();
    const onChanged = () => { void load(); };
    window.addEventListener("monadruk:quota-changed", onChanged);
    return () => { alive = false; window.removeEventListener("monadruk:quota-changed", onChanged); };
  }, [user, getIdToken]);

  return quota;
}
