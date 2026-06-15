"use client";

import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { useTranslations } from "next-intl";
import { onAuthStateChanged, type User } from "firebase/auth";
import { X, Mail, Phone as PhoneIcon, Loader2 } from "lucide-react";
import {
  getFirebaseAuth, isFirebaseAuthConfigured, getIdToken,
  signInWithGoogle, signInWithEmail, signUpWithEmail, resetPassword,
  startPhoneSignIn, resetRecaptcha, signOutUser,
} from "@/lib/firebase";

interface AuthContextValue {
  user: User | null;
  loading: boolean;
  configured: boolean;
  signIn: () => void;        // opens the modal
  openLogin: () => void;
  signOut: () => Promise<void>;
  signInWithGoogle: () => Promise<void>;
  getIdToken: () => Promise<string | null>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const configured = isFirebaseAuthConfigured();
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(configured);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!configured) { setLoading(false); return; }
    const auth = getFirebaseAuth();
    if (!auth) { setLoading(false); return; }
    const unsub = onAuthStateChanged(auth, (u: User | null) => { setUser(u); setLoading(false); });
    return () => unsub();
  }, [configured]);

  const value = useMemo<AuthContextValue>(() => ({
    user, loading, configured,
    signIn: () => setOpen(true),
    openLogin: () => setOpen(true),
    signOut: async () => { await signOutUser(); },
    signInWithGoogle: async () => { await signInWithGoogle(); },
    getIdToken,
  }), [user, loading, configured]);

  return (
    <AuthContext.Provider value={value}>
      {children}
      {open && !user && <LoginModal onClose={() => setOpen(false)} />}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const value = useContext(AuthContext);
  if (!value) {
    return {
      user: null, loading: false, configured: false,
      signIn: () => {}, openLogin: () => {},
      signOut: async () => {}, signInWithGoogle: async () => {},
      getIdToken: async () => null,
    } as AuthContextValue;
  }
  return value;
}

/* ───────────────────── Login modal ───────────────────── */
type Tab = "email" | "phone" | "google";

function LoginModal({ onClose }: { onClose: () => void }) {
  const t = useTranslations("auth");
  const [tab, setTab] = useState<Tab>("email");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [mode, setMode] = useState<"in" | "up">("in");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [phone, setPhone] = useState("");
  const [code, setCode] = useState("");
  const [confirm, setConfirm] = useState<any>(null);

  const field = "w-full rounded-2xl border border-[var(--surface-border,#e3dccb)] bg-white px-4 py-3 text-sm text-[var(--ink,#1B2A22)] outline-none focus:border-[rgba(46,74,58,0.4)]";

  const onCloseSafe = () => { resetRecaptcha(); onClose(); };
  const errText = (code?: string, fallback?: string) => {
    const key = humanize(code);
    return key ? t(key) : (fallback || t("errGeneric"));
  };
  const wrap = async (fn: () => Promise<void>, fallbackMsg?: string) => {
    setBusy(true); setError(null);
    try { await fn(); onCloseSafe(); }
    catch (e: any) { setError(errText(e?.code, e?.message) || fallbackMsg || t("errGeneric")); }
    finally { setBusy(false); }
  };

  const doEmail = () => wrap(async () => {
    if (mode === "up") await signUpWithEmail(email.trim(), password);
    else await signInWithEmail(email.trim(), password);
  });
  const doGoogle = () => wrap(async () => { await signInWithGoogle(); });
  const sendCode = async () => {
    setBusy(true); setError(null);
    try { setConfirm(await startPhoneSignIn(phone.trim())); }
    catch (e: any) { setError(errText(e?.code, e?.message) || t("errSendCode")); resetRecaptcha(); }
    finally { setBusy(false); }
  };
  const verifyCode = () => wrap(async () => { await confirm.confirm(code.trim()); });

  return (
    <div className="fixed inset-0 z-[100] flex items-end justify-center bg-black/45 p-0 backdrop-blur-sm sm:items-center sm:p-4" onClick={onCloseSafe}>
      <div className="w-full max-w-[420px] rounded-t-[26px] border border-[var(--surface-border,#e3dccb)] bg-[var(--paper-2,#fff)] p-6 shadow-[0_30px_80px_rgba(15,23,42,0.35)] sm:rounded-[26px]" onClick={(e) => e.stopPropagation()}>
        <div className="mb-4 flex items-start justify-between">
          <div>
            <h3 className="font-serif text-2xl text-[var(--ink,#1B2A22)]">{t("title")}</h3>
            <p className="text-[12px] text-[var(--ink-3,#7c887f)]">{t("subtitle")}</p>
          </div>
          <button onClick={onCloseSafe} className="rounded-lg p-1 text-[var(--ink-3,#7c887f)] hover:bg-black/5"><X size={20} /></button>
        </div>

        <div className="mb-4 flex gap-1 rounded-2xl border border-[var(--surface-border,#e3dccb)] bg-white/70 p-1 text-xs">
          {([["email", t("tabEmail")], ["phone", t("tabPhone")], ["google", t("tabGoogle")]] as [Tab, string][]).map(([k, l]) => (
            <button key={k} onClick={() => { setTab(k); setError(null); }}
              className={`flex-1 rounded-xl px-2 py-2 font-semibold transition ${tab === k ? "bg-[var(--forest,#2E4A3A)] text-white" : "text-[var(--ink-2,#4b5a50)]"}`}>{l}</button>
          ))}
        </div>

        {tab === "email" && (
          <div className="space-y-2.5">
            <input className={field} type="email" placeholder={t("emailPlaceholder")} value={email} onChange={(e) => setEmail(e.target.value)} />
            <input className={field} type="password" placeholder={t("passwordPlaceholder")} value={password} onChange={(e) => setPassword(e.target.value)} />
            <button onClick={doEmail} disabled={busy} className="inline-flex min-h-12 w-full items-center justify-center gap-2 rounded-full bg-[var(--forest,#2E4A3A)] px-4 py-3 text-sm font-bold text-white disabled:opacity-60">
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Mail size={16} />}
              {mode === "up" ? t("signUp") : t("signIn")}
            </button>
            <div className="flex items-center justify-between text-[12px] text-[var(--ink-3,#7c887f)]">
              <button onClick={() => setMode(mode === "in" ? "up" : "in")} className="underline-offset-2 hover:underline">
                {mode === "in" ? t("noAccount") : t("haveAccount")}
              </button>
              {mode === "in" && (
                <button onClick={() => email && resetPassword(email.trim()).then(() => setError(t("resetSent"))).catch(() => {})} className="underline-offset-2 hover:underline">
                  {t("forgotPassword")}
                </button>
              )}
            </div>
          </div>
        )}

        {tab === "phone" && (
          <div className="space-y-2.5">
            {!confirm ? (
              <>
                <input className={field} type="tel" placeholder={t("phonePlaceholder")} value={phone} onChange={(e) => setPhone(e.target.value)} />
                <button onClick={sendCode} disabled={busy} className="inline-flex min-h-12 w-full items-center justify-center gap-2 rounded-full bg-[var(--forest,#2E4A3A)] px-4 py-3 text-sm font-bold text-white disabled:opacity-60">
                  {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <PhoneIcon size={16} />} {t("sendCode")}
                </button>
              </>
            ) : (
              <>
                <input className={field} inputMode="numeric" placeholder={t("codePlaceholder")} value={code} onChange={(e) => setCode(e.target.value)} />
                <button onClick={verifyCode} disabled={busy} className="inline-flex min-h-12 w-full items-center justify-center gap-2 rounded-full bg-[var(--forest,#2E4A3A)] px-4 py-3 text-sm font-bold text-white disabled:opacity-60">
                  {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : null} {t("verifyCode")}
                </button>
              </>
            )}
          </div>
        )}

        {tab === "google" && (
          <button onClick={doGoogle} disabled={busy} className="inline-flex min-h-12 w-full items-center justify-center gap-2 rounded-full border border-[var(--surface-border,#e3dccb)] bg-white px-4 py-3 text-sm font-bold text-[var(--ink,#1B2A22)] disabled:opacity-60">
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <span className="text-[16px] font-bold">G</span>} {t("continueGoogle")}
          </button>
        )}

        {error && <div className="mt-3 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">{error}</div>}
        <div id="recaptcha-container" />
      </div>
    </div>
  );
}

// Maps a Firebase auth error code to a translation key in the "auth" namespace.
function humanize(code?: string): string | null {
  const m: Record<string, string> = {
    "auth/invalid-email": "errInvalidEmail",
    "auth/missing-password": "errMissingPassword",
    "auth/weak-password": "errWeakPassword",
    "auth/email-already-in-use": "errEmailInUse",
    "auth/invalid-credential": "errInvalidCredential",
    "auth/wrong-password": "errWrongPassword",
    "auth/user-not-found": "errUserNotFound",
    "auth/invalid-phone-number": "errInvalidPhone",
    "auth/invalid-verification-code": "errInvalidCode",
    "auth/too-many-requests": "errTooManyRequests",
    "auth/popup-closed-by-user": "errPopupClosed",
    "auth/billing-not-enabled": "errBillingNotEnabled",
  };
  return code ? (m[code] || null) : null;
}
