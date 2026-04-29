"use client";

import { onAuthStateChanged, type User } from "firebase/auth";
import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { LogIn, X } from "lucide-react";
import { getFirebaseAuth, isFirebaseAuthConfigured, signInWithGoogle, signOutFromGoogle } from "@/lib/firebase";
import { setApiAuthTokenProvider } from "@/lib/api";

interface AuthContextValue {
  user: User | null;
  loading: boolean;
  configured: boolean;
  getIdToken: () => Promise<string | null>;
  signIn: () => Promise<void>;
  signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const configured = isFirebaseAuthConfigured();
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(configured);
  const [loginOpen, setLoginOpen] = useState(false);
  const [loginError, setLoginError] = useState<string | null>(null);
  const [loginBusy, setLoginBusy] = useState(false);

  useEffect(() => {
    const auth = getFirebaseAuth();
    if (!auth) {
      setLoading(false);
      return;
    }
    return onAuthStateChanged(auth, (nextUser) => {
      setUser(nextUser);
      setLoading(false);
    });
  }, []);

  useEffect(() => {
    setApiAuthTokenProvider(async () => user ? user.getIdToken() : null);
    return () => setApiAuthTokenProvider(null);
  }, [user]);

  const value = useMemo<AuthContextValue>(() => ({
    user,
    loading,
    configured,
    getIdToken: async () => user ? user.getIdToken() : null,
    signIn: async () => {
      setLoginError(null);
      setLoginOpen(true);
    },
    signOut: async () => {
      await signOutFromGoogle();
    },
  }), [configured, loading, user]);

  const runGoogleLogin = async () => {
    setLoginBusy(true);
    setLoginError(null);
    try {
      await signInWithGoogle();
      setLoginOpen(false);
    } catch (error: any) {
      setLoginError(error?.message || "Не вдалося увійти через Google");
    } finally {
      setLoginBusy(false);
    }
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
      {loginOpen && (
        <div className="fixed inset-0 z-[10000] grid place-items-center bg-[#1f2420]/45 p-4 backdrop-blur-sm">
          <div className="w-full max-w-[460px] overflow-hidden rounded-[10px] border border-[#dfd7c8] bg-[#fffaf1] shadow-[0_30px_90px_rgba(31,36,32,0.28)]">
            <div className="flex items-center justify-between border-b border-[#dfd7c8] px-5 py-4">
              <div>
                <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-[#8a8173]">Особистий кабінет</p>
                <h2 className="mt-1 font-serif text-3xl text-[#1f2420]">Увійдіть через Google</h2>
              </div>
              <button
                type="button"
                onClick={() => setLoginOpen(false)}
                className="grid h-9 w-9 place-items-center rounded-[6px] border border-[#dfd7c8] bg-[#f7f2e8]"
                aria-label="Закрити"
              >
                <X size={16} />
              </button>
            </div>
            <div className="p-5">
              <p className="text-sm leading-6 text-[#71695e]">
                Кабінет зберігає історію повністю згенерованих моделей, файли для завантаження і ліміт акаунта. Безкоштовно доступно 10 повних генерацій.
              </p>
              {!configured && (
                <div className="mt-4 rounded-[8px] border border-[#efb4a8] bg-[#f5ded8] p-3 text-sm text-[#8a3b2f]">
                  Firebase ще не налаштований на сервері. Додайте NEXT_PUBLIC_FIREBASE_* змінні та перезберіть frontend.
                </div>
              )}
              {loginError && (
                <div className="mt-4 rounded-[8px] border border-[#efb4a8] bg-[#f5ded8] p-3 text-sm text-[#8a3b2f]">{loginError}</div>
              )}
              <button
                type="button"
                onClick={runGoogleLogin}
                disabled={!configured || loginBusy}
                className="mt-5 inline-flex h-11 w-full items-center justify-center gap-2 rounded-[6px] bg-[#1f5b49] px-4 text-sm font-semibold text-white disabled:opacity-60"
              >
                <LogIn size={16} />
                {loginBusy ? "Вхід..." : "Продовжити з Google"}
              </button>
              <p className="mt-3 text-center text-xs text-[#8a8173]">Ми використовуємо Google тільки для входу і привʼязки історії моделей.</p>
            </div>
          </div>
        </div>
      )}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const value = useContext(AuthContext);
  if (!value) throw new Error("useAuth must be used inside AuthProvider");
  return value;
}
