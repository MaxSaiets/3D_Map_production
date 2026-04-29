"use client";

import { onAuthStateChanged, type User } from "firebase/auth";
import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
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
      await signInWithGoogle();
    },
    signOut: async () => {
      await signOutFromGoogle();
    },
  }), [configured, loading, user]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const value = useContext(AuthContext);
  if (!value) throw new Error("useAuth must be used inside AuthProvider");
  return value;
}
