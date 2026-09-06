"use client";

import type { FirebaseApp } from "firebase/app";
import type { Auth, ConfirmationResult, User } from "firebase/auth";

// Public Firebase web config (safe to ship to the client). Env overrides allowed.
const firebaseConfig = {
  apiKey: process.env.NEXT_PUBLIC_FIREBASE_API_KEY || "AIzaSyD5xIX6JsD31XcbT5KXNnJfPjoeRVVum0o",
  authDomain: process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN || "monadruk.firebaseapp.com",
  projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID || "monadruk",
  storageBucket: process.env.NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET || "monadruk.firebasestorage.app",
  messagingSenderId: process.env.NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID || "655484480222",
  appId: process.env.NEXT_PUBLIC_FIREBASE_APP_ID || "1:655484480222:web:71d79bd33caacb1176704f",
};

function hasFirebaseConfig() {
  return Boolean(firebaseConfig.apiKey && firebaseConfig.authDomain && firebaseConfig.projectId && firebaseConfig.appId);
}

export function isFirebaseAuthConfigured() {
  return hasFirebaseConfig();
}

// Firebase SDK (firebase/app + firebase/auth) — завантажується ЛІНИВО через dynamic
// import(), а не статичним top-level import. Інакше AuthProvider (обгортає ВЕСЬ
// застосунок у root layout) тягнув би весь SDK у shared client-чанк КОЖНОЇ сторінки,
// включно з лендингом і юр-сторінками, де авторизація не потрібна одразу.
let app: FirebaseApp | null = null;
let auth: Auth | null = null;
let authModPromise: Promise<typeof import("firebase/auth")> | null = null;

function loadAuthMod() {
  if (!authModPromise) authModPromise = import("firebase/auth");
  return authModPromise;
}

export async function getFirebaseAuth(): Promise<Auth | null> {
  if (!hasFirebaseConfig()) return null;
  const [{ getApps, initializeApp }, authMod] = await Promise.all([
    import("firebase/app"),
    loadAuthMod(),
  ]);
  if (!app) app = getApps().length ? getApps()[0] : initializeApp(firebaseConfig);
  if (!auth) {
    // 06.09.2026: N-2 (initializeAuth без popupRedirectResolver, −92 КБ iframe) ВІДКОЧЕНО —
    // свіжий вхід через попап працював, але ВІДНОВЛЕННЯ збереженої сесії при повторному
    // відкритті кабінету/адмінки не завершувалось (сторінка лишалась порожньою).
    // Надійність важливіша за 92 КБ.
    auth = authMod.getAuth(app);
    try { auth.useDeviceLanguage(); } catch {/* ignore */}
  }
  return auth;
}

async function requireAuth(): Promise<Auth> {
  const a = await getFirebaseAuth();
  if (!a) throw new Error("Firebase не налаштований");
  return a;
}

// ── Auth-state subscription (для AuthProvider — без прямого імпорту firebase/auth) ──
export async function subscribeAuthState(cb: (user: User | null) => void): Promise<() => void> {
  const a = await getFirebaseAuth();
  if (!a) return () => {};
  const authMod = await loadAuthMod();
  return authMod.onAuthStateChanged(a, cb);
}

// ── Google ──
export async function signInWithGoogle() {
  const authMod = await loadAuthMod();
  const a = await requireAuth();
  const provider = new authMod.GoogleAuthProvider();
  provider.setCustomParameters({ prompt: "select_account" });
  return authMod.signInWithPopup(a, provider);
}

// ── Email / password ──
export async function signInWithEmail(email: string, password: string) {
  const authMod = await loadAuthMod();
  return authMod.signInWithEmailAndPassword(await requireAuth(), email, password);
}
export async function signUpWithEmail(email: string, password: string) {
  const authMod = await loadAuthMod();
  const cred = await authMod.createUserWithEmailAndPassword(await requireAuth(), email, password);
  try { if (cred.user) await authMod.sendEmailVerification(cred.user); } catch {/* non-fatal */}
  return cred;
}
export async function resetPassword(email: string) {
  const authMod = await loadAuthMod();
  return authMod.sendPasswordResetEmail(await requireAuth(), email);
}

// ── Phone ──
let recaptcha: InstanceType<typeof import("firebase/auth").RecaptchaVerifier> | null = null;
async function getRecaptcha(containerId = "recaptcha-container") {
  const authMod = await loadAuthMod();
  const a = await requireAuth();
  if (!recaptcha) {
    recaptcha = new authMod.RecaptchaVerifier(a, containerId, { size: "invisible" });
  }
  return recaptcha;
}
export function resetRecaptcha() {
  try { recaptcha?.clear(); } catch {/* ignore */}
  recaptcha = null;
}
export async function startPhoneSignIn(phoneE164: string, containerId = "recaptcha-container"): Promise<ConfirmationResult> {
  const authMod = await loadAuthMod();
  const a = await requireAuth();
  const verifier = await getRecaptcha(containerId);
  return authMod.signInWithPhoneNumber(a, phoneE164, verifier);
}

// ── common ──
export async function signOutUser() {
  const a = await getFirebaseAuth();
  if (a) {
    const authMod = await loadAuthMod();
    await authMod.signOut(a);
  }
}
export async function getIdToken(): Promise<string | null> {
  const a = await getFirebaseAuth();
  const user = a?.currentUser;
  if (!user) return null;
  try { return await user.getIdToken(); } catch { return null; }
}

// back-compat alias
export const signOutFromGoogle = signOutUser;
