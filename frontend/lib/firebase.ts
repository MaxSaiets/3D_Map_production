"use client";

import { getApps, initializeApp, type FirebaseApp } from "firebase/app";
import {
  GoogleAuthProvider,
  getAuth,
  signInWithPopup,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  sendPasswordResetEmail,
  signInWithPhoneNumber,
  RecaptchaVerifier,
  sendEmailVerification,
  signOut,
  type Auth,
  type ConfirmationResult,
} from "firebase/auth";

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

let app: FirebaseApp | null = null;
let auth: Auth | null = null;

export function isFirebaseAuthConfigured() {
  return hasFirebaseConfig();
}

export function getFirebaseAuth() {
  if (!hasFirebaseConfig()) return null;
  if (!app) app = getApps().length ? getApps()[0] : initializeApp(firebaseConfig);
  if (!auth) {
    auth = getAuth(app);
    try { auth.useDeviceLanguage(); } catch {/* ignore */}
  }
  return auth;
}

function requireAuth(): Auth {
  const a = getFirebaseAuth();
  if (!a) throw new Error("Firebase не налаштований");
  return a;
}

// ── Google ──
export async function signInWithGoogle() {
  const provider = new GoogleAuthProvider();
  provider.setCustomParameters({ prompt: "select_account" });
  return signInWithPopup(requireAuth(), provider);
}

// ── Email / password ──
export async function signInWithEmail(email: string, password: string) {
  return signInWithEmailAndPassword(requireAuth(), email, password);
}
export async function signUpWithEmail(email: string, password: string) {
  const cred = await createUserWithEmailAndPassword(requireAuth(), email, password);
  try { if (cred.user) await sendEmailVerification(cred.user); } catch {/* non-fatal */}
  return cred;
}
export async function resetPassword(email: string) {
  return sendPasswordResetEmail(requireAuth(), email);
}

// ── Phone ──
let recaptcha: RecaptchaVerifier | null = null;
export function getRecaptcha(containerId = "recaptcha-container"): RecaptchaVerifier {
  const a = requireAuth();
  if (!recaptcha) {
    recaptcha = new RecaptchaVerifier(a, containerId, { size: "invisible" });
  }
  return recaptcha;
}
export function resetRecaptcha() {
  try { recaptcha?.clear(); } catch {/* ignore */}
  recaptcha = null;
}
export async function startPhoneSignIn(phoneE164: string, containerId = "recaptcha-container"): Promise<ConfirmationResult> {
  return signInWithPhoneNumber(requireAuth(), phoneE164, getRecaptcha(containerId));
}

// ── common ──
export async function signOutUser() {
  const a = getFirebaseAuth();
  if (a) await signOut(a);
}
export async function getIdToken(): Promise<string | null> {
  const a = getFirebaseAuth();
  const user = a?.currentUser;
  if (!user) return null;
  try { return await user.getIdToken(); } catch { return null; }
}

// back-compat alias
export const signOutFromGoogle = signOutUser;
