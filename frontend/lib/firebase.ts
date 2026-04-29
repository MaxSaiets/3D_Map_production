"use client";

import { getApps, initializeApp, type FirebaseApp } from "firebase/app";
import {
  GoogleAuthProvider,
  getAuth,
  signInWithPopup,
  signOut,
  type Auth,
} from "firebase/auth";

const firebaseConfig = {
  apiKey: process.env.NEXT_PUBLIC_FIREBASE_API_KEY,
  authDomain: process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN,
  projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID,
  storageBucket: process.env.NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: process.env.NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID,
  appId: process.env.NEXT_PUBLIC_FIREBASE_APP_ID,
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
  if (!auth) auth = getAuth(app);
  return auth;
}

export async function signInWithGoogle() {
  const authInstance = getFirebaseAuth();
  if (!authInstance) throw new Error("Firebase client config is missing");
  const provider = new GoogleAuthProvider();
  provider.setCustomParameters({ prompt: "select_account" });
  return signInWithPopup(authInstance, provider);
}

export async function signOutFromGoogle() {
  const authInstance = getFirebaseAuth();
  if (authInstance) await signOut(authInstance);
}
