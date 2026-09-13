import { initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';

const cfg = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID,
  measurementId: import.meta.env.VITE_FIREBASE_MEASUREMENT_ID,
};

export const firebaseConfigured = !!cfg.apiKey;

let app, auth;
if (firebaseConfigured) {
  app = initializeApp(cfg);
  auth = getAuth(app);
}

// Firestore, Storage, and Analytics are only needed for cloud sync and usage
// events, not for basic auth state, so they load in their own chunk on first
// use instead of blocking every page's initial bundle.
let firestoreModulePromise;
export function loadFirestore() {
  if (!firebaseConfigured) return Promise.resolve({ db: null });
  if (!firestoreModulePromise) {
    firestoreModulePromise = import('firebase/firestore').then((mod) => ({ ...mod, db: mod.getFirestore(app) }));
  }
  return firestoreModulePromise;
}

let storageModulePromise;
export function loadStorage() {
  if (!firebaseConfigured) return Promise.resolve({ storage: null });
  if (!storageModulePromise) {
    storageModulePromise = import('firebase/storage').then((mod) => ({ ...mod, storage: mod.getStorage(app) }));
  }
  return storageModulePromise;
}

let analyticsModulePromise;
/** Fire a GA4 event; no-op when Analytics is not configured. */
export function logAnalyticsEvent(name, params) {
  if (!firebaseConfigured || !cfg.measurementId) return;
  if (!analyticsModulePromise) {
    analyticsModulePromise = import('firebase/analytics').then((mod) => ({ ...mod, analytics: mod.getAnalytics(app) }));
  }
  analyticsModulePromise.then(({ logEvent, analytics }) => logEvent(analytics, name, params)).catch(() => {});
}

export { auth };
