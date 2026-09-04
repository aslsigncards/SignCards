import { initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';
import { getStorage } from 'firebase/storage';
import { getAnalytics, logEvent } from 'firebase/analytics';

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

let auth, firestore, storage, analytics;
if (firebaseConfigured) {
  const app = initializeApp(cfg);
  auth = getAuth(app);
  firestore = getFirestore(app);
  storage = getStorage(app);
  if (cfg.measurementId) analytics = getAnalytics(app);
}

/** Fire a GA4 event; no-op when Analytics is not configured. */
export function logAnalyticsEvent(name, params) {
  if (analytics) logEvent(analytics, name, params);
}

export { auth, firestore, storage };
