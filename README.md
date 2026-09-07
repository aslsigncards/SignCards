# SignCards

A local-first American Sign Language flashcard trainer. SignCards records **your own** signing as the reference baseline, then compares each practice attempt against it frame-by-frame using webcam hand landmarks — so it grades you against your hands, not a generic dataset.

Built in collaboration with the McMaster ASL Club.

- **Questions:** [aslsigncards@gmail.com](mailto:aslsigncards@gmail.com)

---

## Features

- **Personal baseline matching** — record each sign once, then practice against your own reference
- **Two-hand tracking** with mirrored matching, so left- and right-dominant signers both work
- **Dynamic sign support** — 1-second motion capture compared with dynamic time warping, not single frames
- **Custom sets** — build your own decks, including multi-sign entries (fingerspelled words, compounds, phrases)
- **Practice modes** — in-order or Anki-style weighted random that prioritizes your weakest cards
- **Stats** — pass rates, 14-day activity chart, and weakest-to-strongest card rankings
- **Works offline** — all data lives in IndexedDB by default; no account required
- **Optional cloud sync** — sign in to back up and sync baselines across devices

---

## Tech stack

| Layer | Tool |
| --- | --- |
| UI | React 19 + Tailwind CSS 4 |
| Build | Vite |
| Hand tracking | MediaPipe Tasks Vision (`HandLandmarker`) |
| Local storage | Dexie (IndexedDB) |
| Cloud (optional) | Firebase Auth, Firestore, Cloud Storage |
| Lint | Oxlint |

---

## Run it locally

Requires Node.js 20+.

```bash
git clone https://github.com/aslsigncards/SignCards.git
cd SignCards
npm install
npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`) and allow camera access.

The app is fully usable at this point. Cloud sync stays disabled until you add Firebase credentials, and the sign-in button will say so.

### Scripts

| Command | Purpose |
| --- | --- |
| `npm run dev` | Start the dev server with hot reload |
| `npm run build` | Production build into `dist/` |
| `npm run start` | Serve the built app (used by App Hosting) |
| `npm run preview` | Preview the production build locally |
| `npm run lint` | Run Oxlint |

---

## Optional: enable cloud sync

Only needed if you want accounts and cross-device sync. Skip this to run fully local.

1. Create a project at [console.firebase.google.com](https://console.firebase.google.com).
2. Enable **Authentication** → Email/Password and Google. Add your dev and deployed domains under **Authorized domains**.
3. Create a **Firestore** database (production mode) and a **Cloud Storage** bucket.
4. Copy the web app config from **Project Settings → Your apps** into a `.env.local` file in the project root:

```env
VITE_FIREBASE_API_KEY="..."
VITE_FIREBASE_AUTH_DOMAIN="your-project.firebaseapp.com"
VITE_FIREBASE_PROJECT_ID="your-project"
VITE_FIREBASE_STORAGE_BUCKET="your-project.firebasestorage.app"
VITE_FIREBASE_MESSAGING_SENDER_ID="..."
VITE_FIREBASE_APP_ID="..."
VITE_FIREBASE_MEASUREMENT_ID="G-..."
```

`.env.local` is gitignored. Restart the dev server after creating it.

5. Publish these rules so users can only ever touch their own documents:

**Firestore:**

```
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /users/{userId}/{document=**} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
    }
  }
}
```

**Storage:**

```
rules_version = '2';
service firebase.storage {
  match /b/{bucket}/o {
    match /users/{userId}/{allPaths=**} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
    }
  }
}
```

> **Note on the API key:** Firebase web config values are public by design — they identify your project, they don't authorize access. Your security boundary is the rules above plus API key restrictions in the Google Cloud console. Never commit service account keys or admin credentials.

---

## Deploying your own copy

The repo includes an `apphosting.yaml` for Firebase App Hosting. If you fork it:

1. Replace the `VITE_FIREBASE_*` values in `apphosting.yaml` with your own project's config.
2. In the Firebase console, create an App Hosting backend and connect it to your fork.
3. Push to your default branch — App Hosting builds and rolls out automatically.

The app is a static bundle, so any static host works too (Netlify, Vercel, GitHub Pages, Cloudflare Pages). Build with `npm run build` and serve `dist/`. Camera access requires HTTPS.

---

## Contributing & spinoffs

Contributions are welcome, and so are forks that go their own direction.

**Good first contributions:**

- New preset card sets
- Accessibility and keyboard navigation improvements
- Tuning the matching thresholds in `src/App.jsx` (`SEQUENCE_PASS_THRESHOLD`, `RECORDING_DURATION_MS`)
- Bug reports from real practice sessions — these are genuinely the most useful

**To contribute:**

1. Fork the repo and create a branch.
2. Make your change and run `npm run lint && npm run build`.
3. Open a pull request describing what you changed and how you tested it.

**Spinning it off?** Please do. The baseline-matching approach isn't ASL-specific — it should adapt to other signed languages, sport technique drills, physiotherapy exercises, or any skill with a repeatable hand shape. You don't need to ask permission. If you build something with it, we'd love to hear about it at [aslsigncards@gmail.com](mailto:aslsigncards@gmail.com).

---

## Privacy

- Video **never leaves your device**. MediaPipe runs entirely in the browser, and only numeric hand landmark coordinates are stored — no images or video frames.
- Without sign-in, all data stays in your browser's IndexedDB.
- With sign-in, baselines and history sync to your own Firebase account, readable only by you.
- Export and delete your data anytime from **Settings**, and delete your account from **Profile**.

---

## Project structure

```
src/
  App.jsx          UI, hand tracking loop, DTW matching, all app pages
  db.js            Dexie schema and migrations
  firebase.js      Conditional Firebase init + analytics helper
  syncService.js   Baseline serialization, delta sync, upload/download
apphosting.yaml    Firebase App Hosting build config
```

---

## License

See [LICENSE](LICENSE).
