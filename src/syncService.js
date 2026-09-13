import { loadFirestore, loadStorage } from './firebase';

const FLOATS_PER_FRAME = 63; // 21 landmarks × 3 (x, y, z)

// ── Serialization: Float32Array base64 ≈ 4.7× smaller than JSON ──────────────

export function framesToBase64(frames) {
  if (!frames?.length) return null;
  const buf = new Float32Array(frames.length * FLOATS_PER_FRAME);
  frames.forEach((frame, fi) => {
    frame.forEach((lm, li) => {
      const base = fi * FLOATS_PER_FRAME + li * 3;
      buf[base] = lm.x;
      buf[base + 1] = lm.y;
      buf[base + 2] = lm.z;
    });
  });
  const bytes = new Uint8Array(buf.buffer);
  let str = '';
  for (let i = 0; i < bytes.length; i++) str += String.fromCharCode(bytes[i]);
  return btoa(str);
}

export function base64ToFrames(b64) {
  if (!b64) return null;
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const buf = new Float32Array(bytes.buffer);
  const frameCount = buf.length / FLOATS_PER_FRAME;
  const frames = [];
  for (let fi = 0; fi < frameCount; fi++) {
    const frame = [];
    for (let li = 0; li < 21; li++) {
      const base = fi * FLOATS_PER_FRAME + li * 3;
      frame.push({ x: buf[base], y: buf[base + 1], z: buf[base + 2] });
    }
    frames.push(frame);
  }
  return frames;
}

// ── Storage paths ─────────────────────────────────────────────────────────────

function baselinePath(uid, refKey, slot) {
  const safe = encodeURIComponent(refKey.replace(/\//g, '__'));
  return `users/${uid}/baselines/${safe}_${slot}.b64`;
}

// ── Upload ────────────────────────────────────────────────────────────────────

const packMotion = (samples) => samples.map((s) => ({
  x: s.x, y: s.y, z: s.z, s: s.s,
  ...(Number.isFinite(s.b) ? { b: s.b } : {}),
}));

export async function uploadBaseline(uid, refKey, frames, frames2, timestamp, motion, motion2) {
  const [{ doc, setDoc, db }, { ref, uploadString, storage }] = await Promise.all([loadFirestore(), loadStorage()]);
  if (!db || !storage) return;
  const uploads = [];
  if (frames) uploads.push(uploadString(ref(storage, baselinePath(uid, refKey, 'h1')), framesToBase64(frames)));
  if (frames2) uploads.push(uploadString(ref(storage, baselinePath(uid, refKey, 'h2')), framesToBase64(frames2)));
  await Promise.all(uploads);
  const safe = encodeURIComponent(refKey.replace(/\//g, '__'));
  await setDoc(doc(db, `users/${uid}/baselines/${safe}`), {
    refKey,
    hasH1: !!frames,
    hasH2: !!frames2,
    // Wrist trajectories are small enough to live in the doc itself.
    ...(motion ? { motion: packMotion(motion) } : {}),
    ...(motion2 ? { motion2: packMotion(motion2) } : {}),
    updatedAt: timestamp ?? Date.now(),
  });
}

export async function uploadCustomSet(uid, set) {
  const { doc, setDoc, db } = await loadFirestore();
  if (!db) return;
  const { id, ...data } = set;
  await setDoc(doc(db, `users/${uid}/customSets/${id}`), {
    ...data,
    localId: id,
    updatedAt: Date.now(),
  });
}

export async function uploadHistoryEntry(uid, entry) {
  const { doc, setDoc, db } = await loadFirestore();
  if (!db) return;
  const { id: _id, ...data } = entry;
  // deterministic doc ID prevents duplicates on repeat syncs
  const docId = `${String(data.word).replace(/[^a-zA-Z0-9]/g, '_')}_${data.timestamp}`;
  return setDoc(doc(db, `users/${uid}/history/${docId}`), data).catch(() => {});
}

// ── Download (delta / lazy) ───────────────────────────────────────────────────

async function fetchCloudBaselineMeta(uid) {
  const { collection, getDocs, db } = await loadFirestore();
  if (!db) return {};
  const snap = await getDocs(collection(db, `users/${uid}/baselines`));
  const map = {};
  snap.forEach((d) => { map[d.data().refKey] = d.data(); });
  return map;
}

async function downloadSlot(uid, refKey, slot) {
  try {
    const { ref, getBytes, storage } = await loadStorage();
    const buffer = await getBytes(ref(storage, baselinePath(uid, refKey, slot)));
    const b64 = new TextDecoder().decode(buffer);
    return base64ToFrames(b64);
  } catch {
    return null;
  }
}

// Delta merge: skips baselines where local timestamp ≥ cloud updatedAt.
// setKeyFilter restricts to one set's baselines (lazy loading on set-select).
export async function downloadAndMerge(uid, db, setKeyFilter = null) {
  const { collection, getDocs, db: firestoreDb } = await loadFirestore();
  const { storage } = await loadStorage();
  if (!firestoreDb || !storage) return;

  const cloudMeta = await fetchCloudBaselineMeta(uid);
  const localRefs = await db.references.toArray();
  const localMap = Object.fromEntries(localRefs.map((r) => [r.word, r]));

  const toFetch = Object.entries(cloudMeta).filter(([refKey, meta]) => {
    if (setKeyFilter && !refKey.startsWith(`${setKeyFilter}:`)) return false;
    const local = localMap[refKey];
    return !local || meta.updatedAt > (local.timestamp ?? 0);
  });

  // Fetch in parallel batches of 5 to avoid overwhelming Storage
  for (let i = 0; i < toFetch.length; i += 5) {
    await Promise.all(toFetch.slice(i, i + 5).map(async ([refKey, meta]) => {
      const [frames, frames2] = await Promise.all([
        meta.hasH1 ? downloadSlot(uid, refKey, 'h1') : Promise.resolve(null),
        meta.hasH2 ? downloadSlot(uid, refKey, 'h2') : Promise.resolve(null),
      ]);
      if (frames || frames2) {
        await db.references.put({
          word: refKey,
          timestamp: meta.updatedAt,
          frames: frames ?? null,
          ...(frames2 ? { frames2 } : {}),
          ...(Array.isArray(meta.motion) ? { motion: meta.motion } : {}),
          ...(Array.isArray(meta.motion2) ? { motion2: meta.motion2 } : {}),
        });
      }
    }));
  }

  // Merge customSets by title (skips sets already present locally)
  const setsSnap = await getDocs(collection(firestoreDb, `users/${uid}/customSets`));
  for (const setDocSnap of setsSnap.docs) {
    const data = setDocSnap.data();
    const existing = await db.customSets.where('title').equals(data.title).first();
    if (!existing) {
      const { localId: _lid, updatedAt: _u, ...rest } = data;
      await db.customSets.add(rest);
    }
  }

  // Merge history additively by (word, timestamp) composite key
  const histSnap = await getDocs(collection(firestoreDb, `users/${uid}/history`));
  const localHist = await db.history.toArray();
  const localTsSet = new Set(localHist.map((h) => `${h.word}:${h.timestamp}`));
  const toAdd = [];
  histSnap.forEach((d) => {
    const h = d.data();
    if (!localTsSet.has(`${h.word}:${h.timestamp}`)) toAdd.push(h);
  });
  if (toAdd.length) {
    await db.history.bulkAdd(toAdd).catch(() => {});
  }
}

export function downloadSetBaselines(uid, setKey, db) {
  return downloadAndMerge(uid, db, setKey);
}

export async function uploadAllLocalData(uid, db) {
  const { doc, writeBatch, db: firestoreDb } = await loadFirestore();
  const { storage } = await loadStorage();
  if (!firestoreDb || !storage) return;
  const [refs, sets, hist] = await Promise.all([
    db.references.toArray(),
    db.customSets.toArray(),
    db.history.toArray(),
  ]);

  for (let i = 0; i < refs.length; i += 3) {
    await Promise.all(refs.slice(i, i + 3).map((r) =>
      uploadBaseline(uid, r.word, r.frames, r.frames2, r.timestamp, r.motion, r.motion2)
    ));
  }

  for (const set of sets) {
    await uploadCustomSet(uid, set);
  }

  // writeBatch is idempotent (deterministic doc IDs), safe to call repeatedly
  for (let i = 0; i < hist.length; i += 499) {
    const batch = writeBatch(firestoreDb);
    hist.slice(i, i + 499).forEach((h) => {
      const { id: _id, ...data } = h;
      const docId = `${String(data.word).replace(/[^a-zA-Z0-9]/g, '_')}_${data.timestamp}`;
      batch.set(doc(firestoreDb, `users/${uid}/history/${docId}`), data);
    });
    await batch.commit();
  }
}
