import { loadFirestore, loadStorage } from './firebase';
import { newCloudId } from './db';

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

// ── Legacy Storage ────────────────────────────────────────────────────────────

// Legacy: baselines uploaded before frames moved into Firestore live in Storage.
const baselineDocId = (refKey) => encodeURIComponent(refKey.replace(/\//g, '__'));

async function downloadLegacySlot(uid, refKey, slot) {
  try {
    const { ref, getBytes, storage } = await loadStorage();
    if (!storage) return null;
    const buffer = await getBytes(ref(storage, `users/${uid}/baselines/${baselineDocId(refKey)}_${slot}.b64`));
    return base64ToFrames(new TextDecoder().decode(buffer));
  } catch {
    return null;
  }
}

// ── Upload ────────────────────────────────────────────────────────────────────

const packMotion = (samples) => samples.map((s) => ({
  x: s.x, y: s.y, z: s.z, s: s.s,
  ...(Number.isFinite(s.b) ? { b: s.b } : {}),
}));

const HISTORY_ENTRIES_PER_DOC = 1000;
const HISTORY_FLUSH_DELAY_MS = 30_000;

// Custom-set baseline keys are `custom:<setId>:<word>`. Locally setId is the
// device's auto-increment id; in the cloud it is the set's stable cloudId.
const CUSTOM_KEY = /^custom:([^:]+):(.*)$/s;

async function toCloudKey(localDb, refKey) {
  const m = CUSTOM_KEY.exec(refKey);
  if (!m) return refKey;
  const set = await localDb.customSets.get(Number(m[1]));
  return set?.cloudId ? `custom:${set.cloudId}:${m[2]}` : null;
}

async function toLocalKey(localDb, cloudKey) {
  const m = CUSTOM_KEY.exec(cloudKey);
  if (!m) return cloudKey;
  const set = await localDb.customSets.where('cloudId').equals(m[1]).first();
  return set ? `custom:${set.id}:${m[2]}` : null;
}

// Local rows are tagged with the uid they were synced to, so "Sync now" only
// uploads what the cloud doesn't already have.
export async function uploadBaseline(uid, localKey, frames, frames2, timestamp, motion, motion2, localDb) {
  const { doc, setDoc, serverTimestamp, db } = await loadFirestore();
  if (!db) return;
  const refKey = await toCloudKey(localDb, localKey);
  if (!refKey) return;
  await setDoc(doc(db, `users/${uid}/baselines/${baselineDocId(refKey)}`), {
    refKey,
    // Frames (~10-20 KB per hand) live in the doc so Cloud Storage isn't needed.
    ...(frames ? { h1: framesToBase64(frames) } : {}),
    ...(frames2 ? { h2: framesToBase64(frames2) } : {}),
    ...(motion ? { motion: packMotion(motion) } : {}),
    ...(motion2 ? { motion2: packMotion(motion2) } : {}),
    updatedAt: timestamp ?? Date.now(),
    syncedAt: serverTimestamp(),
  });
  const local = await localDb.references.get(localKey);
  // Skip if the baseline was re-recorded while this upload was in flight.
  if (local && local.timestamp === timestamp) await localDb.references.update(localKey, { syncedUid: uid });
}

const setNeedsUpload = (set, uid) => set.syncedUid !== uid || set.syncedVersion !== (set.updatedAt ?? null);

export async function uploadCustomSet(uid, localId, localDb) {
  const { doc, setDoc, serverTimestamp, db } = await loadFirestore();
  if (!db) return;
  const set = await localDb.customSets.get(localId);
  if (!set?.cloudId) return;
  await setDoc(doc(db, `users/${uid}/customSets/${set.cloudId}`), {
    title: set.title,
    words: set.words,
    cloudId: set.cloudId,
    updatedAt: set.updatedAt ?? Date.now(),
    syncedAt: serverTimestamp(),
  });
  await localDb.customSets.where('id').equals(localId).modify((s) => {
    if (s.updatedAt === set.updatedAt) {
      s.syncedUid = uid;
      s.syncedVersion = set.updatedAt ?? null;
    }
  });
}

// Practice attempts upload in batches (one doc per up to 1000 attempts)
// instead of one write each, which keeps writes far below the free quota.
let historyFlushTimer = null;
let historyFlushChain = Promise.resolve();

export function scheduleHistoryFlush(uid, localDb) {
  clearTimeout(historyFlushTimer);
  historyFlushTimer = setTimeout(() => { flushHistory(uid, localDb).catch(() => {}); }, HISTORY_FLUSH_DELAY_MS);
}

// Runs are chained so overlapping calls never upload the same entries twice.
export function flushHistory(uid, localDb) {
  clearTimeout(historyFlushTimer);
  historyFlushTimer = null;
  historyFlushChain = historyFlushChain.catch(() => {}).then(() => runFlushHistory(uid, localDb));
  return historyFlushChain;
}

async function runFlushHistory(uid, localDb) {
  const { doc, setDoc, serverTimestamp, db } = await loadFirestore();
  if (!db) return;
  const pending = await localDb.history.filter((h) => h.syncedUid !== uid).toArray();
  for (let i = 0; i < pending.length; i += HISTORY_ENTRIES_PER_DOC) {
    const chunk = pending.slice(i, i + HISTORY_ENTRIES_PER_DOC);
    const entries = chunk.map(({ id: _id, syncedUid: _s, ...data }) => data);
    await setDoc(doc(db, `users/${uid}/history/b_${chunk[0].timestamp}_${newCloudId().slice(0, 8)}`), {
      entries,
      syncedAt: serverTimestamp(),
    });
    await localDb.history.where('id').anyOf(chunk.map((h) => h.id)).modify({ syncedUid: uid });
  }
}

// ── Download (delta / lazy) ───────────────────────────────────────────────────

// Per-collection cursor of the newest server `syncedAt` already merged, so each
// sync only reads documents written since the last one instead of everything.
const CURSOR_PREFIX = 'signcards.syncCursor.';

function readCursors(uid) {
  try {
    return JSON.parse(window.localStorage.getItem(CURSOR_PREFIX + uid)) || {};
  } catch {
    return {};
  }
}

function writeCursor(uid, name, ts) {
  const cursors = readCursors(uid);
  cursors[name] = ts ? { s: ts.seconds, n: ts.nanoseconds } : (cursors[name] ?? { s: 0, n: 0 });
  window.localStorage.setItem(CURSOR_PREFIX + uid, JSON.stringify(cursors));
}

/** Forces the next sync to re-read everything; call after local data is wiped. */
export function resetSyncCursors() {
  Object.keys(window.localStorage)
    .filter((k) => k.startsWith(CURSOR_PREFIX))
    .forEach((k) => window.localStorage.removeItem(k));
}

// Returns docs changed since the cursor (all docs if there is no cursor yet)
// plus the newest syncedAt seen, to be committed once the merge succeeds.
async function fetchChanged(fs, uid, name) {
  const { collection, getDocs, query, where, Timestamp, db } = fs;
  const col = collection(db, `users/${uid}/${name}`);
  const cursor = readCursors(uid)[name];
  const since = cursor ? new Timestamp(cursor.s, cursor.n) : null;
  const snap = await getDocs(since ? query(col, where('syncedAt', '>', since)) : col);
  let newest = since;
  const rows = snap.docs.map((d) => {
    const { syncedAt, ...data } = d.data();
    if (syncedAt && (!newest || syncedAt.valueOf() > newest.valueOf())) newest = syncedAt;
    return data;
  });
  return { rows, newest };
}

// Concurrent callers (auth load + set select) share one in-flight sync.
let inflight = null;

export function downloadAndMerge(uid, db) {
  if (inflight?.uid !== uid) {
    const promise = runDownloadAndMerge(uid, db).finally(() => {
      if (inflight?.promise === promise) inflight = null;
    });
    inflight = { uid, promise };
  }
  return inflight.promise;
}

// Delta merge: skips baselines where local timestamp ≥ cloud updatedAt.
async function runDownloadAndMerge(uid, db) {
  const fs = await loadFirestore();
  if (!fs.db) return;

  // Sets first, so custom-set baselines below can resolve to a local set id.
  const sets = await fetchChanged(fs, uid, 'customSets');
  for (const data of sets.rows) {
    if (data.cloudId) {
      const local = await db.customSets.where('cloudId').equals(data.cloudId).first();
      if (local) {
        if ((data.updatedAt ?? 0) > (local.updatedAt ?? 0)) {
          await db.customSets.update(local.id, {
            title: data.title,
            words: data.words,
            updatedAt: data.updatedAt,
            syncedUid: uid,
            syncedVersion: data.updatedAt,
          });
        }
        continue;
      }
    }
    // Docs from before cloudIds existed only match by title, as they always did.
    const sameTitle = await db.customSets.where('title').equals(data.title).first();
    if (!sameTitle) {
      const updatedAt = data.updatedAt ?? Date.now();
      await db.customSets.add({
        title: data.title,
        words: data.words,
        cloudId: data.cloudId || newCloudId(),
        updatedAt,
        ...(data.cloudId ? { syncedUid: uid, syncedVersion: updatedAt } : {}),
      });
    }
  }
  writeCursor(uid, 'customSets', sets.newest);

  const baselines = await fetchChanged(fs, uid, 'baselines');
  const localRefs = await db.references.toArray();
  const localMap = Object.fromEntries(localRefs.map((r) => [r.word, r]));

  const toFetch = [];
  for (const meta of baselines.rows) {
    const localKey = await toLocalKey(db, meta.refKey);
    if (!localKey) continue;
    const local = localMap[localKey];
    if (!local || meta.updatedAt > (local.timestamp ?? 0)) toFetch.push({ ...meta, localKey });
  }

  // Batches of 5 only matter for legacy docs that still fetch from Storage.
  for (let i = 0; i < toFetch.length; i += 5) {
    await Promise.all(toFetch.slice(i, i + 5).map(async (meta) => {
      const { refKey, localKey } = meta;
      let frames = meta.h1 ? base64ToFrames(meta.h1) : null;
      let frames2 = meta.h2 ? base64ToFrames(meta.h2) : null;
      const legacy = !frames && !frames2 && (meta.hasH1 || meta.hasH2);
      if (legacy) {
        [frames, frames2] = await Promise.all([
          meta.hasH1 ? downloadLegacySlot(uid, refKey, 'h1') : null,
          meta.hasH2 ? downloadLegacySlot(uid, refKey, 'h2') : null,
        ]);
      }
      if (frames || frames2) {
        await db.references.put({
          word: localKey,
          timestamp: meta.updatedAt,
          frames: frames ?? null,
          ...(frames2 ? { frames2 } : {}),
          ...(Array.isArray(meta.motion) ? { motion: meta.motion } : {}),
          ...(Array.isArray(meta.motion2) ? { motion2: meta.motion2 } : {}),
          // Legacy downloads stay unsynced so the next upload rewrites them into Firestore.
          ...(legacy ? {} : { syncedUid: uid }),
        });
      }
    }));
  }
  writeCursor(uid, 'baselines', baselines.newest);

  // Merge history additively by (word, timestamp). Docs are either batches
  // ({ entries }) or single attempts written by older app versions.
  const hist = await fetchChanged(fs, uid, 'history');
  const cloudHist = hist.rows.flatMap((row) => (Array.isArray(row.entries) ? row.entries : [row]));
  if (cloudHist.length) {
    const localHist = await db.history.toArray();
    const localTsSet = new Set(localHist.map((h) => `${h.word}:${h.timestamp}`));
    const toAdd = cloudHist
      .filter((h) => !localTsSet.has(`${h.word}:${h.timestamp}`))
      .map((h) => ({ ...h, syncedUid: uid }));
    if (toAdd.length) {
      await db.history.bulkAdd(toAdd).catch(() => {});
    }
  }
  writeCursor(uid, 'history', hist.newest);
}

// Uploads only what isn't in the cloud yet, so it's cheap to run on every sign-in.
export async function uploadAllLocalData(uid, db) {
  const { db: firestoreDb } = await loadFirestore();
  if (!firestoreDb) return;
  const [refs, sets] = await Promise.all([
    db.references.filter((r) => r.syncedUid !== uid).toArray(),
    db.customSets.filter((s) => setNeedsUpload(s, uid)).toArray(),
  ]);

  // Sets first so custom-set baselines can resolve their cloudId.
  for (const set of sets) {
    await uploadCustomSet(uid, set.id, db);
  }

  for (let i = 0; i < refs.length; i += 3) {
    await Promise.all(refs.slice(i, i + 3).map((r) =>
      uploadBaseline(uid, r.word, r.frames, r.frames2, r.timestamp, r.motion, r.motion2, db)
    ));
  }

  await flushHistory(uid, db);
}
