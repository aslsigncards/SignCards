const MAX_IMPORT_SETS = 500;
const MAX_IMPORT_REFERENCES = 5000;
const MAX_IMPORT_HISTORY = 100000;
const MAX_FRAMES_PER_SLOT = 600;

export const MAX_IMPORT_BYTES = 50 * 1024 * 1024;

export const isPlainRecord = (value) =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const cleanText = (value, maxLength) =>
  typeof value === 'string' ? value.replace(/[#[\]*/\\?]/g, '').slice(0, maxLength) : '';

const cleanFrames = (frames) => {
  if (!Array.isArray(frames)) return null;
  const clean = frames.slice(0, MAX_FRAMES_PER_SLOT).map((frame) =>
    Array.isArray(frame) && frame.length === 21
      ? frame.map((lm) => ({
          x: Number(lm?.x) || 0,
          y: Number(lm?.y) || 0,
          z: Number(lm?.z) || 0,
        }))
      : null
  );
  return clean.every(Boolean) && clean.length > 0 ? clean : null;
};

const cleanMotion = (samples) => {
  if (!Array.isArray(samples)) return null;
  const clean = samples.slice(0, MAX_FRAMES_PER_SLOT).map((s) => ({
    x: Number(s?.x) || 0,
    y: Number(s?.y) || 0,
    z: Number(s?.z) || 0,
    s: Number(s?.s) || 1,
    ...(Number.isFinite(Number(s?.b)) ? { b: Number(s.b) } : {}),
  }));
  return clean.length ? clean : null;
};

/**
 * Rebuilds a backup file into known-shape records. Untrusted JSON is never
 * spread into the database, so unexpected or `__proto__` keys cannot survive.
 */
export function sanitizeImport(parsed) {
  if (!isPlainRecord(parsed) || !parsed.schemaVersion || !Array.isArray(parsed.customSets)) return null;

  const customSets = parsed.customSets
    .filter(isPlainRecord)
    .slice(0, MAX_IMPORT_SETS)
    .map((set) => ({
      id: Number.isFinite(set.id) ? set.id : null,
      title: cleanText(set.title, 120),
      words: (Array.isArray(set.words) ? set.words : [])
        .slice(0, 1000)
        .map((card) => ({
          word: cleanText(typeof card === 'string' ? card : card?.word, 64).toUpperCase(),
          isMultiSign: Boolean(isPlainRecord(card) && card.isMultiSign),
          components: (isPlainRecord(card) && Array.isArray(card.components) ? card.components : [])
            .slice(0, 32)
            .map((component) => cleanText(component, 64))
            .filter(Boolean),
        }))
        .filter((card) => card.word),
    }))
    .filter((set) => set.title && set.words.length);

  const references = (Array.isArray(parsed.references) ? parsed.references : [])
    .filter(isPlainRecord)
    .slice(0, MAX_IMPORT_REFERENCES)
    .map((reference) => ({
      word: cleanText(reference.word, 200),
      timestamp: Number.isFinite(reference.timestamp) ? reference.timestamp : Date.now(),
      frames: cleanFrames(reference.frames),
      frames2: cleanFrames(reference.frames2),
      motion: cleanMotion(reference.motion),
      motion2: cleanMotion(reference.motion2),
    }))
    .filter((reference) => reference.word && (reference.frames || reference.frames2));

  const history = (Array.isArray(parsed.history) ? parsed.history : [])
    .filter(isPlainRecord)
    .slice(0, MAX_IMPORT_HISTORY)
    .map((entry) => ({
      word: cleanText(entry.word, 200),
      timestamp: Number.isFinite(entry.timestamp) ? entry.timestamp : 0,
      status: entry.status === 'correct' ? 'correct' : 'incorrect',
      ...(Number.isFinite(entry.similarity)
        ? { similarity: Math.min(1, Math.max(0, entry.similarity)) }
        : {}),
    }))
    .filter((entry) => entry.word && entry.timestamp);

  return { customSets, references, history };
}
