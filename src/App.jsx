import React, { useEffect, useRef, useState } from 'react';
import Webcam from 'react-webcam';
import { useLiveQuery } from 'dexie-react-hooks';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import { db } from './db';
import {
  ArrowRight,
  Camera,
  CameraOff,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  CircleHelp,
  Download,
  GitBranch,
  Home,
  Layers,
  LoaderCircle,
  Mail,
  MoreHorizontal,
  Moon,
  Pause,
  Pencil,
  Play,
  Plus,
  Radio,
  RotateCcw,
  Settings,
  Shuffle,
  Sun,
  Trash2,
  Upload,
  Users,
  X,
  XCircle,
  Zap,
  BarChart2,
  Cloud,
  CloudOff,
  LogIn,
  LogOut,
  User
} from 'lucide-react';
import {
  onAuthStateChanged, GoogleAuthProvider, signInWithPopup,
  signInWithEmailAndPassword, createUserWithEmailAndPassword,
  signOut as fbSignOut, deleteUser, updatePassword,
} from 'firebase/auth';
import { auth, firebaseConfigured, logAnalyticsEvent } from './firebase';
import {
  downloadAndMerge, downloadSetBaselines, uploadBaseline,
  uploadCustomSet, uploadHistoryEntry, uploadAllLocalData,
} from './syncService';

const PRESETS = {
  fingerspelling: Array.from({ length: 26 }, (_, i) => String.fromCharCode(65 + i)),
  numbers: Array.from({ length: 11 }, (_, i) => String(i))
};

const READY_BUFFER_MS = 1500;
const PASS_THRESHOLD = 0.82; // legacy single-frame fallback
const RECORDING_DURATION_MS = 1000; // ms to record hand motion
const SEQUENCE_PASS_THRESHOLD = 0.65; // combined shape + motion + finger score

// Matching tuning. Lower tolerances and a tighter band mean stricter grading.
const SHAPE_DISTANCE_TOLERANCE = 0.45; // normalized units before a frame scores 0
const MOTION_DISTANCE_TOLERANCE = 1.2; // hand-lengths before a trajectory point scores 0
const MOTION_DYNAMIC_PATH = 0.6; // wrist travel (hand-lengths) that marks a sign as dynamic
const FINGER_TOLERANCE = 0.28; // mean extension difference before the finger score hits 0
const DTW_BAND_RATIO = 0.3; // Sakoe-Chiba warping band as a fraction of sequence length
const MOTION_FAIL_CAP = 0.5; // ceiling when a dynamic sign was performed static
const MOTION_PARTIAL_CAP = 0.64; // ceiling when movement is present but clearly wrong
const TIP_PAIR_TOLERANCE = 0.5; // hand-lengths of fingertip gap difference before scoring 0
const TIP_PAIR_NOTE_DELTA = 0.22; // gap difference that earns a spacing note
const SPLAY_TOLERANCE = 0.5; // radians of splay difference before scoring 0
const CROSSING_TOLERANCE = 0.6; // signed-volume difference before scoring 0
const CROSSING_DEADZONE = 0.15; // near-parallel fingers (U) sit near zero; keep margin before calling it crossed
const CROSSING_FAIL_CAP = 0.6; // ceiling when fingers are crossed the wrong way
const THUMB_TOLERANCE = 0.6; // hand-lengths of thumb-distance difference before scoring 0

// Relative influence of each matching feature; renormalized over enabled features.
const MATCH_WEIGHTS = {
  shape: 0.35,
  motion: 0.22,
  extension: 0.15,
  tipPairs: 0.1,
  splay: 0.07,
  crossing: 0.06,
  thumb: 0.05,
};

const MATCH_FEATURE_LIST = [
  { key: 'shape', label: 'Handshape (DTW)', description: 'Overall landmark match across the whole sign.' },
  { key: 'motion', label: 'Movement path', description: 'Requires dynamic signs to actually move, and static signs to stay put.' },
  { key: 'extension', label: 'Finger extension', description: 'Whether each finger is extended or curled in.' },
  { key: 'tipPairs', label: 'Fingertip spacing', description: 'Gaps between neighbouring fingertips. Separates U from V.' },
  { key: 'splay', label: 'Splay angles', description: 'Angles between finger directions.' },
  { key: 'crossing', label: 'Finger crossing', description: 'Detects crossed index and middle fingers, as in R.' },
  { key: 'thumb', label: 'Thumb position', description: 'Which finger the thumb sits nearest. Separates A, S, T, M and N.' },
];

const DEFAULT_MATCH_FEATURES = Object.fromEntries(MATCH_FEATURE_LIST.map((f) => [f.key, true]));

const readMatchFeatures = (stored) => Object.fromEntries(
  MATCH_FEATURE_LIST.map((f) => [f.key, stored?.[f.key] !== false])
);
const TUTORIAL_STORAGE_KEY = 'asl-signcards-tutorial-seen-v1';
const SETTINGS_STORAGE_KEY = 'asl-signcards-settings-v1';
const MAX_IMPORT_BYTES = 50 * 1024 * 1024;
const MAX_IMPORT_SETS = 500;
const MAX_IMPORT_REFERENCES = 5000;
const MAX_IMPORT_HISTORY = 100000;
const MAX_FRAMES_PER_SLOT = 600;

const isPlainRecord = (value) =>
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
  }));
  return clean.length ? clean : null;
};

/**
 * Rebuilds a backup file into known-shape records. Untrusted JSON is never
 * spread into the database, so unexpected or `__proto__` keys cannot survive.
 */
function sanitizeImport(parsed) {
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

const TUTORIAL_STEPS = [
  { selector: '[data-tour="menu"]', title: 'Choose your deck', body: 'Open the menu to switch sets, create custom cards, view stats, or change settings.' },
  { selector: '[data-tour="phase"]', title: 'Build your baseline', body: 'Start in Baseline Setup. Record each sign once so matching is tuned to your hand.' },
  { selector: '[data-tour="camera"]', title: 'Use the camera view', body: 'Keep your signing hand clearly visible. The live hand landmarks show when the camera can see you.' },
  { selector: '[data-tour="action"]', title: 'Record or check', body: 'In Baseline Setup this button records your reference sign. In Practice Mode it becomes Check My Sign and scores your attempt against that reference.' },
  { selector: '[data-tour="more"]', title: 'Switch modes anytime', body: 'This menu moves you between Baseline Setup and Practice Mode, and holds mirroring, re-recording, and card navigation. Open it when you are ready to practice.' },
];

function normalizeLandmarks(landmarks) {
  if (!landmarks || landmarks.length !== 21) return null;

  const wrist = landmarks[0];
  const centered = landmarks.map((lm) => ({
    x: lm.x - wrist.x,
    y: lm.y - wrist.y,
    z: lm.z - wrist.z
  }));

  const scale = Math.hypot(centered[9].x, centered[9].y, centered[9].z) || 1;

  return centered.map((lm) => ({
    x: lm.x / scale,
    y: lm.y / scale,
    z: lm.z / scale
  }));
}

/**
 * Wrist position plus hand size, captured before wrist-centering discards it.
 * Without this, a static hold and a travelling sign look identical.
 */
function motionSample(landmarks) {
  if (!landmarks || landmarks.length !== 21) return null;
  const wrist = landmarks[0];
  const midMcp = landmarks[9];
  const s = Math.hypot(midMcp.x - wrist.x, midMcp.y - wrist.y, midMcp.z - wrist.z) || 1;
  return { x: wrist.x, y: wrist.y, z: wrist.z, s };
}

const pointDistance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);

const median = (values) => {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

const clamp01 = (value) => Math.max(0, Math.min(1, value));

// Expressed in hand-lengths so results are independent of distance from camera.
function normalizeMotion(samples) {
  if (!samples?.length) return null;
  const scale = median(samples.map((s) => s.s)) || 1;
  const cx = samples.reduce((a, s) => a + s.x, 0) / samples.length;
  const cy = samples.reduce((a, s) => a + s.y, 0) / samples.length;
  const cz = samples.reduce((a, s) => a + s.z, 0) / samples.length;
  return samples.map((s) => ({
    x: (s.x - cx) / scale,
    y: (s.y - cy) / scale,
    z: (s.z - cz) / scale,
  }));
}

function motionFeatures(points) {
  if (!points?.length) return null;
  let path = 0;
  for (let i = 1; i < points.length; i++) path += pointDistance(points[i - 1], points[i]);
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  return {
    path,
    rangeX: Math.max(...xs) - Math.min(...xs),
    rangeY: Math.max(...ys) - Math.min(...ys),
    netX: xs[xs.length - 1] - xs[0],
    netY: ys[ys.length - 1] - ys[0],
  };
}

const FINGERS = [
  { name: 'Thumb', joints: [1, 2, 3, 4] },
  { name: 'Index', joints: [5, 6, 7, 8] },
  { name: 'Middle', joints: [9, 10, 11, 12] },
  { name: 'Ring', joints: [13, 14, 15, 16] },
  { name: 'Pinky', joints: [17, 18, 19, 20] },
];

const FINGER_EXTENDED = 0.82;

/**
 * Straightness per finger: tip-to-knuckle distance over the summed joint chain.
 * ~1.0 when the finger is straight, ~0.4 or lower when curled.
 */
function fingerExtensions(frame) {
  return FINGERS.map(({ joints: [a, b, c, d] }) => {
    const chain =
      pointDistance(frame[a], frame[b]) +
      pointDistance(frame[b], frame[c]) +
      pointDistance(frame[c], frame[d]);
    if (chain <= 1e-6) return 0;
    return Math.min(1, pointDistance(frame[a], frame[d]) / chain);
  });
}

function meanFingerExtensions(frames) {
  if (!frames?.length) return null;
  const sums = [0, 0, 0, 0, 0];
  frames.forEach((frame) => {
    fingerExtensions(frame).forEach((value, i) => { sums[i] += value; });
  });
  return sums.map((sum) => sum / frames.length);
}

const vecSub = (a, b) => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
const vecDot = (a, b) => a.x * b.x + a.y * b.y + a.z * b.z;
const vecCross = (a, b) => ({
  x: a.y * b.z - a.z * b.y,
  y: a.z * b.x - a.x * b.z,
  z: a.x * b.y - a.y * b.x,
});
const vecUnit = (v) => {
  const mag = Math.hypot(v.x, v.y, v.z) || 1;
  return { x: v.x / mag, y: v.y / mag, z: v.z / mag };
};
const angleBetween = (a, b) => Math.acos(Math.max(-1, Math.min(1, vecDot(a, b))));

const TIP_PAIR_LABELS = ['index and middle', 'middle and ring', 'ring and pinky', 'thumb and index'];
const THUMB_TARGETS = ['index', 'middle', 'ring', 'pinky'];

/**
 * Geometry that finger extension alone cannot express: R, U and V all have the
 * index and middle extended and differ only in spacing and crossing.
 * Distances are in hand-lengths, angles in radians.
 */
function handConfiguration(frame) {
  const tipPairs = [
    pointDistance(frame[8], frame[12]),
    pointDistance(frame[12], frame[16]),
    pointDistance(frame[16], frame[20]),
    pointDistance(frame[4], frame[8]),
  ];

  const direction = (mcp, tip) => vecUnit(vecSub(frame[tip], frame[mcp]));
  const dIndex = direction(5, 8);
  const dMiddle = direction(9, 12);
  const dRing = direction(13, 16);
  const dPinky = direction(17, 20);

  const splay = [
    angleBetween(dIndex, dMiddle),
    angleBetween(dMiddle, dRing),
    angleBetween(dRing, dPinky),
  ];

  // Signed volume flips when the index and middle fingers swap sides of the palm.
  const palmNormal = vecUnit(vecCross(vecSub(frame[5], frame[0]), vecSub(frame[17], frame[0])));
  const crossing = vecDot(vecCross(dIndex, dMiddle), palmNormal);

  const thumbDists = [
    pointDistance(frame[4], frame[8]),
    pointDistance(frame[4], frame[12]),
    pointDistance(frame[4], frame[16]),
    pointDistance(frame[4], frame[20]),
  ];

  return { tipPairs, splay, crossing, thumbDists };
}

function meanHandConfiguration(frames) {
  if (!frames?.length) return null;
  const tipPairs = [0, 0, 0, 0];
  const splay = [0, 0, 0];
  const thumbDists = [0, 0, 0, 0];
  let crossing = 0;
  frames.forEach((frame) => {
    const config = handConfiguration(frame);
    config.tipPairs.forEach((v, i) => { tipPairs[i] += v; });
    config.splay.forEach((v, i) => { splay[i] += v; });
    config.thumbDists.forEach((v, i) => { thumbDists[i] += v; });
    crossing += config.crossing;
  });
  const n = frames.length;
  return {
    tipPairs: tipPairs.map((v) => v / n),
    splay: splay.map((v) => v / n),
    thumbDists: thumbDists.map((v) => v / n),
    crossing: crossing / n,
  };
}

const meanAbsoluteScore = (refArr, capArr, tolerance) => {
  const diffs = refArr.map((value, i) => Math.abs(value - capArr[i]));
  return clamp01(1 - (diffs.reduce((a, b) => a + b, 0) / diffs.length) / tolerance);
};

const argMin = (values) => values.reduce((best, v, i) => (v < values[best] ? i : best), 0);

function calculateSimilarity(liveNorm, refNorm) {
  if (!liveNorm || !refNorm) return 0;

  let totalDist = 0;
  for (let i = 0; i < 21; i += 1) {
    const dx = liveNorm[i].x - refNorm[i].x;
    const dy = liveNorm[i].y - refNorm[i].y;
    const dz = liveNorm[i].z - refNorm[i].z;
    totalDist += Math.hypot(dx, dy, dz);
  }

  const avgDist = totalDist / 21;
  return Math.max(0, Math.min(1, 1 - avgDist / SHAPE_DISTANCE_TOLERANCE));
}

/**
 * Reduce sensor jitter with a sliding-window frame average.
 */
function smoothFrames(frames, windowSize = 3) {
  if (frames.length <= windowSize) return frames;
  const half = Math.floor(windowSize / 2);
  return frames.map((_, i) => {
    const start = Math.max(0, i - half);
    const end = Math.min(frames.length - 1, i + half);
    const count = end - start + 1;
    const avg = frames[start].map(() => ({ x: 0, y: 0, z: 0 }));
    for (let k = start; k <= end; k++) {
      frames[k].forEach((pt, j) => {
        avg[j].x += pt.x / count;
        avg[j].y += pt.y / count;
        avg[j].z += pt.z / count;
      });
    }
    return avg;
  });
}

/**
 * Dynamic Time Warping similarity (0–1) with a Sakoe-Chiba band.
 * The band stops one held frame from stretching across an entire moving
 * reference, and cost is divided by real path length so long warps are not
 * rewarded the way a fixed (n + m) divisor did.
 */
function dtwSimilarity(seq1, seq2, frameCost) {
  const n = seq1.length;
  const m = seq2.length;
  if (n === 0 || m === 0) return 0;

  const band = Math.max(4, Math.ceil(Math.max(n, m) * DTW_BAND_RATIO));
  const INF = 1e9;
  let prevCost = new Float32Array(m + 1).fill(INF);
  let prevSteps = new Float32Array(m + 1);
  prevCost[0] = 0;

  for (let i = 1; i <= n; i++) {
    const curCost = new Float32Array(m + 1).fill(INF);
    const curSteps = new Float32Array(m + 1);
    const lo = Math.max(1, i - band);
    const hi = Math.min(m, i + band);
    for (let j = lo; j <= hi; j++) {
      const cost = frameCost(seq1[i - 1], seq2[j - 1]);
      let best = prevCost[j];
      let bestSteps = prevSteps[j];
      if (curCost[j - 1] < best) { best = curCost[j - 1]; bestSteps = curSteps[j - 1]; }
      if (prevCost[j - 1] < best) { best = prevCost[j - 1]; bestSteps = prevSteps[j - 1]; }
      curCost[j] = cost + best;
      curSteps[j] = bestSteps + 1;
    }
    prevCost = curCost;
    prevSteps = curSteps;
  }

  if (prevCost[m] >= INF) return 0;
  const steps = prevSteps[m] || 1;
  return Math.max(0, 1 - prevCost[m] / steps);
}

const shapeFrameCost = (a, b) => 1 - calculateSimilarity(a, b);
const motionFrameCost = (a, b) => Math.min(1, pointDistance(a, b) / MOTION_DISTANCE_TOLERANCE);

/**
 * Compare a captured sequence to a reference sequence.
 * Applies temporal smoothing then DTW.
 */
function compareSequences(captured, reference) {
  return dtwSimilarity(smoothFrames(captured, 3), smoothFrames(reference, 3), shapeFrameCost);
}

/**
 * Scores one hand across the enabled matching features and returns notes
 * explaining what was off. Weights are renormalized over whichever features are
 * enabled and have data, so toggling one off never skews the scale.
 */
function analyzeHand(refFrames, refMotion, capFrames, capMotion, features = DEFAULT_MATCH_FEATURES) {
  const notes = [];
  const parts = [];
  let scoreCap = 1;

  const contribute = (key, score, weight) => {
    if (features[key] === false) return;
    parts.push({ score, weight });
  };

  const shape = compareSequences(capFrames, refFrames);
  contribute('shape', shape, MATCH_WEIGHTS.shape);

  const refExt = meanFingerExtensions(refFrames);
  const capExt = meanFingerExtensions(capFrames);
  if (refExt && capExt) {
    const diffs = refExt.map((value, i) => Math.abs(value - capExt[i]));
    const extensionScore = clamp01(1 - (diffs.reduce((a, b) => a + b, 0) / diffs.length) / FINGER_TOLERANCE);
    contribute('extension', extensionScore, MATCH_WEIGHTS.extension);
    if (features.extension !== false) {
      FINGERS.forEach((finger, i) => {
        const refOut = refExt[i] >= FINGER_EXTENDED;
        const capOut = capExt[i] >= FINGER_EXTENDED;
        if (refOut !== capOut && diffs[i] > 0.12) {
          notes.push(`${finger.name} should be ${refOut ? 'extended' : 'curled in'}.`);
        }
      });
    }
  }

  const refConfig = meanHandConfiguration(refFrames);
  const capConfig = meanHandConfiguration(capFrames);
  if (refConfig && capConfig) {
    const tipScore = meanAbsoluteScore(refConfig.tipPairs, capConfig.tipPairs, TIP_PAIR_TOLERANCE);
    contribute('tipPairs', tipScore, MATCH_WEIGHTS.tipPairs);
    if (features.tipPairs !== false) {
      refConfig.tipPairs.forEach((refGap, i) => {
        const capGap = capConfig.tipPairs[i];
        if (Math.abs(refGap - capGap) > TIP_PAIR_NOTE_DELTA) {
          notes.push(
            capGap > refGap
              ? `Keep your ${TIP_PAIR_LABELS[i]} fingers closer together.`
              : `Spread your ${TIP_PAIR_LABELS[i]} fingers further apart.`
          );
        }
      });
    }

    const splayScore = meanAbsoluteScore(refConfig.splay, capConfig.splay, SPLAY_TOLERANCE);
    contribute('splay', splayScore, MATCH_WEIGHTS.splay);
    if (features.splay !== false && splayScore < 0.5) {
      notes.push('Finger spread differs from your baseline.');
    }

    const crossDelta = Math.abs(refConfig.crossing - capConfig.crossing);
    const crossingScore = clamp01(1 - crossDelta / CROSSING_TOLERANCE);
    contribute('crossing', crossingScore, MATCH_WEIGHTS.crossing);
    if (features.crossing !== false) {
      const refCrossed = refConfig.crossing < -CROSSING_DEADZONE;
      const capCrossed = capConfig.crossing < -CROSSING_DEADZONE;
      if (refCrossed !== capCrossed) {
        notes.push(refCrossed
          ? 'Cross your index and middle fingers.'
          : 'Your index and middle fingers should not be crossed.');
        scoreCap = Math.min(scoreCap, CROSSING_FAIL_CAP);
      }
    }

    const thumbScore = meanAbsoluteScore(refConfig.thumbDists, capConfig.thumbDists, THUMB_TOLERANCE);
    contribute('thumb', thumbScore, MATCH_WEIGHTS.thumb);
    if (features.thumb !== false) {
      const refNearest = argMin(refConfig.thumbDists);
      const capNearest = argMin(capConfig.thumbDists);
      if (refNearest !== capNearest && thumbScore < 0.75) {
        notes.push(`Thumb should sit nearest your ${THUMB_TARGETS[refNearest]} finger.`);
      }
    }
  }

  const refPoints = normalizeMotion(refMotion);
  const capPoints = normalizeMotion(capMotion);
  let motionScore = null;

  if (refPoints && capPoints && refPoints.length > 1 && capPoints.length > 1) {
    const refFeat = motionFeatures(refPoints);
    const capFeat = motionFeatures(capPoints);
    const refDynamic = refFeat.path >= MOTION_DYNAMIC_PATH;
    const motionEnabled = features.motion !== false;

    if (refDynamic) {
      const travelRatio = clamp01(capFeat.path / refFeat.path);
      const pathScore = travelRatio >= 0.55 ? 1 : travelRatio / 0.55;
      const shapeOfPath = dtwSimilarity(capPoints, refPoints, motionFrameCost);
      motionScore = 0.5 * pathScore + 0.5 * shapeOfPath;

      if (motionEnabled) {
        if (travelRatio < 0.45) {
          notes.push('This sign needs movement — your hand stayed too still.');
          scoreCap = Math.min(scoreCap, MOTION_FAIL_CAP);
        } else if (refFeat.rangeX > refFeat.rangeY * 1.6 && capFeat.rangeX < refFeat.rangeX * 0.5) {
          notes.push('Expected more side-to-side movement.');
          scoreCap = Math.min(scoreCap, MOTION_PARTIAL_CAP);
        } else if (refFeat.rangeY > refFeat.rangeX * 1.6 && capFeat.rangeY < refFeat.rangeY * 0.5) {
          notes.push('Expected more up-and-down movement.');
          scoreCap = Math.min(scoreCap, MOTION_PARTIAL_CAP);
        } else if (shapeOfPath < 0.6) {
          notes.push('Movement path differs from your baseline.');
        }
      }
    } else {
      const excess = capFeat.path - Math.max(refFeat.path, MOTION_DYNAMIC_PATH * 0.5);
      motionScore = excess <= 0 ? 1 : clamp01(1 - excess / MOTION_DYNAMIC_PATH);
      if (motionEnabled && motionScore < 0.7) {
        notes.push('Hold this sign steadier — it should not travel.');
        if (motionScore < 0.5) scoreCap = Math.min(scoreCap, MOTION_PARTIAL_CAP);
      }
    }
    contribute('motion', motionScore, MATCH_WEIGHTS.motion);
  }

  if (features.shape !== false && shape < 0.6) notes.push('Handshape differs from your baseline.');

  const totalWeight = parts.reduce((sum, p) => sum + p.weight, 0);
  const weighted = totalWeight > 0
    ? parts.reduce((sum, p) => sum + p.score * p.weight, 0) / totalWeight
    : shape;

  return {
    score: Math.min(clamp01(weighted), scoreCap),
    notes,
    hadMotionData: motionScore !== null,
  };
}

function hasRecordedBaseline(reference) {
  return Boolean(reference?.frames || reference?.frames2);
}

const STATS_RANGE_OPTIONS = [3, 7, 14, 30, 90];

// Red at low scores through amber to green at high scores; each bar is one solid step.
function proficiencyBarClass(value) {
  if (value >= 0.8) return 'bg-emerald-500';
  if (value >= 0.65) return 'bg-lime-500';
  if (value >= 0.5) return 'bg-amber-400';
  if (value >= 0.3) return 'bg-orange-500';
  return 'bg-rose-500';
}

function proficiencyTextClass(value) {
  if (value >= 0.8) return 'text-emerald-500';
  if (value >= 0.65) return 'text-lime-600';
  if (value >= 0.5) return 'text-amber-500';
  if (value >= 0.3) return 'text-orange-500';
  return 'text-rose-500';
}

/**
 * Anki-style weighted random: words with lower recent accuracy get higher probability.
 */
function selectNextRandomIndex(currentIdx, words, historyEntries) {
  if (words.length <= 1) return 0;
  const eligible = words.map((_, i) => i).filter((i) => i !== currentIdx);
  if (eligible.length === 0) return currentIdx;
  const scores = eligible.map((i) => {
    const entries = historyEntries
      .filter((h) => h.word === words[i])
      .sort((a, b) => b.timestamp - a.timestamp)
      .slice(0, 10);
    if (entries.length === 0) return 2.0; // unreviewed = high priority
    const correct = entries.filter((h) => h.status === 'correct').length;
    return (1 - correct / entries.length) + 0.15; // min 0.15 so all words have a chance
  });
  const total = scores.reduce((a, b) => a + b, 0);
  let r = Math.random() * total;
  for (let j = 0; j < eligible.length; j++) {
    r -= scores[j];
    if (r <= 0) return eligible[j];
  }
  return eligible[eligible.length - 1];
}

export default function App() {
  const [view, setView] = useState('landing');
  const [currentSet, setCurrentSet] = useState('fingerspelling');
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isMirrored, setIsMirrored] = useState(true);
  const [isSetMenuOpen, setIsSetMenuOpen] = useState(true);
  const [workflowPhase, setWorkflowPhase] = useState('baseline');
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [showDebugLog, setShowDebugLog] = useState(false);
  const [showHandNodes, setShowHandNodes] = useState(true);
  const [settingsHydrated, setSettingsHydrated] = useState(false);
  const [liveLandmarks, setLiveLandmarks] = useState([]);

  const [engineReady, setEngineReady] = useState(false);
  const [engineLoading, setEngineLoading] = useState(false);
  const [engineError, setEngineError] = useState('');
  const [isInterpreterRunning, setIsInterpreterRunning] = useState(false);

  const [liveSimilarity, setLiveSimilarity] = useState(0);
  const [isRecordingReference, setIsRecordingReference] = useState(false);
  const [showReferenceStatus, setShowReferenceStatus] = useState(false);
  const [lastInferenceMs, setLastInferenceMs] = useState(0);
  const [handsDetected, setHandsDetected] = useState(false);
  const [debugLogs, setDebugLogs] = useState([]);
  const [isBuffering, setIsBuffering] = useState(false);
  const [bufferSecondsLeft, setBufferSecondsLeft] = useState(0);
  const [pendingAction, setPendingAction] = useState(null);
  const [showTutorial, setShowTutorial] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [sidebarSection, setSidebarSection] = useState('none');
  const [appPage, setAppPage] = useState('learn');
  const [newSetTitle, setNewSetTitle] = useState('');
  const [newSetCards, setNewSetCards] = useState([{ id: 1, word: '', isMultiSign: false, components: '' }]);
  const [editingSetId, setEditingSetId] = useState(null);
  const [pendingSuccessSet, setPendingSuccessSet] = useState(null);
  const [setDeleteCandidate, setSetDeleteCandidate] = useState(null);
  const [tooltipOpenId, setTooltipOpenId] = useState(null);
  const [tooltipPosition, setTooltipPosition] = useState({ top: 0, left: 0 });
  const [lastBackupAt, setLastBackupAt] = useState(null);
  const [pendingImportData, setPendingImportData] = useState(null);
  const [isOverflowMenuOpen, setIsOverflowMenuOpen] = useState(false);
  const [isActiveRecording, setIsActiveRecording] = useState(false);
  const [activeRecordingProgress, setActiveRecordingProgress] = useState(0);
  const [practiceResult, setPracticeResult] = useState(null);
  const [showRecordingInstructions, setShowRecordingInstructions] = useState(false);
  const [videoAspect, setVideoAspect] = useState(null);
  const [cameraKey, setCameraKey] = useState(0);
  const [cameraLost, setCameraLost] = useState(false);
  const [matchFeatures, setMatchFeatures] = useState(DEFAULT_MATCH_FEATURES);
  const [showMatchFeatures, setShowMatchFeatures] = useState(false);
  const [practiceOrder, setPracticeOrder] = useState('ordered');
  const [showAllBaselinesModal, setShowAllBaselinesModal] = useState(false);
  const [showPracticeWithMissingModal, setShowPracticeWithMissingModal] = useState(false);
  const [showPracticeInstructions, setShowPracticeInstructions] = useState(false);
  const [tutorialStep, setTutorialStep] = useState(0);
  const [tutorialTargetRect, setTutorialTargetRect] = useState(null);
  const [statsSetKey, setStatsSetKey] = useState(null);
  const [statsRangeDays, setStatsRangeDays] = useState(14);
  const [statsSortDir, setStatsSortDir] = useState('weakest');
  const [user, setUser] = useState(null);
  const [authLoading, setAuthLoading] = useState(true);
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [authMode, setAuthMode] = useState('signin');
  const [authEmail, setAuthEmail] = useState('');
  const [authPassword, setAuthPassword] = useState('');
  const [authError, setAuthError] = useState('');
  const [authSubmitting, setAuthSubmitting] = useState(false);
  const [syncStatus, setSyncStatus] = useState('idle');
  const [lastSyncedAt, setLastSyncedAt] = useState(null);
  const [newPassword, setNewPassword] = useState('');
  const [changePasswordError, setChangePasswordError] = useState('');
  const [changePasswordSuccess, setChangePasswordSuccess] = useState(false);
  const [showDeleteAccountConfirm, setShowDeleteAccountConfirm] = useState(false);

  const webcamRef = useRef(null);
  const streamRef = useRef(null);
  const restartCameraRef = useRef(null);
  const handLandmarkerRef = useRef(null);
  const rafRef = useRef(null);
  const bufferTimeoutRef = useRef(null);
  const bufferIntervalRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);
  const stableMatchFramesRef = useRef(0);
  const latestNormalizedHandRef = useRef(null);
  const isInterpreterRunningRef = useRef(false);
  const frameCounterRef = useRef(0);
  const lastStatsLogAtRef = useRef(0);
  const activeReferenceRef = useRef(null);
  const currentWordRef = useRef('');
  const importFileInputRef = useRef(null);
  const recordingFramesRef = useRef({ h1: [], h2: [], m1: [], m2: [] });
  const recordingTotalFramesRef = useRef(0);
  const isActiveRecordingRef = useRef(false);
  const hasShownRecordingInstructionsRef = useRef(false);
  const allReferencesRef = useRef([]);
  const activeWordsRef = useRef([]);
  const currentSetRef = useRef('fingerspelling');
  const baselineSessionHadMissingRef = useRef(false);
  const currentIndexRef = useRef(0);
  const practiceOrderRef = useRef('ordered');
  const modeWordsRef = useRef([]);
  const historyRef = useRef([]);
  const userRef = useRef(null); // mirrors user state for use inside async callbacks

  const allReferencesRaw = useLiveQuery(() => db.references.toArray());
  const allReferences = allReferencesRaw ?? [];
  const referencesLoaded = allReferencesRaw !== undefined;
  const customSets = useLiveQuery(() => db.customSets.toArray()) || [];
  const history = useLiveQuery(() => db.history.toArray()) || [];

  const availableSets = [
    ...Object.keys(PRESETS).map((key) => ({
      key,
      id: null,
      label: key,
      words: PRESETS[key],
      isCustom: false
    })),
    ...customSets.map((set) => {
      const rawWords = Array.isArray(set.words) ? set.words : [];
      const cards = rawWords.map((w) =>
        typeof w === 'string' ? { word: w, isMultiSign: false, components: [] } : w
      );
      return {
        key: `custom:${set.id}`,
        id: set.id,
        label: set.title,
        words: cards.map((c) => c.word),
        cards,
        isCustom: true,
      };
    })
  ];

  const activeSetOption =
    availableSets.find((set) => set.key === currentSet) ||
    availableSets.find((set) => set.key === 'fingerspelling') ||
    availableSets[0];

  const activeWords = activeSetOption?.words?.length ? activeSetOption.words : PRESETS.fingerspelling;
  const getReferenceKey = (word) => `${currentSet}:${word}`;
  const wordsWithBaseline = activeWords.filter((w) =>
    hasRecordedBaseline(allReferences.find((ref) => ref.word === getReferenceKey(w)))
  );
  const modeWords = workflowPhase === 'practice' ? wordsWithBaseline : activeWords;
  const currentWord = modeWords[currentIndex] ?? modeWords[0] ?? '';
  const missingBaselineCount = activeWords.length - wordsWithBaseline.length;
  const activeReference = allReferences.find((ref) => ref.word === getReferenceKey(currentWord));

  const baselineCount = wordsWithBaseline.length;
  const allBaselinesReady = missingBaselineCount === 0;
  const enabledMatchCount = MATCH_FEATURE_LIST.filter((f) => matchFeatures[f.key] !== false).length;

  activeReferenceRef.current = activeReference || null;
  currentWordRef.current = currentWord || '';
  allReferencesRef.current = allReferences;
  activeWordsRef.current = activeWords;
  currentSetRef.current = currentSet;
  currentIndexRef.current = currentIndex;
  practiceOrderRef.current = practiceOrder;
  modeWordsRef.current = modeWords;
  historyRef.current = history;

  const pushDebugLog = (message) => {
    const timestamp = new Date().toLocaleTimeString();
    const line = `[${timestamp}] ${message}`;
    console.info('[ASL Debug]', line);
    setDebugLogs((prev) => [line, ...prev].slice(0, 14));
  };

  const handleValidation = (success, similarity, alwaysAdvance = false) => {
    const histEntry = {
      word: currentWordRef.current,
      timestamp: Date.now(),
      status: success ? 'correct' : 'incorrect',
      ...(similarity !== undefined ? { similarity } : {}),
    };
    db.history.add(histEntry);
    if (userRef.current) uploadHistoryEntry(userRef.current.uid, histEntry);

    pushDebugLog(
      `${success ? 'PASS' : 'FAIL'} on ${currentWordRef.current} at ${similarity !== undefined ? `${(similarity * 100).toFixed(0)}%` : 'manual'} similarity.`
    );

    if (success || alwaysAdvance) {
      const words = modeWordsRef.current;
      if (!words.length) return;
      if (practiceOrderRef.current === 'random') {
        setCurrentIndex(selectNextRandomIndex(currentIndexRef.current, words, historyRef.current));
      } else {
        setCurrentIndex((prev) => (prev + 1) % words.length);
      }
    }
  };

  const clearBufferTimers = () => {
    if (bufferTimeoutRef.current) {
      clearTimeout(bufferTimeoutRef.current);
      bufferTimeoutRef.current = null;
    }
    if (bufferIntervalRef.current) {
      clearInterval(bufferIntervalRef.current);
      bufferIntervalRef.current = null;
    }
  };

  const findFirstMissingBaselineIndex = (refKeySet) =>
    activeWords.findIndex((word) => !refKeySet.has(getReferenceKey(word)));

  const executeRecordBaseline = async () => {
    if (!latestNormalizedHandRef.current) {
      pushDebugLog('Cannot record baseline: no hand currently detected.');
      window.alert('No hand detected. Hold your hand clearly in front of the camera, then try again.');
      return;
    }

    setIsRecordingReference(true);
    recordingFramesRef.current = { h1: [], h2: [], m1: [], m2: [] };
    recordingTotalFramesRef.current = 0;
    isActiveRecordingRef.current = true;
    setIsActiveRecording(true);
    setActiveRecordingProgress(0);

    const recordingStartedAt = Date.now();
    const progressInterval = setInterval(() => {
      setActiveRecordingProgress(Math.min(1, (Date.now() - recordingStartedAt) / RECORDING_DURATION_MS));
    }, 50);

    await new Promise((resolve) => setTimeout(resolve, RECORDING_DURATION_MS));

    clearInterval(progressInterval);
    isActiveRecordingRef.current = false;
    setIsActiveRecording(false);
    setActiveRecordingProgress(0);

    const h1Frames = [...recordingFramesRef.current.h1];
    const h2Frames = [...recordingFramesRef.current.h2];
    const m1Samples = [...recordingFramesRef.current.m1];
    const m2Samples = [...recordingFramesRef.current.m2];
    const totalFrames = recordingTotalFramesRef.current;
    recordingFramesRef.current = { h1: [], h2: [], m1: [], m2: [] };
    recordingTotalFramesRef.current = 0;

    // At least one hand must have been sufficiently present
    const maxFrames = Math.max(h1Frames.length, h2Frames.length);
    if (totalFrames > 0 && maxFrames / totalFrames < 0.5) {
      setIsRecordingReference(false);
      pushDebugLog('Baseline recording failed: hand not visible for enough of the recording.');
      window.alert('Hand not detected for most of the recording. Keep your hand clearly in frame for the full second.');
      return;
    }
    if (maxFrames < 8) {
      setIsRecordingReference(false);
      pushDebugLog('Baseline recording failed: too few frames captured.');
      window.alert('Not enough data captured. Make sure the camera is working and try again.');
      return;
    }

    // Preserve handedness slots directly — Left hand → frames, Right hand → frames2
    const saveH1 = h1Frames.length >= 8 ? h1Frames : null;
    const saveH2 = h2Frames.length >= 8 ? h2Frames : null;
    const saveM1 = saveH1 && m1Samples.length >= 8 ? m1Samples : null;
    const saveM2 = saveH2 && m2Samples.length >= 8 ? m2Samples : null;
    const handLabel = saveH1 && saveH2 ? '2-hand' : saveH1 ? 'left' : 'right';
    const currentKey = getReferenceKey(currentWordRef.current);
    await db.references.put({
      word: currentKey,
      timestamp: Date.now(),
      frames: saveH1,
      ...(saveH2 ? { frames2: saveH2 } : {}),
      ...(saveM1 ? { motion: saveM1 } : {}),
      ...(saveM2 ? { motion2: saveM2 } : {}),
    });
    if (userRef.current) {
      uploadBaseline(userRef.current.uid, currentKey, saveH1, saveH2, Date.now(), saveM1, saveM2).catch(() => {});
    }
    setIsRecordingReference(false);
    logAnalyticsEvent('baseline_recorded', { word: currentWordRef.current, hand: handLabel, set: currentSetRef.current });
    pushDebugLog(`Baseline recorded for "${currentWordRef.current}" (${handLabel}, ${maxFrames}/${totalFrames} frames).`);

    const freshRefs = allReferencesRef.current;
    const freshWords = activeWordsRef.current;
    const freshSet = currentSetRef.current;
    const updatedKeys = new Set(freshRefs.filter(hasRecordedBaseline).map((ref) => ref.word));
    updatedKeys.add(currentKey);
    const nextMissingIndex = freshWords.findIndex((word) => !updatedKeys.has(`${freshSet}:${word}`));

    if (nextMissingIndex === -1) {
      if (baselineSessionHadMissingRef.current) {
        // Newly completed all baselines — prompt the user
        baselineSessionHadMissingRef.current = false;
        pushDebugLog(`All baselines captured for ${freshSet}.`);
        setShowAllBaselinesModal(true);
      } else {
        // Re-recording mode — cycle to next word
        const currentIdx = freshWords.findIndex((w) => w === currentWordRef.current);
        setCurrentIndex((currentIdx + 1) % freshWords.length);
      }
    } else {
      setCurrentIndex(nextMissingIndex);
    }
  };

  const executeCheckAnswer = async () => {
    const ref = activeReferenceRef.current;
    if (!ref?.frames && !ref?.frames2) {
      const msg = ref
        ? 'This baseline was recorded in the old format. Please re-record it to use dynamic matching.'
        : 'No baseline recorded for this sign yet.';
      pushDebugLog(`Cannot check answer: ${msg}`);
      window.alert(msg);
      return;
    }

    if (!latestNormalizedHandRef.current) {
      pushDebugLog('Cannot check answer: no hand currently detected.');
      window.alert('No hand detected. Hold your hand clearly in front of the camera, then try again.');
      return;
    }

    recordingFramesRef.current = { h1: [], h2: [], m1: [], m2: [] };
    recordingTotalFramesRef.current = 0;
    isActiveRecordingRef.current = true;
    setIsActiveRecording(true);
    setActiveRecordingProgress(0);

    const recordingStartedAt = Date.now();
    const progressInterval = setInterval(() => {
      setActiveRecordingProgress(Math.min(1, (Date.now() - recordingStartedAt) / RECORDING_DURATION_MS));
    }, 50);

    await new Promise((resolve) => setTimeout(resolve, RECORDING_DURATION_MS));

    clearInterval(progressInterval);
    isActiveRecordingRef.current = false;
    setIsActiveRecording(false);
    setActiveRecordingProgress(0);

    const h1Check = [...recordingFramesRef.current.h1];
    const h2Check = [...recordingFramesRef.current.h2];
    const m1Check = [...recordingFramesRef.current.m1];
    const m2Check = [...recordingFramesRef.current.m2];
    const totalFrames = recordingTotalFramesRef.current;
    recordingFramesRef.current = { h1: [], h2: [], m1: [], m2: [] };
    recordingTotalFramesRef.current = 0;

    const maxCapFrames = Math.max(h1Check.length, h2Check.length);
    if (totalFrames > 0 && maxCapFrames / totalFrames < 0.5) {
      pushDebugLog('Check failed: hand not visible for enough of the recording.');
      window.alert('Hand not detected for most of the recording. Keep your hand clearly in frame for the full second.');
      return;
    }
    if (maxCapFrames < 8) {
      pushDebugLog('Check failed: too few frames captured.');
      window.alert('Not enough data captured. Make sure the camera is working and try again.');
      return;
    }

    const refH1 = ref.frames;
    const refH2 = ref.frames2;

    const scoreHands = (mirror) => {
      const results = [];
      const capA = mirror ? h2Check : h1Check;
      const capB = mirror ? h1Check : h2Check;
      const capMotionA = mirror ? m2Check : m1Check;
      const capMotionB = mirror ? m1Check : m2Check;
      if (refH1 && refH1.length >= 4 && capA.length >= 8) {
        results.push(analyzeHand(refH1, ref.motion, capA, capMotionA, matchFeatures));
      }
      if (refH2 && refH2.length >= 4 && capB.length >= 8) {
        results.push(analyzeHand(refH2, ref.motion2, capB, capMotionB, matchFeatures));
      }
      const score = results.length
        ? results.reduce((sum, r) => sum + r.score, 0) / results.length
        : 0;
      return { score, results };
    };

    const regular = scoreHands(false);
    const mirrored = scoreHands(true);
    const usedMirror = mirrored.score > regular.score;
    const best = usedMirror ? mirrored : regular;
    const similarity = best.score;

    const notes = [...new Set(best.results.flatMap((r) => r.notes))].slice(0, 4);
    const legacyBaseline = best.results.length > 0 && best.results.every((r) => !r.hadMotionData);
    if (legacyBaseline) {
      notes.push('Re-record this baseline to enable movement checking.');
    }

    const mirrorTag = usedMirror ? ' [mirrored]' : '';
    const verdict = similarity >= SEQUENCE_PASS_THRESHOLD ? 'PASS' : 'FAIL';
    pushDebugLog(
      `Check${mirrorTag} for "${currentWordRef.current}": ${(similarity * 100).toFixed(0)}% (${verdict}, ${best.results.length} hand(s), ${maxCapFrames} frames)${notes.length ? ` — ${notes.join(' ')}` : ''}`
    );

    const passed = similarity >= SEQUENCE_PASS_THRESHOLD;
    logAnalyticsEvent('practice_check', { word: currentWordRef.current, passed, similarity: Math.round(similarity * 100), set: currentSetRef.current });
    setPracticeResult({ similarity, passed, notes });
  };

  const startBufferedAction = async (actionType) => {
    if (!isInterpreterRunningRef.current) {
      await startInterpreter();
      if (!isInterpreterRunningRef.current) return;
    }

    if (isBuffering) return;

    if (actionType === 'record' && !hasShownRecordingInstructionsRef.current) {
      hasShownRecordingInstructionsRef.current = true;
      setShowRecordingInstructions(true);
      return;
    }

    setPendingAction(actionType);
    setIsBuffering(true);
    setBufferSecondsLeft(READY_BUFFER_MS / 1000);
    pushDebugLog(`Starting ${READY_BUFFER_MS / 1000}s get-ready buffer for ${actionType}.`);

    const startedAt = Date.now();
    bufferIntervalRef.current = setInterval(() => {
      const elapsed = Date.now() - startedAt;
      const remainingMs = Math.max(0, READY_BUFFER_MS - elapsed);
      setBufferSecondsLeft(remainingMs / 1000);
    }, 100);

    bufferTimeoutRef.current = setTimeout(async () => {
      clearBufferTimers();
      setIsBuffering(false);
      setBufferSecondsLeft(0);
      const action = actionType;
      setPendingAction(null);

      if (action === 'record') {
        await executeRecordBaseline();
      } else if (action === 'check') {
        await executeCheckAnswer();
      }
    }, READY_BUFFER_MS);
  };

  const handleCameraLost = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setCameraLost(true);
    setVideoAspect(null);
    if (isInterpreterRunningRef.current) stopInterpreter();
    pushDebugLog('Camera disconnected.');
  };

  // react-webcam only calls getUserMedia on mount, so remounting it re-acquires the device.
  const restartCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (isInterpreterRunningRef.current) stopInterpreter();
    setVideoAspect(null);
    setEngineError('');
    setCameraLost(false);
    setCameraKey((key) => key + 1);
    pushDebugLog('Restarting camera...');
  };

  restartCameraRef.current = restartCamera;

  const handleUserMediaError = (error) => {
    const message = typeof error === 'string' ? error : (error?.message || 'Camera unavailable.');
    setCameraLost(true);
    setEngineError(message);
    pushDebugLog(`Camera error: ${message}`);
  };

  const handleUserMedia = (stream) => {
    streamRef.current = stream ?? null;
    setCameraLost(false);
    setEngineError('');
    stream?.getVideoTracks().forEach((track) => {
      track.addEventListener('ended', handleCameraLost, { once: true });
    });
    const trySetAspect = () => {
      const video = webcamRef.current?.video;
      if (video && video.videoWidth > 0 && video.videoHeight > 0) {
        setVideoAspect(video.videoWidth / video.videoHeight);
      } else {
        requestAnimationFrame(trySetAspect);
      }
    };
    requestAnimationFrame(trySetAspect);
  };

  const handleSetSelect = (setKey) => {
    logAnalyticsEvent('select_set', { set_id: setKey });
    setCurrentSet(setKey);
    setCurrentIndex(0);
    setShowReferenceStatus(false);
    setLiveSimilarity(0);
    setIsSetMenuOpen(false);
    setIsSidebarOpen(false);
    setSidebarSection('none');
    setAppPage('learn');
    setView('app');
    if (userRef.current) {
      downloadSetBaselines(userRef.current.uid, setKey, db).catch(() => {});
    }
  };

  const resetCreateForm = () => {
    setNewSetTitle('');
    setNewSetCards([{ id: Date.now(), word: '', isMultiSign: false, components: '' }]);
    setEditingSetId(null);
  };

  const makeCardObjects = (cards) =>
    cards
      .filter((c) => c.word.trim())
      .map((c) => ({
        word: c.word.trim().toUpperCase().replace(/[#[\]*/\\?]/g, '').slice(0, 64),
        isMultiSign: c.isMultiSign,
        components: c.components
          .split(',')
          .map((s) => s.trim())
          .filter(Boolean),
      }));

  const handleCreateSet = async () => {
    const title = newSetTitle.trim();
    const cardObjects = makeCardObjects(newSetCards);

    if (!title) {
      window.alert('Please enter a title for the set.');
      return;
    }
    if (!cardObjects.length) {
      window.alert('Please add at least one card.');
      return;
    }

    const newId = await db.customSets.add({ title, words: cardObjects });
    if (userRef.current) {
      uploadCustomSet(userRef.current.uid, { id: newId, title, words: cardObjects }).catch(() => {});
    }
    logAnalyticsEvent('create_set', { word_count: cardObjects.length });
    resetCreateForm();
    setPendingSuccessSet({ id: newId, title, wordCount: cardObjects.length, isEdit: false });
    pushDebugLog(`Created custom set "${title}" with ${cardObjects.length} cards.`);
  };

  const handleUpdateSet = async () => {
    const title = newSetTitle.trim();
    const cardObjects = makeCardObjects(newSetCards);

    if (!title) {
      window.alert('Please enter a title for the set.');
      return;
    }
    if (!cardObjects.length) {
      window.alert('Please add at least one card.');
      return;
    }

    const setKey = `custom:${editingSetId}`;
    const oldSet = customSets.find((s) => s.id === editingSetId);
    const oldWords = (Array.isArray(oldSet?.words) ? oldSet.words : []).map((w) =>
      typeof w === 'string' ? w : w.word
    );
    const newWords = cardObjects.map((c) => c.word);

    const removedWords = oldWords.filter((w) => !newWords.includes(w));
    for (const word of removedWords) {
      const keys = await db.references.where('word').equals(`${setKey}:${word}`).primaryKeys();
      if (keys.length) await db.references.bulkDelete(keys);
    }

    await db.customSets.update(editingSetId, { title, words: cardObjects });
    if (userRef.current) {
      uploadCustomSet(userRef.current.uid, { id: editingSetId, title, words: cardObjects }).catch(() => {});
    }

    const missingCount = newWords.filter(
      (w) => !allReferences.some((ref) => ref.word === `${setKey}:${w}`)
    ).length;

    const savedId = editingSetId;
    resetCreateForm();
    setPendingSuccessSet({ id: savedId, title, wordCount: newWords.length, newCount: missingCount, isEdit: true });
    pushDebugLog(`Updated custom set "${title}".`);
  };

  const handleSignInGoogle = async () => {
    if (!auth) return;
    setAuthError('');
    setAuthSubmitting(true);
    try {
      await signInWithPopup(auth, new GoogleAuthProvider());
      logAnalyticsEvent('login', { method: 'google' });
      setShowAuthModal(false);
      setAuthEmail('');
      setAuthPassword('');
    } catch (e) {
      setAuthError(e.code === 'auth/popup-closed-by-user' ? 'Sign-in was cancelled.' : (e.message || 'Sign-in failed.'));
    } finally {
      setAuthSubmitting(false);
    }
  };

  const handleSignInEmail = async () => {
    if (!auth) return;
    setAuthError('');
    setAuthSubmitting(true);
    try {
      await signInWithEmailAndPassword(auth, authEmail.trim(), authPassword);
      logAnalyticsEvent('login', { method: 'email' });
      setShowAuthModal(false);
      setAuthEmail('');
      setAuthPassword('');
    } catch (e) {
      setAuthError(
        e.code === 'auth/invalid-credential' || e.code === 'auth/wrong-password' || e.code === 'auth/user-not-found'
          ? 'Incorrect email or password.'
          : (e.message || 'Sign-in failed.')
      );
    } finally {
      setAuthSubmitting(false);
    }
  };

  const handleSignUpEmail = async () => {
    if (!auth) return;
    if (authPassword.length < 8) { setAuthError('Password must be at least 8 characters.'); return; }
    setAuthError('');
    setAuthSubmitting(true);
    try {
      await createUserWithEmailAndPassword(auth, authEmail.trim(), authPassword);
      logAnalyticsEvent('sign_up', { method: 'email' });
      setShowAuthModal(false);
      setAuthEmail('');
      setAuthPassword('');
    } catch (e) {
      setAuthError(
        e.code === 'auth/email-already-in-use'
          ? 'An account with this email already exists.'
          : (e.message || 'Sign-up failed.')
      );
    } finally {
      setAuthSubmitting(false);
    }
  };

  const handleSignOut = async () => {
    if (!auth) return;
    logAnalyticsEvent('logout');
    await fbSignOut(auth);
    setUser(null);
    userRef.current = null;
    setSyncStatus('idle');
    setLastSyncedAt(null);
    setAppPage('learn');
    setIsSidebarOpen(false);
  };

  const handleSyncNow = async () => {
    if (!userRef.current) return;
    setSyncStatus('syncing');
    try {
      await uploadAllLocalData(userRef.current.uid, db);
      await downloadAndMerge(userRef.current.uid, db);
      setSyncStatus('done');
      setLastSyncedAt(Date.now());
    } catch {
      setSyncStatus('error');
    }
  };

  const handleChangePassword = async () => {
    if (!auth?.currentUser) return;
    if (newPassword.length < 8) { setChangePasswordError('Password must be at least 8 characters.'); return; }
    setChangePasswordError('');
    try {
      await updatePassword(auth.currentUser, newPassword);
      setNewPassword('');
      setChangePasswordSuccess(true);
      setTimeout(() => setChangePasswordSuccess(false), 3000);
    } catch (e) {
      setChangePasswordError(
        e.code === 'auth/requires-recent-login'
          ? 'Please sign out and sign back in before changing your password.'
          : (e.message || 'Failed to update password.')
      );
    }
  };

  const handleDeleteAccount = async (clearLocal) => {
    if (!auth?.currentUser) return;
    try {
      if (clearLocal) {
        await Promise.all([db.customSets.clear(), db.references.clear(), db.history.clear()]);
      }
      await deleteUser(auth.currentUser);
      setUser(null);
      userRef.current = null;
      setShowDeleteAccountConfirm(false);
      setSyncStatus('idle');
      setLastSyncedAt(null);
      setAppPage('learn');
      setIsSidebarOpen(false);
    } catch (e) {
      window.alert(
        e.code === 'auth/requires-recent-login'
          ? 'Please sign out and sign back in before deleting your account.'
          : ('Failed to delete account: ' + (e.message || 'Unknown error'))
      );
    }
  };

  const handleEditSet = (setOption) => {
    const rawSet = customSets.find((s) => s.id === setOption.id);
    const rawWords = Array.isArray(rawSet?.words) ? rawSet.words : [];
    const loadedCards = rawWords.map((w) => ({
      id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
      word: typeof w === 'string' ? w : (w.word || ''),
      isMultiSign: typeof w === 'object' && !Array.isArray(w) ? (w.isMultiSign || false) : false,
      components:
        typeof w === 'object' && !Array.isArray(w) && Array.isArray(w.components)
          ? w.components.join(', ')
          : '',
    }));
    setEditingSetId(setOption.id);
    setNewSetTitle(setOption.label);
    setNewSetCards(
      loadedCards.length
        ? loadedCards
        : [{ id: `${Date.now()}`, word: '', isMultiSign: false, components: '' }]
    );
    setAppPage('create');
    setIsSidebarOpen(false);
  };

  const handleDeleteSet = async (candidate) => {
    if (!candidate?.isCustom || candidate.id == null) return;

    await db.customSets.delete(candidate.id);
    const referencePrimaryKeys = await db.references.where('word').startsWith(`${candidate.key}:`).primaryKeys();
    if (referencePrimaryKeys.length) {
      await db.references.bulkDelete(referencePrimaryKeys);
    }

    if (currentSet === candidate.key) {
      setCurrentSet('fingerspelling');
      setCurrentIndex(0);
    }

    setSetDeleteCandidate(null);
    pushDebugLog(`Deleted custom set "${candidate.label}".`);
  };

  const handleExportData = async () => {
    const [sets, refs, hist] = await Promise.all([
      db.customSets.toArray(),
      db.references.toArray(),
      db.history.toArray(),
    ]);

    const payload = {
      schemaVersion: 1,
      exportedAt: Date.now(),
      customSets: sets,
      references: refs,
      history: hist,
    };

    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `asl-signcards-backup-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    setLastBackupAt(Date.now());
    pushDebugLog('Data exported successfully.');
  };

  const handleImportFile = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    event.target.value = '';

    if (file.size > MAX_IMPORT_BYTES) {
      window.alert('That backup file is too large to import (limit 50 MB).');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const sanitized = sanitizeImport(JSON.parse(e.target.result));
        if (!sanitized) {
          window.alert('Invalid backup file. Please use a file exported from SignCards.');
          return;
        }
        setPendingImportData(sanitized);
      } catch {
        window.alert('Could not read the file. Make sure it is a valid SignCards backup (.json).');
      }
    };
    reader.readAsText(file);
  };

  const handleImportConfirm = async (mode) => {
    const data = pendingImportData;
    if (!data) return;

    const idRemap = {};

    if (mode === 'replace') {
      await db.customSets.clear();
      await db.references.clear();
      await db.history.clear();
    }

    for (const set of (data.customSets || [])) {
      if (mode === 'merge') {
        const existing = await db.customSets.where('title').equals(set.title).first();
        if (existing) {
          idRemap[`custom:${set.id}`] = `custom:${existing.id}`;
          continue;
        }
      }
      const { id: _oldId, ...setWithoutId } = set;
      const newId = await db.customSets.add(setWithoutId);
      idRemap[`custom:${set.id}`] = `custom:${newId}`;
    }

    for (const ref of (data.references || [])) {
      let key = ref.word;
      for (const [oldPrefix, newPrefix] of Object.entries(idRemap)) {
        if (key.startsWith(`${oldPrefix}:`)) {
          key = newPrefix + key.slice(oldPrefix.length);
          break;
        }
      }
      if (mode === 'merge') {
        const exists = await db.references.get(key);
        if (exists) continue;
      }
      await db.references.put({ ...ref, word: key });
    }

    if ((data.history || []).length) {
      const rows = data.history.map(({ id: _id, ...rest }) => rest);
      await db.history.bulkAdd(rows);
    }

    setPendingImportData(null);
    pushDebugLog(`Import complete (${mode} mode). ${(data.customSets || []).length} set(s) processed.`);
    window.alert(`Import complete! ${(data.customSets || []).length} set(s) imported.`);
  };

  const enterPractice = () => {
    setWorkflowPhase('practice');
    setCurrentIndex(0);
    setPracticeResult(null);
    if (!hasShownRecordingInstructionsRef.current) {
      setShowPracticeInstructions(true);
      hasShownRecordingInstructionsRef.current = true;
    }
  };

  const switchToPracticeIfReady = () => {
    if (wordsWithBaseline.length === 0) {
      window.alert('Record at least one baseline before switching to practice.');
      return;
    }
    if (!allBaselinesReady) {
      setShowPracticeWithMissingModal(true);
      return;
    }
    enterPractice();
  };

  const switchToBaseline = () => {
    hasShownRecordingInstructionsRef.current = false;
    const refs = allReferencesRef.current;
    const words = activeWordsRef.current;
    const set = currentSetRef.current;
    const missingCount = words.filter((w) => !hasRecordedBaseline(refs.find((ref) => ref.word === `${set}:${w}`))).length;
    baselineSessionHadMissingRef.current = missingCount > 0;
    setWorkflowPhase('baseline');
    setPracticeResult(null);
    const existingKeys = new Set(refs.filter(hasRecordedBaseline).map((ref) => ref.word));
    const nextMissingIndex = words.findIndex((word) => !existingKeys.has(`${set}:${word}`));
    setCurrentIndex(nextMissingIndex === -1 ? 0 : nextMissingIndex);
  };

  const stopInterpreter = () => {
    clearBufferTimers();
    setIsBuffering(false);
    setPendingAction(null);
    setBufferSecondsLeft(0);
    isInterpreterRunningRef.current = false;
    setIsInterpreterRunning(false);
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }

    isActiveRecordingRef.current = false;
    setIsActiveRecording(false);
    setActiveRecordingProgress(0);
    recordingFramesRef.current = { h1: [], h2: [], m1: [], m2: [] };
    recordingTotalFramesRef.current = 0;
    stableMatchFramesRef.current = 0;
    lastVideoTimeRef.current = -1;
    setHandsDetected(false);
    setLiveLandmarks([]);
    setLastInferenceMs(0);
    setLiveSimilarity(0);
    pushDebugLog('Interpreter stopped.');
  };

  const initInterpreter = async () => {
    if (handLandmarkerRef.current) {
      setEngineReady(true);
      return;
    }

    setEngineLoading(true);
    setEngineError('');

    try {
      pushDebugLog('Initializing MediaPipe hand landmarker...');

      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
      );

      handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
        },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.35,
        minHandPresenceConfidence: 0.35,
        minTrackingConfidence: 0.35
      });

      setEngineReady(true);
      pushDebugLog('MediaPipe initialized successfully.');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to initialize ASL engine.';
      setEngineError(message);
      setEngineReady(false);
      pushDebugLog(`Initialization failed: ${message}`);
    } finally {
      setEngineLoading(false);
    }
  };

  const runInterpreterFrame = () => {
    if (!isInterpreterRunningRef.current || !handLandmarkerRef.current) return;

    const video = webcamRef.current?.video;

    if (video && video.readyState >= 2) {
      if (video.currentTime !== lastVideoTimeRef.current) {
        lastVideoTimeRef.current = video.currentTime;
        frameCounterRef.current += 1;
        if (isActiveRecordingRef.current) {
          recordingTotalFramesRef.current += 1;
        }

        const startedAt = performance.now();

        try {
          const results = handLandmarkerRef.current.detectForVideo(video, performance.now());
          const ref = activeReferenceRef.current;

          // Slot hands by handedness: Left → h1 (slot 0), Right → h2 (slot 1)
          const handSlots = [null, null];
          for (let i = 0; i < (results.landmarks?.length || 0); i++) {
            const cat = results.handedness?.[i]?.[0]?.categoryName; // "Left" or "Right"
            const slot = cat === 'Left' ? 0 : 1;
            if (!handSlots[slot]) handSlots[slot] = results.landmarks[i];
          }
          const rawH1 = handSlots[0]; // left hand
          const rawH2 = handSlots[1]; // right hand
          const anyHand = rawH1 || rawH2;

          if (anyHand) {
            setHandsDetected(true);
            setLiveLandmarks([...(rawH1 || []), ...(rawH2 || [])]);
            const normH1 = rawH1 ? normalizeLandmarks(rawH1) : null;
            const normH2 = rawH2 ? normalizeLandmarks(rawH2) : null;
            latestNormalizedHandRef.current = normH1 || normH2;
            if (isActiveRecordingRef.current) {
              if (normH1) {
                recordingFramesRef.current.h1.push(normH1);
                const sample = motionSample(rawH1);
                if (sample) recordingFramesRef.current.m1.push(sample);
              }
              if (normH2) {
                recordingFramesRef.current.h2.push(normH2);
                const sample = motionSample(rawH2);
                if (sample) recordingFramesRef.current.m2.push(sample);
              }
            }

            const normalized = normH1 || normH2;
            if (normalized && ref?.landmarks) {
              const similarity = calculateSimilarity(normalized, ref.landmarks);
              setLiveSimilarity(similarity);

              if (similarity > 0.82) {
                stableMatchFramesRef.current += 1;
                if (stableMatchFramesRef.current > 15) {
                  stableMatchFramesRef.current = 0;
                  pushDebugLog(
                    `Auto-pass for "${currentWordRef.current}" at ${(similarity * 100).toFixed(0)}% similarity.`
                  );
                  handleValidation(true);
                }
              } else {
                stableMatchFramesRef.current = 0;
              }
            } else {
              setLiveSimilarity(0);
              stableMatchFramesRef.current = 0;
            }
          } else {
            setHandsDetected(false);
            setLiveLandmarks([]);
            latestNormalizedHandRef.current = null;
            setLiveSimilarity(0);
            stableMatchFramesRef.current = 0;
          }

          const inferenceMs = performance.now() - startedAt;
          setLastInferenceMs(inferenceMs);

          if (performance.now() - lastStatsLogAtRef.current > 2500) {
            lastStatsLogAtRef.current = performance.now();
            pushDebugLog(
              `Frame ${frameCounterRef.current}: hands=${anyHand ? (rawH1 && rawH2 ? '2' : '1') : '0'}, baseline=${ref ? 'yes' : 'no'}, readyState=${video.readyState}, ${inferenceMs.toFixed(1)}ms`
            );
          }
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown inference error';
          pushDebugLog(`Inference failed: ${message}`);
          setEngineError(message);
          stopInterpreter();
          return;
        }
      }
    } else if (video && performance.now() - lastStatsLogAtRef.current > 2500) {
      lastStatsLogAtRef.current = performance.now();
      const tracks = streamRef.current?.getVideoTracks() ?? [];
      // Some drivers drop the device without ever firing 'ended'.
      if (tracks.length && !tracks.some((track) => track.readyState === 'live')) {
        handleCameraLost();
        return;
      }
      pushDebugLog(`Waiting for webcam stream. readyState=${video.readyState}`);
    }

    if (isInterpreterRunningRef.current) {
      rafRef.current = requestAnimationFrame(runInterpreterFrame);
    }
  };

  const startInterpreter = async () => {
    if (cameraLost) {
      pushDebugLog('Cannot start interpreter: camera is disconnected.');
      return;
    }
    await initInterpreter();
    if (!handLandmarkerRef.current) return;

    const video = webcamRef.current?.video;
    if (!video) {
      const message = 'Webcam element not ready yet. Please wait one second and try again.';
      setEngineError(message);
      pushDebugLog(message);
      return;
    }

    setEngineError('');
    isInterpreterRunningRef.current = true;
    setIsInterpreterRunning(true);
    frameCounterRef.current = 0;
    stableMatchFramesRef.current = 0;
    setLiveSimilarity(0);
    pushDebugLog('Interpreter started.');
    rafRef.current = requestAnimationFrame(runInterpreterFrame);
  };

  const toggleInterpreter = async () => {
    if (isInterpreterRunningRef.current) {
      stopInterpreter();
      return;
    }

    await startInterpreter();
  };

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(SETTINGS_STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (typeof parsed.isDarkMode === 'boolean') setIsDarkMode(parsed.isDarkMode);
        if (typeof parsed.isMirrored === 'boolean') setIsMirrored(parsed.isMirrored);
        if (typeof parsed.showHandNodes === 'boolean') setShowHandNodes(parsed.showHandNodes);
        if (typeof parsed.showDebugLog === 'boolean') setShowDebugLog(parsed.showDebugLog);
        if (typeof parsed.lastBackupAt === 'number') setLastBackupAt(parsed.lastBackupAt);
        if (isPlainRecord(parsed.matchFeatures)) setMatchFeatures(readMatchFeatures(parsed.matchFeatures));
      }
    } catch (error) {
      console.warn('Failed to read persisted settings.', error);
    } finally {
      setSettingsHydrated(true);
    }
  }, []);

  useEffect(() => {
    if (!settingsHydrated) return;
    const payload = {
      isDarkMode,
      isMirrored,
      showHandNodes,
      showDebugLog,
      lastBackupAt,
      matchFeatures,
    };
    window.localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(payload));
  }, [isDarkMode, isMirrored, showHandNodes, showDebugLog, lastBackupAt, matchFeatures, settingsHydrated]);

  useEffect(() => {
    const handleStorage = (event) => {
      if (event.key !== SETTINGS_STORAGE_KEY || !event.newValue) return;
      try {
        const parsed = JSON.parse(event.newValue);
        if (typeof parsed.isDarkMode === 'boolean') setIsDarkMode(parsed.isDarkMode);
        if (typeof parsed.isMirrored === 'boolean') setIsMirrored(parsed.isMirrored);
        if (typeof parsed.showHandNodes === 'boolean') setShowHandNodes(parsed.showHandNodes);
        if (typeof parsed.showDebugLog === 'boolean') setShowDebugLog(parsed.showDebugLog);
        if (typeof parsed.lastBackupAt === 'number') setLastBackupAt(parsed.lastBackupAt);
        if (isPlainRecord(parsed.matchFeatures)) setMatchFeatures(readMatchFeatures(parsed.matchFeatures));
      } catch (error) {
        console.warn('Failed to sync settings from another tab.', error);
      }
    };

    window.addEventListener('storage', handleStorage);
    return () => window.removeEventListener('storage', handleStorage);
  }, []);

  useEffect(() => {
    if (view === 'landing' && isInterpreterRunningRef.current) {
      stopInterpreter();
    }
    if (view === 'landing') {
      setIsSidebarOpen(false);
      setSidebarSection('none');
    }
  }, [view]);

  useEffect(() => {
    if (view !== 'app') return;
    const hasSeenTutorial = window.localStorage.getItem(TUTORIAL_STORAGE_KEY) === '1';
    if (!hasSeenTutorial) {
      setShowTutorial(true);
    }
  }, [view]);

  useEffect(() => {
    if (!showTutorial || view !== 'app') return undefined;
    const updateTutorialTarget = () => {
      const target = document.querySelector(TUTORIAL_STEPS[tutorialStep]?.selector);
      if (!target) {
        setTutorialTargetRect(null);
        return;
      }
      const rect = target.getBoundingClientRect();
      setTutorialTargetRect({ top: rect.top, left: rect.left, width: rect.width, height: rect.height });
    };
    updateTutorialTarget();
    window.addEventListener('resize', updateTutorialTarget);
    return () => window.removeEventListener('resize', updateTutorialTarget);
  }, [showTutorial, tutorialStep, view, workflowPhase]);

  useEffect(() => {
    if (!referencesLoaded) return;
    hasShownRecordingInstructionsRef.current = false;
    const refs = allReferencesRef.current;
    const words = activeWordsRef.current;
    const set = currentSet;

    const setReady = words.every((word) =>
      hasRecordedBaseline(refs.find((ref) => ref.word === `${set}:${word}`))
    );

    if (setReady) {
      baselineSessionHadMissingRef.current = false;
      setWorkflowPhase('practice');
      setCurrentIndex(0);
    } else {
      baselineSessionHadMissingRef.current = true;
      setWorkflowPhase('baseline');
      const existingKeys = new Set(refs.filter(hasRecordedBaseline).map((ref) => ref.word));
      const nextMissingIndex = words.findIndex((word) => !existingKeys.has(`${set}:${word}`));
      setCurrentIndex(nextMissingIndex === -1 ? 0 : nextMissingIndex);
    }
  }, [currentSet, referencesLoaded]);

  useEffect(() => {
    setLiveSimilarity(0);
    stableMatchFramesRef.current = 0;
    setShowReferenceStatus(false);
    setPracticeResult(null);
  }, [currentSet, currentIndex]);

  useEffect(() => {
    if (navigator.storage?.persist) {
      navigator.storage.persist().then((granted) => {
        pushDebugLog(`Persistent storage ${granted ? 'granted' : 'not granted (browser discretion)'}.`);
      });
    }
  }, []);

  useEffect(() => {
    return () => {
      clearBufferTimers();
      stopInterpreter();
      streamRef.current?.getTracks().forEach((track) => track.stop());
      if (handLandmarkerRef.current) {
        handLandmarkerRef.current.close();
      }
    };
  }, []);

  useEffect(() => {
    const media = navigator.mediaDevices;
    if (!media?.addEventListener) return undefined;
    // Fires on unplug and replug; only re-acquire when the current stream is already dead,
    // so plugging in unrelated devices never interrupts a working session.
    const onDeviceChange = () => {
      const live = streamRef.current?.getVideoTracks().some((track) => track.readyState === 'live');
      if (!live) restartCameraRef.current?.();
    };
    media.addEventListener('devicechange', onDeviceChange);
    return () => media.removeEventListener('devicechange', onDeviceChange);
  }, []);

  useEffect(() => {
    logAnalyticsEvent('page_view', { page_title: appPage });
    setPracticeResult(null);
    isActiveRecordingRef.current = false;
    setIsActiveRecording(false);
    setActiveRecordingProgress(0);
    recordingFramesRef.current = { h1: [], h2: [], m1: [], m2: [] };
    recordingTotalFramesRef.current = 0;
  }, [appPage]);

  useEffect(() => {
    if (!auth) { setAuthLoading(false); return; }
    const unsub = onAuthStateChanged(auth, async (fbUser) => {
      setUser(fbUser);
      userRef.current = fbUser;
      setAuthLoading(false);
      if (fbUser) {
        setSyncStatus('syncing');
        try {
          await downloadAndMerge(fbUser.uid, db);
          setSyncStatus('done');
          setLastSyncedAt(Date.now());
        } catch {
          setSyncStatus('error');
        }
      }
    });
    return () => unsub();
  }, []);

  const startTutorial = () => {
    setView('app');
    setAppPage('learn');
    // The tour narrates the baseline-first flow and highlights the learn page controls.
    if (workflowPhase !== 'baseline') switchToBaseline();
    setTutorialStep(0);
    setTutorialTargetRect(null);
    setShowTutorial(true);
  };

  const closeTutorial = (markSeen = true) => {
    if (markSeen) {
      window.localStorage.setItem(TUTORIAL_STORAGE_KEY, '1');
    }
    setShowTutorial(false);
    setTutorialStep(0);
    setTutorialTargetRect(null);
  };

  return (
    <div className={`${isDarkMode ? 'bg-slate-950 text-slate-100' : 'bg-slate-50 text-slate-900'} min-h-screen font-sans antialiased`}>
      <header className={`sticky top-0 z-50 border-b ${isDarkMode ? 'border-slate-700 bg-slate-900' : 'border-slate-200 bg-white'}`}>
        <div className="w-full flex h-16 items-center justify-between px-6">
          <div
            data-tour="menu"
            className="flex cursor-pointer items-center gap-2"
            onClick={() => setIsSidebarOpen(true)}
          >
            <div className="rounded-lg bg-indigo-600 px-2 py-1.5 text-sm font-bold tracking-tight text-white">ASL</div>
            <span className="text-lg font-bold tracking-tight">SignCards</span>
          </div>
          {!authLoading && (
            user ? (
              <button
                onClick={() => { setAppPage('profile'); setView('app'); }}
                className={`flex items-center gap-2 rounded-lg border px-3 py-1.5 text-sm font-semibold transition-colors ${isDarkMode ? 'border-slate-600 text-slate-300 hover:bg-slate-800' : 'border-slate-200 text-slate-700 hover:bg-slate-50'}`}
              >
                <User className="h-4 w-4 flex-shrink-0" />
                <span className="hidden sm:inline max-w-[120px] truncate">{user.displayName || user.email?.split('@')[0] || 'Account'}</span>
              </button>
            ) : (
              <button
                onClick={() => setShowAuthModal(true)}
                className="flex items-center gap-2 rounded-lg bg-indigo-600 px-3 py-1.5 text-sm font-semibold text-white hover:bg-indigo-700"
              >
                <LogIn className="h-4 w-4" />
                <span>Sign In</span>
              </button>
            )
          )}
        </div>
      </header>

      {view === 'landing' ? (
        <main>
          <section className="w-full px-6 pb-16 pt-20 text-center">
            <h1 className={`mx-auto max-w-3xl text-5xl font-black leading-tight tracking-tight md:text-6xl ${isDarkMode ? 'text-slate-100' : 'text-slate-900'}`}>
              Master American Sign Language with{' '}
              <span className={isDarkMode ? 'text-indigo-400' : 'text-indigo-600'}>Personalized</span>{' '}
              AI Verification.
            </h1>
            <p className={`mx-auto mt-6 max-w-2xl text-xl leading-relaxed ${isDarkMode ? 'text-slate-400' : 'text-slate-600'}`}>
              Record your own baseline signs, build custom decks, and let real-time spatial tracking verify your accuracy frame-by-frame.
            </p>
            <div className="mt-10 flex justify-center">
              <button
                onClick={() => setView('app')}
                className={`group flex items-center space-x-2 rounded-xl bg-indigo-600 px-8 py-4 text-lg font-semibold text-white shadow-lg transition-all hover:bg-indigo-700 hover:shadow-xl ${isDarkMode ? 'shadow-indigo-900/60' : 'shadow-indigo-200'}`}
              >
                <span>Launch Flashcards</span>
                <ArrowRight className="h-5 w-5 transition-transform group-hover:translate-x-1" />
              </button>
            </div>
            {!user && !authLoading ? (
              <p className={`mt-5 text-sm ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>
                <button
                  onClick={() => setShowAuthModal(true)}
                  className="font-semibold underline underline-offset-2 hover:text-indigo-600"
                >
                  Sign in or create an account
                </button>
                {' '}to sync your baselines and progress across devices.
              </p>
            ) : user ? (
              <p className={`mt-5 text-sm ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>
                Signed in as <strong>{user.email}</strong>. Your data syncs automatically.
              </p>
            ) : null}
          </section>

          <section className={`border-y py-16 ${isDarkMode ? 'border-slate-700 bg-slate-900' : 'border-slate-200 bg-white'}`}>
            <div className="mx-auto grid max-w-5xl gap-8 px-4 md:grid-cols-3">
              <div className={`rounded-xl border p-6 ${isDarkMode ? 'border-slate-700 bg-slate-800' : 'border-slate-100 bg-slate-50'}`}>
                <div className={`mb-4 flex h-12 w-12 items-center justify-center rounded-lg ${isDarkMode ? 'bg-indigo-900 text-indigo-400' : 'bg-indigo-100 text-indigo-600'}`}><Camera /></div>
                <h3 className="mb-2 text-xl font-bold">Personal Baseline Matching</h3>
                <p className={isDarkMode ? 'text-slate-400' : 'text-slate-600'}>Compares your live hand spatial angles against your own recorded baselines rather than generic datasets.</p>
              </div>
              <div className={`rounded-xl border p-6 ${isDarkMode ? 'border-slate-700 bg-slate-800' : 'border-slate-100 bg-slate-50'}`}>
                <div className={`mb-4 flex h-12 w-12 items-center justify-center rounded-lg ${isDarkMode ? 'bg-indigo-900 text-indigo-400' : 'bg-indigo-100 text-indigo-600'}`}><Layers /></div>
                <h3 className="mb-2 text-xl font-bold">Custom Vocab Bundles</h3>
                <p className={isDarkMode ? 'text-slate-400' : 'text-slate-600'}>Create target list decks effortlessly or import preset sequences mapped out directly by campus clubs.</p>
              </div>
              <div className={`rounded-xl border p-6 ${isDarkMode ? 'border-slate-700 bg-slate-800' : 'border-slate-100 bg-slate-50'}`}>
                <div className={`mb-4 flex h-12 w-12 items-center justify-center rounded-lg ${isDarkMode ? 'bg-indigo-900 text-indigo-400' : 'bg-indigo-100 text-indigo-600'}`}><Zap /></div>
                <h3 className="mb-2 text-xl font-bold">Optional Cloud Sync</h3>
                <p className={isDarkMode ? 'text-slate-400' : 'text-slate-600'}>No login required. Data lives locally by default. Sign in to back up and sync your baselines across devices with encrypted cloud storage.</p>
              </div>
            </div>
          </section>
        </main>
      ) : appPage === 'learn' ? (
        <main className="w-full h-[calc(100vh-4rem)] flex flex-col">
          {/* Phase header */}
          <div data-tour="phase" className={`relative flex flex-shrink-0 items-center justify-between px-4 py-3 ${workflowPhase === 'baseline' ? 'bg-orange-500' : 'bg-emerald-600'}`}>
            <div className="flex items-center gap-2 text-white">
              <div className="h-2 w-2 rounded-full bg-white/80" />
              <span className="text-sm font-bold tracking-wide uppercase">
                {workflowPhase === 'baseline' ? 'Baseline Setup' : 'Practice Mode'}
              </span>
            </div>
            <div className="flex items-center gap-4">
              <div className="hidden items-center gap-2 text-white sm:flex">
                <div className="h-1.5 w-24 rounded-full bg-white/30">
                  <div
                    className="h-full rounded-full bg-white transition-all duration-300"
                    style={{ width: `${(baselineCount / activeWords.length) * 100}%` }}
                  />
                </div>
                <span className="text-xs font-semibold opacity-80">{baselineCount}/{activeWords.length}</span>
              </div>
              <span className="text-sm font-semibold text-white opacity-90">Card {currentIndex + 1} / {modeWords.length}</span>
            </div>
            {workflowPhase === 'practice' ? (
              <button
                onClick={() => setPracticeOrder((prev) => prev === 'ordered' ? 'random' : 'ordered')}
                className="flex items-center gap-1 rounded-lg px-2 py-1 text-xs font-semibold text-white/80 transition-colors hover:bg-white/20"
                title={practiceOrder === 'ordered' ? 'Switch to randomized' : 'Switch to in-order'}
              >
                <Shuffle className="h-3.5 w-3.5" />
                <span className="hidden sm:inline">{practiceOrder === 'ordered' ? 'In Order' : 'Random'}</span>
              </button>
            ) : null}
            <button
              data-tour="more"
              onClick={() => setIsOverflowMenuOpen((prev) => !prev)}
              className="rounded-lg p-2 text-white transition-colors hover:bg-white/20"
              title="More options"
            >
              <MoreHorizontal className="h-5 w-5" />
            </button>

            {isOverflowMenuOpen ? (
              <>
                <div className="fixed inset-0 z-[60]" onClick={() => setIsOverflowMenuOpen(false)} />
                <div className={`absolute right-2 top-full z-[65] mt-1 w-56 overflow-hidden rounded-xl border shadow-xl ${isDarkMode ? 'border-slate-700 bg-slate-800' : 'border-slate-200 bg-white'}`}>
                  {workflowPhase === 'baseline' ? (
                    <button
                      onClick={() => { switchToPracticeIfReady(); setIsOverflowMenuOpen(false); }}
                      disabled={wordsWithBaseline.length === 0}
                      className={`flex w-full items-center gap-2 px-4 py-3 text-left text-sm font-semibold disabled:opacity-40 ${isDarkMode ? 'text-slate-200 hover:bg-slate-700' : 'text-slate-700 hover:bg-slate-50'}`}
                    >
                      <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                      Go to Practice
                    </button>
                  ) : (
                    <>
                      <button
                        onClick={() => { switchToBaseline(); setIsOverflowMenuOpen(false); }}
                        className={`flex w-full items-center gap-2 px-4 py-3 text-left text-sm font-semibold ${isDarkMode ? 'text-slate-200 hover:bg-slate-700' : 'text-slate-700 hover:bg-slate-50'}`}
                      >
                        <RotateCcw className="h-4 w-4 text-orange-500" />
                        Re-record Baselines
                      </button>
                      <button
                        onClick={() => { handleValidation(false); setIsOverflowMenuOpen(false); }}
                        className={`flex w-full items-center gap-2 px-4 py-3 text-left text-sm font-semibold ${isDarkMode ? 'text-slate-200 hover:bg-slate-700' : 'text-slate-700 hover:bg-slate-50'}`}
                      >
                        <XCircle className="h-4 w-4 text-rose-500" />
                        Mark Fail
                      </button>
                    </>
                  )}
                  <div className={`border-t ${isDarkMode ? 'border-slate-700' : 'border-slate-100'}`} />
                  {activeSetOption?.isCustom ? (
                    <button
                      onClick={() => { handleEditSet(activeSetOption); setIsOverflowMenuOpen(false); }}
                      className={`flex w-full items-center gap-2 px-4 py-3 text-left text-sm font-semibold ${isDarkMode ? 'text-slate-200 hover:bg-slate-700' : 'text-slate-700 hover:bg-slate-50'}`}
                    >
                      <Pencil className="h-4 w-4 text-indigo-500" />
                      Edit Set
                    </button>
                  ) : null}
                  <button
                    onClick={() => { setIsMirrored((prev) => !prev); setIsOverflowMenuOpen(false); }}
                    className={`flex w-full items-center gap-2 px-4 py-3 text-left text-sm font-semibold ${isDarkMode ? 'text-slate-200 hover:bg-slate-700' : 'text-slate-700 hover:bg-slate-50'}`}
                  >
                    <Camera className="h-4 w-4" />
                    {isMirrored ? 'Disable Mirroring' : 'Enable Mirroring'}
                  </button>
                  <button
                    onClick={() => { setCurrentIndex(0); setIsOverflowMenuOpen(false); }}
                    className={`flex w-full items-center gap-2 px-4 py-3 text-left text-sm font-semibold ${isDarkMode ? 'text-slate-200 hover:bg-slate-700' : 'text-slate-700 hover:bg-slate-50'}`}
                  >
                    <RotateCcw className="h-4 w-4" />
                    Reset to Card 1
                  </button>
                </div>
              </>
            ) : null}
          </div>

          {workflowPhase === 'practice' && missingBaselineCount > 0 ? (
            <div className={`flex flex-shrink-0 items-center justify-between gap-3 border-b px-4 py-2 text-xs sm:text-sm ${isDarkMode ? 'border-amber-900/70 bg-amber-950/40 text-amber-100' : 'border-amber-200 bg-amber-50 text-amber-900'}`}>
              <p><strong>Only cards with recorded baselines appear in practice.</strong> {missingBaselineCount} card{missingBaselineCount !== 1 ? 's are' : ' is'} missing.</p>
              <button
                onClick={switchToBaseline}
                className={`flex-shrink-0 rounded-lg px-3 py-1.5 text-xs font-bold ${isDarkMode ? 'bg-amber-300 text-amber-950 hover:bg-amber-200' : 'bg-amber-500 text-white hover:bg-amber-600'}`}
              >
                Record Missing
              </button>
            </div>
          ) : null}

          {/* Camera */}
          <div data-tour="camera" className={`relative min-h-0 flex-1 overflow-hidden flex items-center justify-center ${isDarkMode ? 'bg-slate-900' : 'bg-white'}`}>
            <div
              className="relative max-w-full max-h-full"
              style={videoAspect ? { aspectRatio: videoAspect } : { width: '100%', height: '100%' }}
            >
            <Webcam
              key={cameraKey}
              audio={false}
              ref={webcamRef}
              screenshotFormat="image/jpeg"
              onUserMedia={handleUserMedia}
              onUserMediaError={handleUserMediaError}
              className={`h-full w-full object-cover transition-transform duration-200 ${isMirrored ? 'scale-x-[-1]' : ''}`}
            />

            {/* Word overlay */}
            <div className="absolute left-3 top-3 rounded-xl bg-black/65 px-3 py-2 text-white backdrop-blur-md">
              <p className="text-[10px] font-semibold uppercase tracking-widest opacity-70">Sign this</p>
              <p className="mt-0.5 text-4xl font-black leading-none tracking-tight">{currentWord}</p>
              <div className="mt-1.5 flex items-center gap-1.5">
                <div className={`h-1.5 w-1.5 flex-shrink-0 rounded-full ${hasRecordedBaseline(activeReference) ? 'bg-emerald-400' : 'bg-amber-400'}`} />
                <span className="text-[10px] font-semibold opacity-70">{hasRecordedBaseline(activeReference) ? 'Baseline recorded' : 'Record baseline'}</span>
              </div>
            </div>

            {/* Recording progress */}
            {isActiveRecording ? (
              <div className="absolute bottom-0 left-0 right-0 p-3">
                <div className="flex items-center gap-3 rounded-xl bg-black/75 px-4 py-2.5 backdrop-blur-sm">
                  <span className="h-2 w-2 flex-shrink-0 animate-ping rounded-full bg-rose-500" />
                  <p className="text-xs font-bold uppercase tracking-widest text-rose-400">Recording</p>
                  <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-white/20">
                    <div
                      className="h-full rounded-full bg-rose-500 transition-none"
                      style={{ width: `${activeRecordingProgress * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            ) : null}

            {/* Practice result */}
            {practiceResult && !isActiveRecording ? (
              <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                <div className="rounded-2xl bg-white px-8 py-6 text-center shadow-2xl">
                  <div className={`mx-auto flex h-14 w-14 items-center justify-center rounded-full ${practiceResult.passed ? 'bg-emerald-100' : 'bg-amber-100'}`}>
                    {practiceResult.passed
                      ? <CheckCircle2 className="h-8 w-8 text-emerald-600" />
                      : <XCircle className="h-8 w-8 text-amber-500" />}
                  </div>
                  <p className={`mt-3 text-xl font-black ${practiceResult.passed ? 'text-emerald-700' : 'text-slate-800'}`}>
                    {practiceResult.passed ? 'Nice work!' : 'Not quite...'}
                  </p>
                  <p className="mt-1 text-4xl font-black text-slate-900">{(practiceResult.similarity * 100).toFixed(0)}%</p>
                  {practiceResult.notes?.length ? (
                    <ul className="mx-auto mt-3 max-w-xs space-y-1 text-left">
                      {practiceResult.notes.map((note) => (
                        <li key={note} className="flex items-start gap-1.5 text-xs text-slate-600">
                          <span className="mt-1 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-amber-400" />
                          <span>{note}</span>
                        </li>
                      ))}
                    </ul>
                  ) : null}
                  <div className="mt-5 flex gap-3">
                    <button
                      onClick={() => setPracticeResult(null)}
                      className="rounded-xl border border-slate-200 px-5 py-2.5 text-sm font-semibold text-slate-700 hover:bg-slate-50"
                    >
                      Try Again
                    </button>
                    <button
                      onClick={() => {
                        handleValidation(practiceResult.passed, practiceResult.similarity, true);
                        setPracticeResult(null);
                      }}
                      className={`rounded-xl px-5 py-2.5 text-sm font-bold text-white ${practiceResult.passed ? 'bg-emerald-600 hover:bg-emerald-700' : 'bg-slate-700 hover:bg-slate-800'}`}
                    >
                      Move On →
                    </button>
                  </div>
                </div>
              </div>
            ) : null}

            {/* Buffering countdown */}
            {isBuffering ? (
              <div className="absolute inset-0 flex items-center justify-center bg-black/35">
                <div className="rounded-2xl bg-white px-6 py-4 text-center shadow-xl">
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Get Ready</p>
                  <p className="text-2xl font-black text-slate-900">{bufferSecondsLeft.toFixed(1)}s</p>
                  <p className="mt-1 text-xs text-slate-600">
                    {pendingAction === 'record' ? 'Capturing baseline soon...' : 'Checking answer soon...'}
                  </p>
                </div>
              </div>
            ) : null}

            {/* Hand joints */}
            {showHandNodes && liveLandmarks?.length ? (
              <div className="pointer-events-none absolute inset-0">
                {liveLandmarks.map((landmark, index) => {
                  const x = isMirrored ? 1 - landmark.x : landmark.x;
                  const y = landmark.y;
                  return (
                    <div
                      key={`joint-${index}`}
                      className="absolute h-2.5 w-2.5 rounded-full border border-white/70 bg-emerald-400/90"
                      style={{
                        left: `${x * 100}%`,
                        top: `${y * 100}%`,
                        transform: 'translate(-50%, -50%)'
                      }}
                    />
                  );
                })}
              </div>
            ) : null}

            {/* Camera initializing overlay */}
            {engineLoading ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-900/80">
                <LoaderCircle className="h-8 w-8 animate-spin text-white/70" />
                <p className="mt-3 text-sm font-semibold text-white/70">Initializing camera…</p>
              </div>
            ) : null}

            {cameraLost ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-900/85 p-4 text-center">
                <CameraOff className="h-8 w-8 text-white/70" />
                <p className="mt-3 text-sm font-semibold text-white">Camera disconnected</p>
                <p className="mt-1 max-w-xs text-xs text-white/60">
                  Reconnect your camera. If it is already plugged back in, tap below to reconnect.
                </p>
                <button
                  onClick={restartCamera}
                  className="mt-4 rounded-xl bg-indigo-600 px-5 py-2.5 text-sm font-bold text-white hover:bg-indigo-700"
                >
                  Reconnect Camera
                </button>
              </div>
            ) : null}

            {/* Engine error */}
            {engineError ? (
              <div className="absolute bottom-3 left-3 right-3 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
                Camera error: {engineError}
              </div>
            ) : null}
            </div>
          </div>

          {/* Action bar */}
          <div data-tour="action" className={`flex flex-shrink-0 items-center gap-3 border-t px-4 py-3 ${isDarkMode ? 'border-slate-700 bg-slate-900' : 'border-slate-200 bg-white'}`}>
            <button
              onClick={() => { setCurrentIndex((prev) => Math.max(0, prev - 1)); setPracticeResult(null); }}
              disabled={currentIndex === 0 || isActiveRecording || !!practiceResult}
              className={`rounded-lg px-3 py-2.5 text-sm font-semibold transition-colors disabled:opacity-30 ${isDarkMode ? 'text-slate-300 hover:bg-slate-800' : 'text-slate-700 hover:bg-slate-100'}`}
            >
              ← Prev
            </button>

            {workflowPhase === 'baseline' ? (
              <button
                onClick={() => startBufferedAction('record')}
                disabled={isBuffering || isRecordingReference || engineLoading || isActiveRecording}
                className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-orange-500 py-3 text-sm font-bold text-white transition-colors hover:bg-orange-600 disabled:cursor-not-allowed"
              >
                {engineLoading
                  ? <><LoaderCircle className="h-5 w-5 animate-spin" /><span>Initializing…</span></>
                  : <><Radio className={`h-5 w-5 ${isRecordingReference ? 'animate-ping' : ''}`} /><span>{hasRecordedBaseline(activeReference) ? 'Re-record Baseline' : 'Record Baseline'}</span></>
                }
              </button>
            ) : (
              <button
                onClick={() => startBufferedAction('check')}
                disabled={isBuffering || !hasRecordedBaseline(activeReference) || engineLoading || !!practiceResult || isActiveRecording}
                className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-emerald-600 py-3 text-sm font-bold text-white transition-colors hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {engineLoading
                  ? <><LoaderCircle className="h-5 w-5 animate-spin" /><span>Initializing…</span></>
                  : <><CheckCircle2 className="h-5 w-5" /><span>Check My Sign</span></>
                }
              </button>
            )}

            <button
              onClick={() => {
                if (workflowPhase === 'practice') {
                  setCurrentIndex((prev) => (prev + 1) % modeWords.length);
                } else {
                  setCurrentIndex((prev) => Math.min(activeWords.length - 1, prev + 1));
                }
                setPracticeResult(null);
              }}
              disabled={(workflowPhase === 'baseline' && currentIndex === activeWords.length - 1) || isActiveRecording || !!practiceResult}
              className={`rounded-lg px-3 py-2.5 text-sm font-semibold transition-colors disabled:opacity-30 ${isDarkMode ? 'text-slate-300 hover:bg-slate-800' : 'text-slate-700 hover:bg-slate-100'}`}
            >
              Next →
            </button>
          </div>
        </main>
      ) : appPage === 'create' ? (
        <main className="w-full h-[calc(100vh-4rem)] overflow-y-auto p-3 lg:p-4">
          <section className={`relative mx-auto max-w-4xl rounded-2xl border p-6 shadow-sm ${isDarkMode ? 'border-slate-700 bg-slate-900 text-slate-100' : 'border-slate-200 bg-white text-slate-900'}`}>
            <div className="mb-6 flex items-center gap-3">
              <button
                onClick={() => { resetCreateForm(); setAppPage('learn'); }}
                className={`rounded-lg border px-3 py-2 text-sm font-semibold ${isDarkMode ? 'border-slate-600 text-slate-300 hover:bg-slate-800' : 'border-slate-200 text-slate-700 hover:bg-slate-50'}`}
              >
                Back
              </button>
              <h2 className={`text-3xl font-black tracking-tight ${isDarkMode ? 'text-slate-100' : 'text-slate-900'}`}>
                {editingSetId ? 'Edit Set' : 'Create New Set'}
              </h2>
            </div>

            <label className="block">
              <span className={`mb-1 block text-sm font-semibold ${isDarkMode ? 'text-slate-300' : 'text-slate-700'}`}>Set Title</span>
              <input
                value={newSetTitle}
                onChange={(e) => setNewSetTitle(e.target.value)}
                placeholder="e.g. Week 1 Vocabulary"
                className={`w-full rounded-lg border px-3 py-2 outline-none focus:ring-2 ${isDarkMode ? 'border-slate-600 bg-slate-800 text-slate-100 placeholder:text-slate-500 focus:ring-indigo-500' : 'border-slate-300 bg-white text-slate-900 focus:ring-indigo-200'}`}
              />
            </label>

            <div className="mt-6">
              <span className={`mb-2 block text-sm font-semibold ${isDarkMode ? 'text-slate-300' : 'text-slate-700'}`}>Cards</span>
              <div className="space-y-2">
                {newSetCards.map((card, idx) => {
                  const addNewCard = () => {
                    const newCard = { id: `${Date.now()}-${Math.random().toString(36).slice(2)}`, word: '', isMultiSign: false, components: '' };
                    setNewSetCards((prev) => [...prev, newCard]);
                    setTimeout(() => {
                      const inputs = document.querySelectorAll('[data-card-word]');
                      const last = inputs[inputs.length - 1];
                      if (last) last.focus();
                    }, 0);
                  };
                  return (
                    <div key={card.id} className={`rounded-xl border p-3 ${isDarkMode ? 'border-slate-600 bg-slate-800' : 'border-slate-200 bg-slate-50'}`}>
                      <div className="flex items-center gap-2">
                        <input
                          data-card-word
                          value={card.word}
                          onChange={(e) => setNewSetCards((prev) => prev.map((c) => c.id === card.id ? { ...c, word: e.target.value } : c))}
                          onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); addNewCard(); } }}
                          placeholder={`Card ${idx + 1}`}
                          className={`flex-1 rounded-lg border px-3 py-1.5 text-sm outline-none focus:ring-2 ${isDarkMode ? 'border-slate-600 bg-slate-700 text-slate-100 placeholder:text-slate-500 focus:ring-indigo-500' : 'border-slate-300 bg-white text-slate-900 focus:ring-indigo-200'}`}
                        />

                        <label className={`flex shrink-0 cursor-pointer select-none items-center gap-1.5 text-xs font-semibold ${isDarkMode ? 'text-slate-300' : 'text-slate-600'}`}>
                          <input
                            type="checkbox"
                            checked={card.isMultiSign}
                            onChange={(e) => setNewSetCards((prev) => prev.map((c) => c.id === card.id ? { ...c, isMultiSign: e.target.checked } : c))}
                            className="h-3.5 w-3.5 rounded accent-indigo-600"
                          />
                          Multi-sign
                        </label>

                        <button
                          type="button"
                          tabIndex={-1}
                          onClick={(e) => {
                            if (tooltipOpenId === card.id) {
                              setTooltipOpenId(null);
                            } else {
                              const rect = e.currentTarget.getBoundingClientRect();
                              setTooltipPosition({
                                top: rect.bottom + 8,
                                left: Math.max(8, Math.min(rect.right - 288, window.innerWidth - 296)),
                              });
                              setTooltipOpenId(card.id);
                            }
                          }}
                          className={`shrink-0 rounded-full p-0.5 transition-colors ${isDarkMode ? 'text-slate-500 hover:text-slate-200' : 'text-slate-400 hover:text-slate-700'} ${tooltipOpenId === card.id ? (isDarkMode ? 'text-slate-200' : 'text-slate-700') : ''}`}
                        >
                          <CircleHelp className="h-4 w-4" />
                        </button>

                        <button
                          type="button"
                          onClick={() => setNewSetCards((prev) => {
                            const filtered = prev.filter((c) => c.id !== card.id);
                            return filtered.length ? filtered : [{ id: `${Date.now()}`, word: '', isMultiSign: false, components: '' }];
                          })}
                          className={`shrink-0 rounded-md p-1 text-rose-500 hover:bg-rose-100 ${isDarkMode ? 'hover:bg-rose-900/30' : 'hover:bg-rose-100'}`}
                          title="Remove card"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>

                      {card.isMultiSign && (
                        <div className="mt-2">
                          <input
                            value={card.components}
                            onChange={(e) => setNewSetCards((prev) => prev.map((c) => c.id === card.id ? { ...c, components: e.target.value } : c))}
                            onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); addNewCard(); } }}
                            placeholder="Component signs, comma-separated (e.g. TEACH, PERSON)"
                            className={`w-full rounded-lg border px-3 py-1.5 text-xs outline-none focus:ring-2 ${isDarkMode ? 'border-slate-600 bg-slate-700 text-slate-100 placeholder:text-slate-500 focus:ring-indigo-500' : 'border-slate-300 bg-white text-slate-600 focus:ring-indigo-200'}`}
                          />
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              <div className="mt-3 flex items-center gap-3">
                <button
                  type="button"
                  onClick={() => {
                    const newCard = { id: `${Date.now()}-${Math.random().toString(36).slice(2)}`, word: '', isMultiSign: false, components: '' };
                    setNewSetCards((prev) => [...prev, newCard]);
                    setTimeout(() => {
                      const inputs = document.querySelectorAll('[data-card-word]');
                      const last = inputs[inputs.length - 1];
                      if (last) last.focus();
                    }, 0);
                  }}
                  className={`flex items-center gap-1 rounded-lg border px-3 py-1.5 text-sm font-semibold ${isDarkMode ? 'border-slate-600 bg-slate-800 text-slate-100 hover:bg-slate-700' : 'border-slate-300 bg-white text-slate-700 hover:bg-slate-50'}`}
                >
                  <Plus className="h-4 w-4" />
                  Add Card
                </button>
                <span className={`text-xs ${isDarkMode ? 'text-slate-500' : 'text-slate-400'}`}>
                  or press Enter in any card field
                </span>
              </div>
            </div>

            <div className="mt-6 flex flex-wrap gap-2">
              <button
                onClick={editingSetId ? handleUpdateSet : handleCreateSet}
                className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-bold text-white hover:bg-indigo-700"
              >
                {editingSetId ? 'Save Changes' : 'Save Set'}
              </button>
              <button
                onClick={() => { resetCreateForm(); setAppPage('learn'); }}
                className={`rounded-lg border px-4 py-2 text-sm font-semibold ${isDarkMode ? 'border-slate-600 text-slate-300 hover:bg-slate-800' : 'border-slate-300 text-slate-700 hover:bg-slate-50'}`}
              >
                Cancel
              </button>
            </div>
          </section>
        </main>
      ) : appPage === 'settings' ? (
        <main className="w-full h-[calc(100vh-4rem)] p-3 lg:p-4">
          <section className={`mx-auto h-full max-w-4xl overflow-y-auto rounded-2xl border p-6 shadow-sm ${isDarkMode ? 'border-slate-700 bg-slate-900 text-slate-100' : 'border-slate-200 bg-white text-slate-900'}`}>
            <div className="mb-4 flex items-center gap-3">
              <button
                onClick={() => setAppPage('learn')}
                className={`rounded-lg border px-3 py-2 text-sm font-semibold ${isDarkMode ? 'border-slate-600 text-slate-300 hover:bg-slate-800' : 'border-slate-200 text-slate-700 hover:bg-slate-50'}`}
              >
                Back
              </button>
              <h2 className={`text-3xl font-black tracking-tight ${isDarkMode ? 'text-slate-100' : 'text-slate-900'}`}>Settings</h2>
            </div>
            <div className="mt-6 space-y-3">
              <button
                onClick={() => setIsDarkMode((prev) => !prev)}
                role="switch"
                aria-checked={isDarkMode}
                className={`flex w-full items-center justify-between rounded-xl border px-4 py-3 text-left ${isDarkMode ? 'border-slate-600 bg-slate-800 text-slate-100' : 'border-slate-200 bg-slate-50 text-slate-800'}`}
              >
                <span className="flex items-center gap-2 text-sm font-semibold">
                  <Moon className="h-4 w-4" />
                  <span>Dark Mode</span>
                </span>
                <span className={`relative inline-flex h-7 w-12 items-center rounded-full transition-colors ${isDarkMode ? 'bg-indigo-600' : 'bg-slate-300'}`}>
                  <span className={`inline-block h-5 w-5 transform rounded-full bg-white transition-transform ${isDarkMode ? 'translate-x-6' : 'translate-x-1'}`} />
                </span>
              </button>

              <button
                onClick={() => setIsMirrored((prev) => !prev)}
                role="switch"
                aria-checked={isMirrored}
                className={`flex w-full items-center justify-between rounded-xl border px-4 py-3 text-left ${isDarkMode ? 'border-slate-600 bg-slate-800 text-slate-100' : 'border-slate-200 bg-slate-50 text-slate-800'}`}
              >
                <span className="flex items-center gap-2 text-sm font-semibold">
                  <Camera className="h-4 w-4" />
                  <span>Camera Mirroring</span>
                </span>
                <span className={`relative inline-flex h-7 w-12 items-center rounded-full transition-colors ${isMirrored ? 'bg-indigo-600' : 'bg-slate-300'}`}>
                  <span className={`inline-block h-5 w-5 transform rounded-full bg-white transition-transform ${isMirrored ? 'translate-x-6' : 'translate-x-1'}`} />
                </span>
              </button>

              <button
                onClick={() => setShowHandNodes((prev) => !prev)}
                role="switch"
                aria-checked={showHandNodes}
                className={`flex w-full items-center justify-between rounded-xl border px-4 py-3 text-left ${isDarkMode ? 'border-slate-600 bg-slate-800 text-slate-100' : 'border-slate-200 bg-slate-50 text-slate-800'}`}
              >
                <span className="flex items-center gap-2 text-sm font-semibold">
                  <Users className="h-4 w-4" />
                  <span>Show Hand Joints</span>
                </span>
                <span className={`relative inline-flex h-7 w-12 items-center rounded-full transition-colors ${showHandNodes ? 'bg-indigo-600' : 'bg-slate-300'}`}>
                  <span className={`inline-block h-5 w-5 transform rounded-full bg-white transition-transform ${showHandNodes ? 'translate-x-6' : 'translate-x-1'}`} />
                </span>
              </button>

              <button
                onClick={() => setShowDebugLog((prev) => !prev)}
                role="switch"
                aria-checked={showDebugLog}
                className={`flex w-full items-center justify-between rounded-xl border px-4 py-3 text-left ${isDarkMode ? 'border-slate-600 bg-slate-800 text-slate-100' : 'border-slate-200 bg-slate-50 text-slate-800'}`}
              >
                <span className="flex items-center gap-2 text-sm font-semibold">
                  <CircleHelp className="h-4 w-4" />
                  <span>Show Debug Log</span>
                </span>
                <span className={`relative inline-flex h-7 w-12 items-center rounded-full transition-colors ${showDebugLog ? 'bg-indigo-600' : 'bg-slate-300'}`}>
                  <span className={`inline-block h-5 w-5 transform rounded-full bg-white transition-transform ${showDebugLog ? 'translate-x-6' : 'translate-x-1'}`} />
                </span>
              </button>

              <div className={`rounded-xl border ${isDarkMode ? 'border-slate-600 bg-slate-800' : 'border-slate-200 bg-slate-50'}`}>
                <button
                  onClick={() => setShowMatchFeatures((prev) => !prev)}
                  aria-expanded={showMatchFeatures}
                  className={`flex w-full items-center justify-between px-4 py-3 text-left ${isDarkMode ? 'text-slate-100' : 'text-slate-800'}`}
                >
                  <span className="flex items-center gap-2 text-sm font-semibold">
                    <Zap className="h-4 w-4" />
                    <span>Sign Matching Features</span>
                    <span className={`rounded-full px-2 py-0.5 text-[10px] font-bold ${enabledMatchCount === MATCH_FEATURE_LIST.length ? (isDarkMode ? 'bg-slate-700 text-slate-300' : 'bg-slate-200 text-slate-600') : 'bg-amber-500 text-white'}`}>
                      {enabledMatchCount}/{MATCH_FEATURE_LIST.length}
                    </span>
                  </span>
                  {showMatchFeatures ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                </button>

                {showMatchFeatures ? (
                  <div className={`border-t px-4 pb-4 pt-3 ${isDarkMode ? 'border-slate-700' : 'border-slate-200'}`}>
                    <p className={`mb-3 text-xs ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>
                      Turn individual checks off to isolate problems. Remaining checks are re-weighted automatically, so scores stay on the same scale.
                    </p>
                    <div className="space-y-2">
                      {MATCH_FEATURE_LIST.map((feature) => {
                        const enabled = matchFeatures[feature.key] !== false;
                        return (
                          <button
                            key={feature.key}
                            onClick={() => setMatchFeatures((prev) => ({ ...prev, [feature.key]: !enabled }))}
                            role="switch"
                            aria-checked={enabled}
                            className={`flex w-full items-center justify-between gap-3 rounded-lg border px-3 py-2.5 text-left ${isDarkMode ? 'border-slate-700 bg-slate-900' : 'border-slate-200 bg-white'}`}
                          >
                            <span className="min-w-0">
                              <span className={`block text-sm font-semibold ${isDarkMode ? 'text-slate-100' : 'text-slate-800'}`}>{feature.label}</span>
                              <span className={`block text-xs ${isDarkMode ? 'text-slate-500' : 'text-slate-500'}`}>{feature.description}</span>
                            </span>
                            <span className={`relative inline-flex h-6 w-11 flex-shrink-0 items-center rounded-full transition-colors ${enabled ? 'bg-indigo-600' : 'bg-slate-300'}`}>
                              <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${enabled ? 'translate-x-6' : 'translate-x-1'}`} />
                            </span>
                          </button>
                        );
                      })}
                    </div>
                    <button
                      onClick={() => setMatchFeatures(DEFAULT_MATCH_FEATURES)}
                      disabled={enabledMatchCount === MATCH_FEATURE_LIST.length}
                      className={`mt-3 rounded-lg border px-3 py-1.5 text-xs font-semibold disabled:opacity-40 ${isDarkMode ? 'border-slate-600 text-slate-300 hover:bg-slate-700' : 'border-slate-300 text-slate-700 hover:bg-slate-100'}`}
                    >
                      Enable all
                    </button>
                  </div>
                ) : null}
              </div>
            </div>

            <div className={`mt-8 border-t pt-6 ${isDarkMode ? 'border-slate-700' : 'border-slate-200'}`}>
              <p className={`mb-1 text-sm font-semibold ${isDarkMode ? 'text-slate-300' : 'text-slate-700'}`}>Data Backup</p>
              <p className={`mb-3 text-xs ${isDarkMode ? 'text-slate-500' : 'text-slate-400'}`}>
                {lastBackupAt
                  ? `Last exported: ${new Date(lastBackupAt).toLocaleString()}`
                  : 'No backup yet. Export to protect your sets and baselines from accidental browser storage clearing.'}
              </p>
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={handleExportData}
                  className="flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-700"
                >
                  <Download className="h-4 w-4" />
                  Export My Data
                </button>
                <button
                  onClick={() => importFileInputRef.current?.click()}
                  className={`flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-semibold ${isDarkMode ? 'border-slate-600 text-slate-300 hover:bg-slate-800' : 'border-slate-300 text-slate-700 hover:bg-slate-50'}`}
                >
                  <Upload className="h-4 w-4" />
                  Import Data
                </button>
              </div>
            </div>
          </section>
        </main>
      ) : appPage === 'stats' ? (() => {
        const statsKey = statsSetKey ?? currentSet;
        const statsSetOption = availableSets.find((s) => s.key === statsKey) ?? availableSets[0];
        const statsWords = statsSetOption?.words ?? [];

        const setHistory = history.filter((h) => statsWords.includes(h.word));
        const totalAttempts = setHistory.length;
        const totalPasses = setHistory.filter((h) => h.status === 'correct').length;
        const overallRate = totalAttempts > 0 ? totalPasses / totalAttempts : null;

        const wordStats = statsWords.map((word) => {
          const entries = history.filter((h) => h.word === word);
          const attempts = entries.length;
          const passes = entries.filter((h) => h.status === 'correct').length;
          const withSim = entries.filter((h) => h.similarity !== undefined);
          const avgSim = withSim.length > 0
            ? withSim.reduce((a, h) => a + h.similarity, 0) / withSim.length
            : null;
          const passRate = attempts > 0 ? passes / attempts : null;
          return { word, attempts, passes, passRate, avgSim, score: avgSim ?? passRate ?? -1 };
        }).sort((a, b) => statsSortDir === 'weakest' ? a.score - b.score : b.score - a.score);

        const DAY_MS = 86_400_000;
        const midnightToday = new Date();
        midnightToday.setHours(0, 0, 0, 0);
        const todayStart = midnightToday.getTime();
        const dailyBuckets = Array.from({ length: statsRangeDays }, (_, i) => {
          const start = todayStart - (statsRangeDays - 1 - i) * DAY_MS;
          const end = start + DAY_MS;
          const dayEntries = setHistory.filter((h) => h.timestamp >= start && h.timestamp < end);
          const passes = dayEntries.filter((h) => h.status === 'correct').length;
          return {
            label: new Date(start).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }),
            attempts: dayEntries.length,
            passes,
            rate: dayEntries.length > 0 ? passes / dayEntries.length : null,
          };
        });
        const labelIndices = new Set([0, Math.floor((statsRangeDays - 1) / 2), statsRangeDays - 1]);
        const rangeAttempts = dailyBuckets.reduce((sum, day) => sum + day.attempts, 0);

        return (
          <main className="w-full h-[calc(100vh-4rem)] p-3 lg:p-4">
            <section className={`mx-auto h-full max-w-4xl overflow-y-auto rounded-2xl border p-6 shadow-sm ${isDarkMode ? 'border-slate-700 bg-slate-900 text-slate-100' : 'border-slate-200 bg-white text-slate-900'}`}>
              <div className="mb-5 flex items-center gap-3">
                <button
                  onClick={() => setAppPage('learn')}
                  className={`rounded-lg border px-3 py-2 text-sm font-semibold ${isDarkMode ? 'border-slate-600 text-slate-300 hover:bg-slate-800' : 'border-slate-200 text-slate-700 hover:bg-slate-50'}`}
                >
                  Back
                </button>
                <h2 className={`text-3xl font-black tracking-tight ${isDarkMode ? 'text-slate-100' : 'text-slate-900'}`}>Stats</h2>
              </div>

              {/* Set selector */}
              <div className="mb-6">
                <label className={`mb-1.5 block text-xs font-semibold uppercase tracking-wide ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>Set</label>
                <select
                  value={statsKey}
                  onChange={(e) => setStatsSetKey(e.target.value)}
                  className={`rounded-lg border px-3 py-2 text-sm font-semibold outline-none focus:ring-2 focus:ring-indigo-400 ${isDarkMode ? 'border-slate-600 bg-slate-800 text-slate-100' : 'border-slate-300 bg-white text-slate-900'}`}
                >
                  {availableSets.map((s) => (
                    <option key={s.key} value={s.key}>{s.label} ({s.words.length} cards)</option>
                  ))}
                </select>
              </div>

              {/* Summary */}
              <div className="mb-6 grid grid-cols-3 gap-3">
                <div className={`rounded-xl border p-4 ${isDarkMode ? 'border-slate-700 bg-slate-800' : 'border-slate-200 bg-slate-50'}`}>
                  <p className={`text-xs font-semibold uppercase tracking-wide ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>Total Checks</p>
                  <p className={`mt-1 text-3xl font-black ${isDarkMode ? 'text-slate-100' : 'text-slate-900'}`}>{totalAttempts}</p>
                </div>
                <div className={`rounded-xl border p-4 ${isDarkMode ? 'border-slate-700 bg-slate-800' : 'border-slate-200 bg-slate-50'}`}>
                  <p className={`text-xs font-semibold uppercase tracking-wide ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>Pass Rate</p>
                  <p className={`mt-1 text-3xl font-black ${overallRate !== null ? (overallRate >= 0.65 ? 'text-emerald-500' : 'text-amber-500') : isDarkMode ? 'text-slate-600' : 'text-slate-300'}`}>
                    {overallRate !== null ? `${(overallRate * 100).toFixed(0)}%` : '—'}
                  </p>
                </div>
                <div className={`rounded-xl border p-4 ${isDarkMode ? 'border-slate-700 bg-slate-800' : 'border-slate-200 bg-slate-50'}`}>
                  <p className={`text-xs font-semibold uppercase tracking-wide ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>Active Cards</p>
                  <p className={`mt-1 text-3xl font-black ${isDarkMode ? 'text-slate-100' : 'text-slate-900'}`}>
                    {wordStats.filter((w) => w.attempts > 0).length}
                    <span className={`text-base font-semibold ${isDarkMode ? 'text-slate-500' : 'text-slate-400'}`}>/{statsWords.length}</span>
                  </p>
                </div>
              </div>

              {/* Activity chart */}
              <div className="mb-6">
                <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                  <p className={`text-sm font-semibold ${isDarkMode ? 'text-slate-300' : 'text-slate-700'}`}>
                    Activity
                    <span className={`ml-2 text-xs font-normal ${isDarkMode ? 'text-slate-500' : 'text-slate-400'}`}>
                      {rangeAttempts} check{rangeAttempts !== 1 ? 's' : ''} in the last {statsRangeDays} days
                    </span>
                  </p>
                  <div className={`flex overflow-hidden rounded-lg border ${isDarkMode ? 'border-slate-600' : 'border-slate-300'}`}>
                    {STATS_RANGE_OPTIONS.map((days) => (
                      <button
                        key={days}
                        onClick={() => setStatsRangeDays(days)}
                        className={`px-2.5 py-1 text-xs font-bold transition-colors ${
                          statsRangeDays === days
                            ? 'bg-indigo-600 text-white'
                            : isDarkMode ? 'text-slate-300 hover:bg-slate-800' : 'text-slate-600 hover:bg-slate-50'
                        }`}
                      >
                        {days}d
                      </button>
                    ))}
                  </div>
                </div>
                <div className={`flex h-24 items-end ${statsRangeDays > 30 ? 'gap-px' : 'gap-1'}`}>
                  {dailyBuckets.map((day, i) => (
                    <div key={i} className="flex min-w-0 flex-1 flex-col items-center gap-0.5">
                      <div
                        className={`relative flex w-full flex-1 items-end overflow-hidden rounded-sm ${
                          day.attempts > 0
                            ? (isDarkMode ? 'bg-slate-700' : 'bg-slate-200')
                            : (isDarkMode ? 'bg-slate-800' : 'bg-slate-100')
                        }`}
                        title={day.attempts > 0
                          ? `${day.label}: ${day.passes}/${day.attempts} (${(day.rate * 100).toFixed(0)}%)`
                          : `${day.label}: no practice`}
                      >
                        {day.attempts > 0 ? (
                          <div
                            className={`w-full rounded-sm ${proficiencyBarClass(day.rate)}`}
                            style={{ height: `${Math.max(4, day.rate * 100)}%` }}
                          />
                        ) : null}
                      </div>
                      {labelIndices.has(i) ? (
                        <span className={`truncate text-[9px] font-medium ${isDarkMode ? 'text-slate-500' : 'text-slate-400'}`}>{day.label}</span>
                      ) : <span className="h-3" />}
                    </div>
                  ))}
                </div>
                <div className={`mt-2 flex flex-wrap gap-4 text-xs ${isDarkMode ? 'text-slate-500' : 'text-slate-400'}`}>
                  <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-sm bg-emerald-500" />strong day</span>
                  <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-sm bg-amber-400" />mixed day</span>
                  <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-sm bg-rose-500" />tough day</span>
                  <span className="flex items-center gap-1"><span className={`inline-block h-2 w-2 rounded-sm ${isDarkMode ? 'bg-slate-800' : 'bg-slate-100'}`} />no practice</span>
                </div>
              </div>

              {/* Card proficiencies */}
              <div>
                <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                  <div>
                    <p className={`text-sm font-semibold ${isDarkMode ? 'text-slate-300' : 'text-slate-700'}`}>Card Proficiencies</p>
                    <p className={`mt-0.5 text-xs ${isDarkMode ? 'text-slate-500' : 'text-slate-400'}`}>
                      Sorted {statsSortDir === 'weakest' ? 'weakest to strongest' : 'strongest to weakest'}
                    </p>
                  </div>
                  <button
                    onClick={() => setStatsSortDir((prev) => prev === 'weakest' ? 'strongest' : 'weakest')}
                    className={`flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-semibold ${isDarkMode ? 'border-slate-600 text-slate-300 hover:bg-slate-800' : 'border-slate-300 text-slate-700 hover:bg-slate-50'}`}
                    title="Reverse sort order"
                  >
                    {statsSortDir === 'weakest'
                      ? <ChevronUp className="h-3.5 w-3.5" />
                      : <ChevronDown className="h-3.5 w-3.5" />}
                    <span>{statsSortDir === 'weakest' ? 'Weakest first' : 'Strongest first'}</span>
                  </button>
                </div>
                {statsWords.length === 0 ? (
                  <p className={`text-sm ${isDarkMode ? 'text-slate-500' : 'text-slate-400'}`}>No cards in this set.</p>
                ) : (
                  <div className="space-y-2">
                    {wordStats.map((ws) => (
                      <div
                        key={ws.word}
                        className={`rounded-xl border p-3.5 ${isDarkMode ? 'border-slate-700 bg-slate-800' : 'border-slate-200 bg-slate-50'}`}
                      >
                        <div className="flex items-baseline justify-between gap-3">
                          <span className={`text-base font-black tracking-wide ${isDarkMode ? 'text-slate-100' : 'text-slate-900'}`}>{ws.word}</span>
                          <span className={`flex-shrink-0 text-xs font-medium ${isDarkMode ? 'text-slate-500' : 'text-slate-400'}`}>
                            {ws.attempts === 0 ? 'Not practiced yet' : `${ws.attempts} check${ws.attempts !== 1 ? 's' : ''}`}
                          </span>
                        </div>

                        <div className="mt-2.5">
                          <div className="flex items-baseline justify-between gap-2">
                            <span className={`text-xs font-semibold ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>Pass rate</span>
                            <span className={`text-sm font-black ${ws.passRate !== null ? proficiencyTextClass(ws.passRate) : isDarkMode ? 'text-slate-600' : 'text-slate-300'}`}>
                              {ws.passRate !== null ? `${(ws.passRate * 100).toFixed(0)}%` : '—'}
                            </span>
                          </div>
                          <div className={`mt-1 h-2.5 w-full overflow-hidden rounded-full ${isDarkMode ? 'bg-slate-700' : 'bg-slate-200'}`}>
                            {ws.passRate !== null ? (
                              <div
                                className={`h-full rounded-full transition-all duration-300 ${proficiencyBarClass(ws.passRate)}`}
                                style={{ width: `${Math.max(2, ws.passRate * 100)}%` }}
                              />
                            ) : null}
                          </div>
                        </div>

                        <div className="mt-2.5">
                          <div className="flex items-baseline justify-between gap-2">
                            <span className={`text-xs font-semibold ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>Average match score</span>
                            <span className={`text-sm font-black ${ws.avgSim !== null ? proficiencyTextClass(ws.avgSim) : isDarkMode ? 'text-slate-600' : 'text-slate-300'}`}>
                              {ws.avgSim !== null ? `${(ws.avgSim * 100).toFixed(0)}%` : '—'}
                            </span>
                          </div>
                          <div className={`mt-1 h-2.5 w-full overflow-hidden rounded-full ${isDarkMode ? 'bg-slate-700' : 'bg-slate-200'}`}>
                            {ws.avgSim !== null ? (
                              <div
                                className={`h-full rounded-full transition-all duration-300 ${proficiencyBarClass(ws.avgSim)}`}
                                style={{ width: `${Math.max(2, ws.avgSim * 100)}%` }}
                              />
                            ) : null}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </section>
          </main>
        );
      })() : appPage === 'profile' ? (
        <main className="w-full h-[calc(100vh-4rem)] p-3 lg:p-4">
          <section className={`mx-auto h-full max-w-4xl overflow-y-auto rounded-2xl border p-6 shadow-sm ${isDarkMode ? 'border-slate-700 bg-slate-900 text-slate-100' : 'border-slate-200 bg-white text-slate-900'}`}>
            <div className="mb-5 flex items-center gap-3">
              <button
                onClick={() => setAppPage('learn')}
                className={`rounded-lg border px-3 py-2 text-sm font-semibold ${isDarkMode ? 'border-slate-600 text-slate-300 hover:bg-slate-800' : 'border-slate-200 text-slate-700 hover:bg-slate-50'}`}
              >
                Back
              </button>
              <h2 className={`text-3xl font-black tracking-tight ${isDarkMode ? 'text-slate-100' : 'text-slate-900'}`}>Profile</h2>
            </div>

            {!user ? (
              <div className="flex flex-col items-center gap-4 py-12">
                <p className={`text-center text-sm ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>Sign in to access cloud sync and account management.</p>
                <button onClick={() => setShowAuthModal(true)} className="flex items-center gap-2 rounded-xl bg-indigo-600 px-5 py-2.5 text-sm font-bold text-white hover:bg-indigo-700">
                  <LogIn className="h-4 w-4" />
                  Sign In / Create Account
                </button>
              </div>
            ) : (
              <>
                {/* Account info */}
                <div className={`mb-5 rounded-xl border p-4 ${isDarkMode ? 'border-slate-700 bg-slate-800' : 'border-slate-200 bg-slate-50'}`}>
                  <p className={`mb-3 text-sm font-semibold ${isDarkMode ? 'text-slate-300' : 'text-slate-700'}`}>Account</p>
                  <div className="flex items-center gap-3">
                    <div className={`flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full ${isDarkMode ? 'bg-slate-700' : 'bg-slate-200'}`}>
                      <User className="h-5 w-5" />
                    </div>
                    <div className="min-w-0">
                      {user.displayName ? <p className="truncate font-semibold">{user.displayName}</p> : null}
                      <p className={`truncate text-sm ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>{user.email}</p>
                      <span className={`mt-0.5 inline-block rounded-full px-2 py-0.5 text-xs font-semibold ${user.providerData[0]?.providerId === 'google.com' ? 'bg-blue-100 text-blue-700' : 'bg-slate-100 text-slate-600'}`}>
                        {user.providerData[0]?.providerId === 'google.com' ? 'Google' : 'Email / Password'}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Cloud sync */}
                <div className={`mb-5 rounded-xl border p-4 ${isDarkMode ? 'border-slate-700 bg-slate-800' : 'border-slate-200 bg-slate-50'}`}>
                  <p className={`mb-3 text-sm font-semibold ${isDarkMode ? 'text-slate-300' : 'text-slate-700'}`}>Cloud Sync</p>
                  <div className="flex items-center justify-between gap-3">
                    <div className="flex items-center gap-2">
                      {syncStatus === 'syncing' ? (
                        <><LoaderCircle className="h-4 w-4 animate-spin text-indigo-500" /><span className="text-sm text-indigo-500">Syncing…</span></>
                      ) : syncStatus === 'done' ? (
                        <><Cloud className="h-4 w-4 text-emerald-500" /><span className="text-sm text-emerald-500">Synced{lastSyncedAt ? ` · ${new Date(lastSyncedAt).toLocaleTimeString()}` : ''}</span></>
                      ) : syncStatus === 'error' ? (
                        <><CloudOff className="h-4 w-4 text-rose-500" /><span className="text-sm text-rose-500">Sync error — try again</span></>
                      ) : (
                        <span className={`text-sm ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>Not yet synced this session</span>
                      )}
                    </div>
                    <button
                      onClick={handleSyncNow}
                      disabled={syncStatus === 'syncing'}
                      className="flex items-center gap-1.5 rounded-lg bg-indigo-600 px-3 py-1.5 text-xs font-bold text-white hover:bg-indigo-700 disabled:opacity-50"
                    >
                      {syncStatus === 'syncing' ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : <Cloud className="h-3.5 w-3.5" />}
                      Sync Now
                    </button>
                  </div>
                </div>

                {/* Change password (email users only) */}
                {user.providerData[0]?.providerId === 'password' ? (
                  <div className={`mb-5 rounded-xl border p-4 ${isDarkMode ? 'border-slate-700 bg-slate-800' : 'border-slate-200 bg-slate-50'}`}>
                    <p className={`mb-3 text-sm font-semibold ${isDarkMode ? 'text-slate-300' : 'text-slate-700'}`}>Change Password</p>
                    <div className="flex gap-2">
                      <input
                        type="password"
                        placeholder="New password (min 8 chars)"
                        value={newPassword}
                        onChange={(e) => setNewPassword(e.target.value)}
                        onKeyDown={(e) => { if (e.key === 'Enter') handleChangePassword(); }}
                        className={`flex-1 rounded-lg border px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-indigo-400 ${isDarkMode ? 'border-slate-600 bg-slate-700 text-slate-100' : 'border-slate-300 bg-white text-slate-900'}`}
                      />
                      <button
                        onClick={handleChangePassword}
                        disabled={newPassword.length < 1}
                        className="rounded-lg bg-indigo-600 px-3 py-2 text-sm font-bold text-white hover:bg-indigo-700 disabled:opacity-40"
                      >
                        Update
                      </button>
                    </div>
                    {changePasswordError ? <p className="mt-2 text-xs text-rose-600">{changePasswordError}</p> : null}
                    {changePasswordSuccess ? <p className="mt-2 text-xs text-emerald-600">Password updated successfully!</p> : null}
                  </div>
                ) : null}

                {/* Danger zone */}
                <div className={`rounded-xl border p-4 ${isDarkMode ? 'border-rose-900 bg-rose-950/30' : 'border-rose-200 bg-rose-50'}`}>
                  <p className="mb-3 text-sm font-semibold text-rose-600">Danger Zone</p>
                  {showDeleteAccountConfirm ? (
                    <div>
                      <p className={`mb-2 text-sm ${isDarkMode ? 'text-slate-300' : 'text-slate-700'}`}>
                        Permanently delete your account? This cannot be undone.
                      </p>
                      <p className={`mb-3 text-sm font-semibold ${isDarkMode ? 'text-slate-200' : 'text-slate-800'}`}>
                        Also clear all local data (baselines, sets, history) from this device?
                      </p>
                      <div className="flex flex-wrap gap-2">
                        <button
                          onClick={() => handleDeleteAccount(true)}
                          className="rounded-lg bg-rose-600 px-3 py-2 text-xs font-bold text-white hover:bg-rose-700"
                        >
                          Delete Account + Clear Local Data
                        </button>
                        <button
                          onClick={() => handleDeleteAccount(false)}
                          className={`rounded-lg border px-3 py-2 text-xs font-bold ${isDarkMode ? 'border-rose-700 text-rose-400 hover:bg-rose-900/30' : 'border-rose-300 text-rose-700 hover:bg-rose-100'}`}
                        >
                          Delete Account (Keep Local Data)
                        </button>
                        <button
                          onClick={() => setShowDeleteAccountConfirm(false)}
                          className={`rounded-lg border px-3 py-2 text-xs font-semibold ${isDarkMode ? 'border-slate-600 text-slate-300 hover:bg-slate-800' : 'border-slate-300 text-slate-700 hover:bg-slate-50'}`}
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  ) : (
                    <button
                      onClick={() => setShowDeleteAccountConfirm(true)}
                      className="rounded-lg border border-rose-400 px-3 py-2 text-sm font-semibold text-rose-600 hover:bg-rose-100"
                    >
                      Delete Account…
                    </button>
                  )}
                </div>
              </>
            )}
          </section>
        </main>
      ) : (
        <main className="w-full h-[calc(100vh-4rem)] p-3 lg:p-4">
          <section className={`mx-auto h-full max-w-4xl overflow-y-auto rounded-2xl border p-6 shadow-sm ${isDarkMode ? 'border-slate-700 bg-slate-900 text-slate-100' : 'border-slate-200 bg-white text-slate-900'}`}>
            <div className="mb-4 flex items-center gap-3">
              <button
                onClick={() => setAppPage('learn')}
                className={`rounded-lg border px-3 py-2 text-sm font-semibold ${isDarkMode ? 'border-slate-600 text-slate-300 hover:bg-slate-800' : 'border-slate-200 text-slate-700 hover:bg-slate-50'}`}
              >
                Back
              </button>
              <h2 className={`text-3xl font-black tracking-tight ${isDarkMode ? 'text-slate-100' : 'text-slate-900'}`}>About SignCards</h2>
            </div>
            <p className={`mt-1 ${isDarkMode ? 'text-slate-300' : 'text-slate-700'}`}>
              SignCards is a local-first ASL training app using personalized baseline verification from webcam hand landmarks.
            </p>
            <div className={`mt-3 rounded-lg border p-3 ${isDarkMode ? 'border-slate-600 bg-slate-800 text-slate-300' : 'border-slate-200 bg-slate-50 text-slate-700'}`}>
              <p className="font-semibold">Built with the community</p>
              <p className="mt-1 text-sm">
                SignCards is developed in collaboration with the McMaster ASL Club and the learners who
                test it, report bugs, and suggest signs to add. Their feedback shapes what gets built next.
              </p>
            </div>
            <div className={`mt-3 rounded-lg border p-3 ${isDarkMode ? 'border-slate-600 bg-slate-800 text-slate-300' : 'border-slate-200 bg-slate-50 text-slate-700'}`}>
              <div className="flex items-center gap-2">
                <Users className="h-4 w-4" />
                <p className="font-semibold">Build on it</p>
              </div>
              <p className="mt-1 text-sm">
                The project is open source. Fork it, remix it, or spin it off for another signed language,
                a different curriculum, or your own club. Pull requests, new card sets, and independent
                offshoots are all welcome — you do not need permission to start.
              </p>
            </div>
            <div className={`mt-4 rounded-lg border p-3 ${isDarkMode ? 'border-slate-600 bg-slate-800' : 'border-slate-200 bg-slate-50'}`}>
              <p className="font-semibold">GitHub Repository</p>
              <a
                href="https://github.com/aslsigncards/SignCards"
                target="_blank"
                rel="noopener noreferrer"
                className="mt-1 inline-flex items-center gap-2 text-indigo-400 underline"
              >
                <GitBranch className="h-4 w-4" />
                <span>aslsigncards/SignCards</span>
              </a>
            </div>
            <div className={`mt-3 rounded-lg border p-3 ${isDarkMode ? 'border-slate-600 bg-slate-800' : 'border-slate-200 bg-slate-50'}`}>
              <p className="font-semibold">Questions</p>
              <a
                href="mailto:aslsigncards@gmail.com"
                className="mt-1 inline-flex items-center gap-2 text-indigo-400 underline"
              >
                <Mail className="h-4 w-4" />
                <span>aslsigncards@gmail.com</span>
              </a>
            </div>
          </section>
        </main>
      )}

      {isSidebarOpen ? (
            <button
              onClick={() => setIsSidebarOpen(false)}
              className="fixed inset-0 z-[70] bg-black/35"
              aria-label="Close sidebar overlay"
            />
          ) : null}

          <aside
            className={`fixed inset-y-0 left-0 z-[75] h-screen w-[360px] max-w-[92vw] border-r shadow-2xl transition-transform duration-300 ${
              isDarkMode ? 'border-slate-700 bg-slate-900' : 'border-slate-200 bg-white'
            } ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'}`}
          >
            <div className="flex h-full flex-col">
              <div className={`flex items-center justify-between border-b px-3 py-3 ${isDarkMode ? 'border-slate-700' : 'border-slate-200'}`}>
                <button
                  onClick={() => setIsSidebarOpen(false)}
                  className={`rounded-lg border p-2 ${isDarkMode ? 'border-slate-600 text-slate-300 hover:bg-slate-800' : 'border-slate-200 text-slate-700 hover:bg-slate-50'}`}
                  title="Close sidebar"
                >
                  <X className="h-4 w-4" />
                </button>
                <button
                  onClick={() => {
                    setView('landing');
                    setIsSidebarOpen(false);
                  }}
                  className={`rounded-lg border p-2 ${isDarkMode ? 'border-slate-600 text-slate-300 hover:bg-slate-800' : 'border-slate-200 text-slate-700 hover:bg-slate-50'}`}
                  title="Go to homepage"
                >
                  <Home className="h-4 w-4" />
                </button>
              </div>

              <div className={`flex-1 min-h-0 overflow-y-auto p-3 text-sm ${isDarkMode ? 'text-slate-300' : 'text-slate-700'}`}>
                {user ? (
                  <div className={`mb-3 rounded-xl border p-3 ${isDarkMode ? 'border-slate-700 bg-slate-800' : 'border-slate-200 bg-slate-50'}`}>
                    <div className="flex items-center gap-2 min-w-0">
                      <User className="h-4 w-4 flex-shrink-0 text-indigo-500" />
                      <span className="text-sm font-semibold truncate">{user.displayName || user.email}</span>
                    </div>
                    <div className="mt-1.5 flex items-center gap-1.5">
                      {syncStatus === 'syncing' ? (
                        <><LoaderCircle className="h-3 w-3 animate-spin text-indigo-400" /><span className="text-xs text-indigo-400">Syncing…</span></>
                      ) : syncStatus === 'done' ? (
                        <><Cloud className="h-3 w-3 text-emerald-400" /><span className="text-xs text-emerald-400">Synced</span></>
                      ) : syncStatus === 'error' ? (
                        <><CloudOff className="h-3 w-3 text-rose-400" /><span className="text-xs text-rose-400">Sync error</span></>
                      ) : (
                        <span className={`text-xs ${isDarkMode ? 'text-slate-500' : 'text-slate-400'}`}>Cloud sync enabled</span>
                      )}
                    </div>
                  </div>
                ) : !authLoading ? (
                  <button
                    onClick={() => { setShowAuthModal(true); setIsSidebarOpen(false); }}
                    className="mb-3 flex w-full items-center justify-center gap-2 rounded-xl bg-indigo-600 px-3 py-3 text-sm font-bold text-white hover:bg-indigo-700"
                  >
                    <LogIn className="h-4 w-4" />
                    <span>Sign In / Create Account</span>
                  </button>
                ) : null}
                <button
                  onClick={() => {
                    setSidebarSection((prev) => (prev === 'sets' ? 'none' : 'sets'));
                  }}
                  className={`mb-2 flex w-full items-center justify-between rounded-lg border px-3 py-3 text-left font-semibold ${isDarkMode ? 'border-slate-600 bg-slate-800 text-slate-100 hover:bg-slate-700' : 'border-slate-200 bg-white text-slate-800 hover:bg-slate-50'}`}
                >
                  <span>Select Set</span>
                  {sidebarSection === 'sets' ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                </button>

                {sidebarSection === 'sets' ? (
                  <div className={`mb-2 space-y-1.5 rounded-lg border p-2 ${isDarkMode ? 'border-slate-600 bg-slate-800' : 'border-slate-200 bg-slate-50'}`}>
                    {availableSets.map((set) => (
                      <div
                        key={set.key}
                        className={`group flex items-center gap-2 rounded-lg border px-3 py-2 transition-colors ${
                          currentSet === set.key
                            ? 'border-indigo-200 bg-indigo-50 text-indigo-700'
                            : isDarkMode
                            ? 'border-slate-600 bg-slate-700 text-slate-100 hover:bg-slate-600'
                            : 'border-slate-200 bg-white text-slate-700 hover:bg-slate-50'
                        }`}
                      >
                        <button
                          onClick={() => handleSetSelect(set.key)}
                          className="flex-1 text-left"
                        >
                          <p className="font-semibold">{set.label}</p>
                          <p className="text-xs opacity-70">{set.words.length} cards</p>
                        </button>
                        {set.isCustom ? (
                          <>
                            <button
                              onClick={() => handleEditSet(set)}
                              className={`rounded-md border p-1 ${isDarkMode ? 'border-indigo-500 text-indigo-300 hover:bg-indigo-900/30' : 'border-indigo-300 text-indigo-700 hover:bg-indigo-50'}`}
                              title="Edit set"
                            >
                              <Pencil className="h-5 w-5" strokeWidth={2.5} />
                            </button>
                            <button
                              onClick={() => setSetDeleteCandidate(set)}
                              className={`rounded-md border p-1 ${isDarkMode ? 'border-rose-500 text-rose-300 hover:bg-rose-900/30' : 'border-rose-300 text-rose-700 hover:bg-rose-50'}`}
                              title="Delete set"
                            >
                              <Trash2 className="h-5 w-5" strokeWidth={2.5} />
                            </button>
                          </>
                        ) : null}
                      </div>
                    ))}
                  </div>
                ) : null}

                <button
                  onClick={() => {
                    setAppPage('create');
                    setIsSidebarOpen(false);
                  }}
                  className={`mb-2 flex w-full items-center gap-2 rounded-lg border px-3 py-3 text-left font-semibold ${isDarkMode ? 'border-slate-600 bg-slate-800 text-slate-100 hover:bg-slate-700' : 'border-slate-200 bg-white text-slate-800 hover:bg-slate-50'}`}
                >
                  <Plus className="h-4 w-4" />
                  <span>Create New Set</span>
                </button>

                <button
                  onClick={() => {
                    setAppPage('stats');
                    setIsSidebarOpen(false);
                  }}
                  className={`mb-2 flex w-full items-center gap-2 rounded-lg border px-3 py-3 text-left font-semibold ${isDarkMode ? 'border-slate-600 bg-slate-800 text-slate-100 hover:bg-slate-700' : 'border-slate-200 bg-white text-slate-800 hover:bg-slate-50'}`}
                >
                  <BarChart2 className="h-4 w-4" />
                  <span>Stats</span>
                </button>

                {user ? (
                  <button
                    onClick={() => { setAppPage('profile'); setIsSidebarOpen(false); }}
                    className={`mb-2 flex w-full items-center gap-2 rounded-lg border px-3 py-3 text-left font-semibold ${isDarkMode ? 'border-slate-600 bg-slate-800 text-slate-100 hover:bg-slate-700' : 'border-slate-200 bg-white text-slate-800 hover:bg-slate-50'}`}
                  >
                    <User className="h-4 w-4" />
                    <span>Profile</span>
                  </button>
                ) : null}

                <button
                  onClick={() => {
                    setAppPage('settings');
                    setIsSidebarOpen(false);
                  }}
                  className={`mb-2 flex w-full items-center gap-2 rounded-lg border px-3 py-3 text-left font-semibold ${isDarkMode ? 'border-slate-600 bg-slate-800 text-slate-100 hover:bg-slate-700' : 'border-slate-200 bg-white text-slate-800 hover:bg-slate-50'}`}
                >
                  <Settings className="h-4 w-4" />
                  <span>Settings</span>
                </button>

                <button
                  onClick={() => {
                    startTutorial();
                    setIsSidebarOpen(false);
                  }}
                  className={`mb-2 flex w-full items-center gap-2 rounded-lg border px-3 py-3 text-left font-semibold ${isDarkMode ? 'border-slate-600 bg-slate-800 text-slate-100 hover:bg-slate-700' : 'border-slate-200 bg-white text-slate-800 hover:bg-slate-50'}`}
                >
                  <CircleHelp className="h-4 w-4" />
                  <span>Watch Tutorial</span>
                </button>

                <button
                  onClick={() => {
                    setAppPage('about');
                    setIsSidebarOpen(false);
                  }}
                  className={`mb-2 flex w-full items-center gap-2 rounded-lg border px-3 py-3 text-left font-semibold ${isDarkMode ? 'border-slate-600 bg-slate-800 text-slate-100 hover:bg-slate-700' : 'border-slate-200 bg-white text-slate-800 hover:bg-slate-50'}`}
                >
                  <Users className="h-4 w-4" />
                  <span>About</span>
                </button>
              </div>
              {user ? (
                <div className={`flex-shrink-0 border-t p-3 ${isDarkMode ? 'border-slate-700' : 'border-slate-200'}`}>
                  <button
                    onClick={handleSignOut}
                    className={`flex w-full items-center gap-2 rounded-lg border px-3 py-2.5 text-left text-sm font-semibold ${isDarkMode ? 'border-slate-600 text-slate-300 hover:bg-slate-800' : 'border-slate-200 text-slate-700 hover:bg-slate-50'}`}
                  >
                    <LogOut className="h-4 w-4" />
                    <span>Sign Out</span>
                  </button>
                </div>
              ) : null}
            </div>
          </aside>

      {tooltipOpenId ? (
        <>
          <button
            className="fixed inset-0 z-[40] cursor-default"
            onClick={() => setTooltipOpenId(null)}
            aria-label="Close tooltip"
          />
          <div
            className={`fixed z-[45] w-72 rounded-xl border p-3 text-xs shadow-2xl ${isDarkMode ? 'border-slate-600 bg-slate-800 text-slate-200' : 'border-slate-200 bg-white text-slate-700'}`}
            style={{ top: tooltipPosition.top, left: tooltipPosition.left }}
          >
            <p className={`mb-1.5 font-bold text-sm ${isDarkMode ? 'text-slate-100' : 'text-slate-900'}`}>Multi-sign</p>
            <p className="mb-1.5">Use for signs that require multiple sequential handshapes:</p>
            <ul className="mb-2 list-inside list-disc space-y-1">
              <li>Fingerspelled words (e.g. F-R-I-E-N-D)</li>
              <li>Compound signs (e.g. TEACHER: TEACH + PERSON)</li>
              <li>Phrases (e.g. What is your name?)</li>
            </ul>
            <p className={`border-t pt-2 ${isDarkMode ? 'border-slate-700 text-slate-400' : 'border-slate-100 text-slate-500'}`}>Enter each component sign label separated by commas in the field that appears below the card.</p>
          </div>
        </>
      ) : null}

      {pendingSuccessSet ? (
        <div className="fixed inset-0 z-[90] flex items-center justify-center bg-black/55 p-4">
          <div className={`w-full max-w-md rounded-2xl border p-6 shadow-2xl ${isDarkMode ? 'border-slate-700 bg-slate-800 text-slate-100' : 'border-slate-200 bg-white text-slate-900'}`}>
            <h2 className="text-xl font-black tracking-tight">
              {pendingSuccessSet.isEdit ? 'Set updated!' : 'Congrats on your new set!'}
            </h2>
            <p className={`mt-2 text-sm ${isDarkMode ? 'text-slate-300' : 'text-slate-700'}`}>
              {pendingSuccessSet.isEdit
                ? `\u201c${pendingSuccessSet.title}\u201d has been updated. ${
                    pendingSuccessSet.newCount > 0
                      ? `${pendingSuccessSet.newCount} card(s) still need a recorded baseline.`
                      : 'All cards already have baselines.'
                  }`
                : `\u201c${pendingSuccessSet.title}\u201d is ready with ${pendingSuccessSet.wordCount} card(s). Next up: record your sign definitions so the app can verify your accuracy.`}
            </p>
            <div className="mt-5 flex flex-wrap gap-2">
              <button
                onClick={() => {
                  handleSetSelect(`custom:${pendingSuccessSet.id}`);
                  setPendingSuccessSet(null);
                  setAppPage('learn');
                }}
                className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-bold text-white hover:bg-indigo-700"
              >
                Get Started
              </button>
              <button
                onClick={() => {
                  setPendingSuccessSet(null);
                  setAppPage('learn');
                }}
                className={`rounded-lg border px-4 py-2 text-sm font-semibold ${isDarkMode ? 'border-slate-600 text-slate-300 hover:bg-slate-700' : 'border-slate-300 text-slate-700 hover:bg-slate-50'}`}
              >
                Not Now
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {setDeleteCandidate ? (
        <div className="fixed inset-0 z-[85] flex items-center justify-center bg-black/55 p-4">
          <div className={`w-full max-w-md rounded-2xl border p-6 shadow-2xl ${isDarkMode ? 'border-slate-700 bg-slate-800 text-slate-100' : 'border-slate-200 bg-white text-slate-900'}`}>
            <h2 className="text-xl font-black tracking-tight">Delete Set?</h2>
            <p className={`mt-2 text-sm ${isDarkMode ? 'text-slate-300' : 'text-slate-700'}`}>
              Are you sure you want to delete "{setDeleteCandidate.label}"? This also removes all saved baselines for that set.
            </p>
            <div className="mt-5 flex flex-wrap gap-2">
              <button
                onClick={() => handleDeleteSet(setDeleteCandidate)}
                className="rounded-lg bg-rose-600 px-4 py-2 text-sm font-bold text-white hover:bg-rose-700"
              >
                Yes, Delete
              </button>
              <button
                onClick={() => setSetDeleteCandidate(null)}
                className={`rounded-lg border px-4 py-2 text-sm font-semibold ${isDarkMode ? 'border-slate-600 text-slate-300 hover:bg-slate-700' : 'border-slate-300 text-slate-700 hover:bg-slate-50'}`}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {pendingImportData ? (
        <div className="fixed inset-0 z-[90] flex items-center justify-center bg-black/55 p-4">
          <div className={`w-full max-w-md rounded-2xl border p-6 shadow-2xl ${isDarkMode ? 'border-slate-700 bg-slate-800 text-slate-100' : 'border-slate-200 bg-white text-slate-900'}`}>
            <h2 className="text-xl font-black tracking-tight">Import Backup</h2>
            <p className={`mt-2 text-sm ${isDarkMode ? 'text-slate-300' : 'text-slate-700'}`}>
              Found <strong>{pendingImportData.customSets?.length ?? 0} set(s)</strong> and{' '}
              <strong>{pendingImportData.references?.length ?? 0} baseline(s)</strong> in this file.
            </p>
            <div className={`mt-3 rounded-lg border p-3 text-xs ${isDarkMode ? 'border-slate-700 bg-slate-900 text-slate-400' : 'border-slate-200 bg-slate-50 text-slate-600'}`}>
              <p className="mb-1"><strong>Merge</strong> — adds new sets, skips any whose title matches an existing set.</p>
              <p><strong>Replace All</strong> — clears all sets, baselines, and history before restoring the file.</p>
            </div>
            <div className="mt-5 flex flex-wrap gap-2">
              <button
                onClick={() => handleImportConfirm('merge')}
                className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-bold text-white hover:bg-indigo-700"
              >
                Merge
              </button>
              <button
                onClick={() => handleImportConfirm('replace')}
                className="rounded-lg bg-rose-600 px-4 py-2 text-sm font-bold text-white hover:bg-rose-700"
              >
                Replace All
              </button>
              <button
                onClick={() => setPendingImportData(null)}
                className={`rounded-lg border px-4 py-2 text-sm font-semibold ${isDarkMode ? 'border-slate-600 text-slate-300 hover:bg-slate-700' : 'border-slate-300 text-slate-700 hover:bg-slate-50'}`}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {showAllBaselinesModal ? (
        <div className="fixed inset-0 z-[90] flex items-center justify-center bg-black/55 p-4">
          <div className={`w-full max-w-md rounded-2xl border p-6 shadow-2xl ${isDarkMode ? 'border-slate-700 bg-slate-800 text-slate-100' : 'border-slate-200 bg-white text-slate-900'}`}>
            <h2 className="text-xl font-black tracking-tight">All Baselines Recorded!</h2>
            <p className={`mt-2 text-sm ${isDarkMode ? 'text-slate-300' : 'text-slate-700'}`}>
              All {activeWords.length} sign{activeWords.length !== 1 ? 's' : ''} now have baselines recorded. Ready to practice?
            </p>
            <div className="mt-5 flex gap-2">
              <button
                onClick={() => {
                  setShowAllBaselinesModal(false);
                  enterPractice();
                }}
                className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-bold text-white hover:bg-emerald-700"
              >
                Go to Practice
              </button>
              <button
                onClick={() => setShowAllBaselinesModal(false)}
                className={`rounded-lg border px-4 py-2 text-sm font-semibold ${isDarkMode ? 'border-slate-600 text-slate-300 hover:bg-slate-700' : 'border-slate-300 text-slate-700 hover:bg-slate-50'}`}
              >
                Keep Recording
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {showPracticeWithMissingModal ? (
        <div className="fixed inset-0 z-[90] flex items-center justify-center bg-black/55 p-4">
          <div className={`w-full max-w-md rounded-2xl border p-6 shadow-2xl ${isDarkMode ? 'border-slate-700 bg-slate-800 text-slate-100' : 'border-slate-200 bg-white text-slate-900'}`}>
            <h2 className="text-xl font-black tracking-tight">{missingBaselineCount} Baseline{missingBaselineCount !== 1 ? 's' : ''} Missing</h2>
            <p className={`mt-2 text-sm ${isDarkMode ? 'text-slate-300' : 'text-slate-700'}`}>
              {missingBaselineCount} sign{missingBaselineCount !== 1 ? "s don't" : " doesn't"} have a baseline yet and won't appear in practice. You'll practice with the {wordsWithBaseline.length} recorded sign{wordsWithBaseline.length !== 1 ? 's' : ''}. Continue?
            </p>
            <div className="mt-5 flex gap-2">
              <button
                onClick={() => {
                  setShowPracticeWithMissingModal(false);
                  enterPractice();
                }}
                className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-bold text-white hover:bg-indigo-700"
              >
                Continue
              </button>
              <button
                onClick={() => setShowPracticeWithMissingModal(false)}
                className={`rounded-lg border px-4 py-2 text-sm font-semibold ${isDarkMode ? 'border-slate-600 text-slate-300 hover:bg-slate-700' : 'border-slate-300 text-slate-700 hover:bg-slate-50'}`}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {showPracticeInstructions ? (
        <div className="fixed inset-0 z-[90] flex items-center justify-center bg-black/55 p-4">
          <div className={`w-full max-w-md rounded-2xl border p-6 shadow-2xl ${isDarkMode ? 'border-slate-700 bg-slate-800 text-slate-100' : 'border-slate-200 bg-white text-slate-900'}`}>
            <h2 className="text-xl font-black tracking-tight">How Practice Works</h2>
            <ol className={`mt-4 list-decimal space-y-2 pl-5 text-sm ${isDarkMode ? 'text-slate-300' : 'text-slate-700'}`}>
              <li>Click <strong>Check My Sign</strong> — a <strong>1.5-second countdown</strong> gives you time to get ready.</li>
              <li>The app <strong>records for 1 second</strong> — perform your sign during this window.</li>
              <li>Your sign is compared <strong>frame-by-frame</strong> against your recorded baseline.</li>
              <li>For <strong>one-handed signs</strong>, keep your other hand out of frame — both hands are captured if visible.</li>
              <li>Use the <strong>In Order / Random</strong> toggle in the header to switch practice modes.</li>
            </ol>
            <div className="mt-5">
              <button
                onClick={() => setShowPracticeInstructions(false)}
                className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-bold text-white hover:bg-indigo-700"
              >
                Got It
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {showRecordingInstructions ? (
        <div className="fixed inset-0 z-[90] flex items-center justify-center bg-black/55 p-4">
          <div className={`w-full max-w-md rounded-2xl border p-6 shadow-2xl ${isDarkMode ? 'border-slate-700 bg-slate-800 text-slate-100' : 'border-slate-200 bg-white text-slate-900'}`}>
            <h2 className="text-xl font-black tracking-tight">How Baseline Recording Works</h2>
            <ol className={`mt-4 list-decimal space-y-2 pl-5 text-sm ${isDarkMode ? 'text-slate-300' : 'text-slate-700'}`}>
              <li>A <strong>1.5-second countdown</strong> gives you time to get into position.</li>
              <li>The app then <strong>records for 1 second</strong> — perform your sign during this window.</li>
              <li>For <strong>static signs</strong>, hold the handshape steady. For <strong>dynamic signs</strong>, complete the full natural motion.</li>
              <li>Keep your hand <strong>clearly visible</strong> in the camera for the entire second.</li>
              <li>For <strong>one-handed signs</strong>, keep your other hand out of frame — both hands are captured if visible.</li>
            </ol>
            <div className="mt-5 flex gap-2">
              <button
                onClick={() => {
                  setShowRecordingInstructions(false);
                  startBufferedAction('record');
                }}
                className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-bold text-white hover:bg-indigo-700"
              >
                Got It, Start
              </button>
              <button
                onClick={() => setShowRecordingInstructions(false)}
                className={`rounded-lg border px-4 py-2 text-sm font-semibold ${isDarkMode ? 'border-slate-600 text-slate-300 hover:bg-slate-700' : 'border-slate-300 text-slate-700 hover:bg-slate-50'}`}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {showAuthModal ? (
        <div className="fixed inset-0 z-[90] flex items-center justify-center bg-black/55 p-4">
          <div className={`w-full max-w-sm rounded-2xl border p-6 shadow-2xl ${isDarkMode ? 'border-slate-700 bg-slate-800 text-slate-100' : 'border-slate-200 bg-white text-slate-900'}`}>
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-xl font-black">{authMode === 'signin' ? 'Sign In' : 'Create Account'}</h2>
              <button onClick={() => { setShowAuthModal(false); setAuthError(''); }} className={`rounded-lg p-1 ${isDarkMode ? 'hover:bg-slate-700' : 'hover:bg-slate-100'}`}>
                <X className="h-5 w-5" />
              </button>
            </div>

            {!firebaseConfigured ? (
              <p className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-700">
                Firebase is not configured yet. Add your VITE_FIREBASE_* keys to .env.local to enable cloud sync.
              </p>
            ) : (
              <>
                <button
                  onClick={handleSignInGoogle}
                  disabled={authSubmitting}
                  className={`flex w-full items-center justify-center gap-2 rounded-xl border px-4 py-2.5 text-sm font-semibold disabled:opacity-50 ${isDarkMode ? 'border-slate-600 text-slate-200 hover:bg-slate-700' : 'border-slate-200 text-slate-700 hover:bg-slate-50'}`}
                >
                  <svg className="h-4 w-4" viewBox="0 0 24 24">
                    <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
                    <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
                    <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z" fill="#FBBC05"/>
                    <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
                  </svg>
                  Continue with Google
                </button>
                <div className="my-4 flex items-center gap-3">
                  <div className={`flex-1 border-t ${isDarkMode ? 'border-slate-600' : 'border-slate-200'}`} />
                  <span className={`text-xs ${isDarkMode ? 'text-slate-500' : 'text-slate-400'}`}>or</span>
                  <div className={`flex-1 border-t ${isDarkMode ? 'border-slate-600' : 'border-slate-200'}`} />
                </div>
                <div className="space-y-3">
                  <input
                    type="email"
                    placeholder="Email"
                    value={authEmail}
                    onChange={(e) => setAuthEmail(e.target.value)}
                    className={`w-full rounded-lg border px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-indigo-400 ${isDarkMode ? 'border-slate-600 bg-slate-700 text-slate-100 placeholder:text-slate-500' : 'border-slate-300 bg-white text-slate-900'}`}
                  />
                  <input
                    type="password"
                    placeholder={authMode === 'signup' ? 'Password (min 8 chars)' : 'Password'}
                    value={authPassword}
                    onChange={(e) => setAuthPassword(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter') authMode === 'signin' ? handleSignInEmail() : handleSignUpEmail(); }}
                    className={`w-full rounded-lg border px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-indigo-400 ${isDarkMode ? 'border-slate-600 bg-slate-700 text-slate-100 placeholder:text-slate-500' : 'border-slate-300 bg-white text-slate-900'}`}
                  />
                </div>
                {authError ? <p className="mt-2 text-xs text-rose-500">{authError}</p> : null}
                <button
                  onClick={authMode === 'signin' ? handleSignInEmail : handleSignUpEmail}
                  disabled={authSubmitting || !authEmail || !authPassword}
                  className="mt-4 w-full rounded-xl bg-indigo-600 py-2.5 text-sm font-bold text-white hover:bg-indigo-700 disabled:opacity-50"
                >
                  {authSubmitting ? 'Please wait…' : authMode === 'signin' ? 'Sign In' : 'Create Account'}
                </button>
                <p className={`mt-3 text-center text-xs ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>
                  {authMode === 'signin' ? "Don't have an account? " : 'Already have an account? '}
                  <button
                    onClick={() => { setAuthMode(authMode === 'signin' ? 'signup' : 'signin'); setAuthError(''); }}
                    className="font-semibold text-indigo-600 underline"
                  >
                    {authMode === 'signin' ? 'Create one' : 'Sign in'}
                  </button>
                </p>
              </>
            )}
          </div>
        </div>
      ) : null}

      {showTutorial && view === 'app' ? (
        <div className="fixed inset-0 z-[80] bg-slate-950/65">
          {tutorialTargetRect ? (
            <div
              className="pointer-events-none fixed rounded-xl border-2 border-indigo-300 shadow-[0_0_0_9999px_rgba(2,6,23,0.65),0_0_24px_rgba(165,180,252,0.9)] transition-all duration-300"
              style={{ top: tutorialTargetRect.top - 6, left: tutorialTargetRect.left - 6, width: tutorialTargetRect.width + 12, height: tutorialTargetRect.height + 12 }}
            />
          ) : null}
          <div className="fixed inset-x-4 bottom-6 mx-auto max-w-md sm:bottom-10">
            <div className={`rounded-2xl border p-5 shadow-2xl ${isDarkMode ? 'border-indigo-400/40 bg-slate-800 text-slate-100' : 'border-indigo-200 bg-white text-slate-900'}`}>
              <div className="mb-3 flex items-center justify-between">
                <p className="text-xs font-bold uppercase tracking-widest text-indigo-500">Quick Start</p>
                <span className={`text-xs font-semibold ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>{tutorialStep + 1} of {TUTORIAL_STEPS.length}</span>
              </div>
              <h2 className="text-xl font-black tracking-tight">{TUTORIAL_STEPS[tutorialStep].title}</h2>
              <p className={`mt-2 text-sm leading-6 ${isDarkMode ? 'text-slate-300' : 'text-slate-600'}`}>{TUTORIAL_STEPS[tutorialStep].body}</p>
              <div className="mt-5 flex items-center justify-between gap-2">
                <button
                  onClick={() => tutorialStep === 0 ? closeTutorial(false) : setTutorialStep((step) => step - 1)}
                  className={`rounded-lg border px-3 py-2 text-sm font-semibold ${isDarkMode ? 'border-slate-600 text-slate-300 hover:bg-slate-700' : 'border-slate-300 text-slate-700 hover:bg-slate-50'}`}
                >
                  {tutorialStep === 0 ? 'Skip' : 'Back'}
                </button>
                <button
                  onClick={() => tutorialStep === TUTORIAL_STEPS.length - 1 ? closeTutorial(true) : setTutorialStep((step) => step + 1)}
                  className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-bold text-white hover:bg-indigo-700"
                >
                  {tutorialStep === TUTORIAL_STEPS.length - 1 ? 'Finish' : 'Next'}
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : null}

      <input
        ref={importFileInputRef}
        type="file"
        accept=".json,application/json"
        className="hidden"
        onChange={handleImportFile}
      />
    </div>
  );
}
