import { DEFAULT_MATCH_FEATURES } from './constants';

// Matching tuning. Lower tolerances mean stricter grading.
export const SHAPE_DISTANCE_TOLERANCE = 0.45; // normalized units before a frame scores 0
export const MOTION_DISTANCE_TOLERANCE = 1.2; // hand-lengths before a trajectory point scores 0
export const MOTION_DYNAMIC_PATH = 0.6; // wrist travel (hand-lengths) that marks a sign as dynamic
export const FINGER_TOLERANCE = 0.28; // mean extension difference before the finger score hits 0
export const DTW_BAND_RATIO = 0.3; // Sakoe-Chiba warping band as a fraction of sequence length
export const MOTION_FAIL_CAP = 0.5; // ceiling when a dynamic sign was performed static
export const MOTION_PARTIAL_CAP = 0.64; // ceiling when movement is present but clearly wrong
export const TIP_PAIR_TOLERANCE = 0.5; // hand-lengths of fingertip gap difference before scoring 0
export const TIP_PAIR_NOTE_DELTA = 0.22; // gap difference that earns a spacing note
export const SPLAY_TOLERANCE = 0.5; // radians of splay difference before scoring 0
export const CROSSING_TOLERANCE = 0.6; // signed-volume difference before scoring 0
export const CROSSING_DEADZONE = 0.15; // near-parallel fingers (U) sit near zero; keep margin
export const CROSSING_FAIL_CAP = 0.6; // ceiling when fingers are crossed the wrong way
export const THUMB_TOLERANCE = 0.6; // hand-lengths of thumb-distance difference before scoring 0
export const BODY_TOLERANCE = 1.0; // face-heights of vertical difference before scoring 0
export const BODY_NOTE_DELTA = 0.5; // vertical difference that earns a placement note

// Relative influence of each matching feature; renormalized over enabled features.
export const MATCH_WEIGHTS = {
  shape: 0.32,
  motion: 0.2,
  extension: 0.13,
  tipPairs: 0.09,
  splay: 0.06,
  crossing: 0.06,
  thumb: 0.04,
  bodyPosition: 0.1,
};

const TIP_PAIR_LABELS = ['index and middle', 'middle and ring', 'ring and pinky', 'thumb and index'];
const THUMB_TARGETS = ['index', 'middle', 'ring', 'pinky'];

/**
 * Vertical bands measured in face-heights below the top of the head.
 * Only vertical placement is compared; horizontal varies too much between
 * signers to test reliably.
 */
export const BODY_ZONES = [
  { key: 'forehead', label: 'forehead', max: 0.45 },
  { key: 'face', label: 'nose and mouth', max: 0.85 },
  { key: 'chin', label: 'chin', max: 1.35 },
  { key: 'chest', label: 'chest', max: Infinity },
];

export const bodyZoneFor = (value) =>
  BODY_ZONES.find((zone) => value < zone.max) ?? BODY_ZONES[BODY_ZONES.length - 1];

export const pointDistance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
export const clamp01 = (value) => Math.max(0, Math.min(1, value));

const median = (values) => {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

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

const meanAbsoluteScore = (refArr, capArr, tolerance) => {
  const diffs = refArr.map((value, i) => Math.abs(value - capArr[i]));
  return clamp01(1 - (diffs.reduce((a, b) => a + b, 0) / diffs.length) / tolerance);
};

const argMin = (values) => values.reduce((best, v, i) => (v < values[best] ? i : best), 0);

export function normalizeLandmarks(landmarks) {
  if (!landmarks || landmarks.length !== 21) return null;

  const wrist = landmarks[0];
  const centered = landmarks.map((lm) => ({
    x: lm.x - wrist.x,
    y: lm.y - wrist.y,
    z: lm.z - wrist.z,
  }));

  const scale = Math.hypot(centered[9].x, centered[9].y, centered[9].z) || 1;

  return centered.map((lm) => ({
    x: lm.x / scale,
    y: lm.y / scale,
    z: lm.z / scale,
  }));
}

/**
 * Face bounding box in normalized video coordinates, used as the vertical ruler
 * for body placement. Returns null when no face is visible.
 */
export function faceReference(detection, videoWidth, videoHeight) {
  const box = detection?.boundingBox;
  if (!box || !videoHeight) return null;
  const height = box.height / videoHeight;
  if (!(height > 0.01)) return null;
  return {
    top: box.originY / videoHeight,
    height,
    left: box.originX / (videoWidth || 1),
    width: box.width / (videoWidth || 1),
  };
}

/**
 * Wrist position plus hand size, captured before wrist-centering discards it.
 * `b` is the wrist height in face-heights below the top of the head.
 */
export function motionSample(landmarks, face) {
  if (!landmarks || landmarks.length !== 21) return null;
  const wrist = landmarks[0];
  const midMcp = landmarks[9];
  const s = Math.hypot(midMcp.x - wrist.x, midMcp.y - wrist.y, midMcp.z - wrist.z) || 1;
  const sample = { x: wrist.x, y: wrist.y, z: wrist.z, s };
  if (face) sample.b = (wrist.y - face.top) / face.height;
  return sample;
}

// Expressed in hand-lengths so results are independent of distance from camera.
export function normalizeMotion(samples) {
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

// Read separately from normalizeMotion, which recenters and would erase it.
function meanBodyHeight(samples) {
  if (!samples?.length) return null;
  const values = samples.map((s) => s.b).filter((v) => Number.isFinite(v));
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
}

export function motionFeatures(points) {
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
export function fingerExtensions(frame) {
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

/**
 * Geometry that finger extension alone cannot express: R, U and V all have the
 * index and middle extended and differ only in spacing and crossing.
 * Distances are in hand-lengths, angles in radians.
 */
export function handConfiguration(frame) {
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

export function calculateSimilarity(liveNorm, refNorm) {
  if (!liveNorm || !refNorm) return 0;

  let totalDist = 0;
  for (let i = 0; i < 21; i += 1) {
    const dx = liveNorm[i].x - refNorm[i].x;
    const dy = liveNorm[i].y - refNorm[i].y;
    const dz = liveNorm[i].z - refNorm[i].z;
    totalDist += Math.hypot(dx, dy, dz);
  }

  const avgDist = totalDist / 21;
  return clamp01(1 - avgDist / SHAPE_DISTANCE_TOLERANCE);
}

/**
 * Reduce sensor jitter with a sliding-window frame average.
 */
export function smoothFrames(frames, windowSize = 3) {
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
export function dtwSimilarity(seq1, seq2, frameCost) {
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

export function compareSequences(captured, reference) {
  return dtwSimilarity(smoothFrames(captured, 3), smoothFrames(reference, 3), shapeFrameCost);
}

export function hasRecordedBaseline(reference) {
  return Boolean(reference?.frames || reference?.frames2);
}

/**
 * Scores one hand across the enabled matching features and returns notes
 * explaining what was off. Weights are renormalized over whichever features are
 * enabled and have data, so toggling one off never skews the scale.
 */
export function analyzeHand(refFrames, refMotion, capFrames, capMotion, features = DEFAULT_MATCH_FEATURES) {
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

  // Skipped entirely when either recording lacks a face, so a detection miss
  // never costs points for a correctly performed sign.
  const refBody = meanBodyHeight(refMotion);
  const capBody = meanBodyHeight(capMotion);
  if (refBody !== null && capBody !== null) {
    const delta = Math.abs(refBody - capBody);
    const bodyScore = clamp01(1 - delta / BODY_TOLERANCE);
    contribute('bodyPosition', bodyScore, MATCH_WEIGHTS.bodyPosition);
    if (features.bodyPosition !== false && delta > BODY_NOTE_DELTA) {
      const refZone = bodyZoneFor(refBody);
      const capZone = bodyZoneFor(capBody);
      if (refZone.key !== capZone.key) {
        notes.push(`Sign this at ${refZone.label} level — you signed at ${capZone.label} level.`);
      } else {
        notes.push(capBody > refBody ? 'Sign this a little higher.' : 'Sign this a little lower.');
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
    hadBodyData: refBody !== null && capBody !== null,
  };
}
