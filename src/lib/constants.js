export const PRESETS = {
  fingerspelling: Array.from({ length: 26 }, (_, i) => String.fromCharCode(65 + i)),
  numbers: Array.from({ length: 11 }, (_, i) => String(i)),
};

export const READY_BUFFER_MS = 1500;
export const RECORDING_DURATION_MS = 1000; // ms to record hand motion
export const SEQUENCE_PASS_THRESHOLD = 0.65; // combined score needed to pass

export const TUTORIAL_STORAGE_KEY = 'asl-signcards-tutorial-seen-v1';
export const SETTINGS_STORAGE_KEY = 'asl-signcards-settings-v1';

export const STATS_RANGE_OPTIONS = [3, 7, 14, 30, 90];

export const MATCH_FEATURE_LIST = [
  { key: 'shape', label: 'Handshape (DTW)', description: 'Overall landmark match across the whole sign.' },
  { key: 'motion', label: 'Movement path', description: 'Requires dynamic signs to actually move, and static signs to stay put.' },
  { key: 'extension', label: 'Finger extension', description: 'Whether each finger is extended or curled in.' },
  { key: 'tipPairs', label: 'Fingertip spacing', description: 'Gaps between neighbouring fingertips. Separates U from V.' },
  { key: 'splay', label: 'Splay angles', description: 'Angles between finger directions.' },
  { key: 'crossing', label: 'Finger crossing', description: 'Detects crossed index and middle fingers, as in R.' },
  { key: 'thumb', label: 'Thumb position', description: 'Which finger the thumb sits nearest. Separates A, S, T, M and N.' },
  { key: 'bodyPosition', label: 'Height on body', description: 'Vertical position against forehead, nose, chin and chest. Needs a visible face.' },
];

export const DEFAULT_MATCH_FEATURES = Object.fromEntries(
  MATCH_FEATURE_LIST.map((feature) => [feature.key, true])
);

export const readMatchFeatures = (stored) => Object.fromEntries(
  MATCH_FEATURE_LIST.map((feature) => [feature.key, stored?.[feature.key] !== false])
);

export const TUTORIAL_STEPS = [
  { selector: '[data-tour="menu"]', title: 'Choose your deck', body: 'Open the menu to switch sets, create custom cards, view stats, or change settings.' },
  { selector: '[data-tour="phase"]', title: 'Build your baseline', body: 'Start in Baseline Setup. Record each sign once so matching is tuned to your hand.' },
  { selector: '[data-tour="camera"]', title: 'Use the camera view', body: 'Keep your signing hand clearly visible. The live hand landmarks show when the camera can see you.' },
  { selector: '[data-tour="action"]', title: 'Record or check', body: 'In Baseline Setup this button records your reference sign. In Practice Mode it becomes Check My Sign and scores your attempt against that reference.' },
  { selector: '[data-tour="more"]', title: 'Switch modes anytime', body: 'This menu moves you between Baseline Setup and Practice Mode, and holds mirroring, re-recording, and card navigation. Open it when you are ready to practice.' },
];
