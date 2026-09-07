// Red at low scores through amber to green at high scores; each bar is one solid step.
export function proficiencyBarClass(value) {
  if (value >= 0.8) return 'bg-emerald-500';
  if (value >= 0.65) return 'bg-lime-500';
  if (value >= 0.5) return 'bg-amber-400';
  if (value >= 0.3) return 'bg-orange-500';
  return 'bg-rose-500';
}

export function proficiencyTextClass(value) {
  if (value >= 0.8) return 'text-emerald-500';
  if (value >= 0.65) return 'text-lime-600';
  if (value >= 0.5) return 'text-amber-500';
  if (value >= 0.3) return 'text-orange-500';
  return 'text-rose-500';
}

/**
 * Anki-style weighted random: words with lower recent accuracy get higher probability.
 */
export function selectNextRandomIndex(currentIdx, words, historyEntries) {
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
