// Agreement statistics between metric scores and ordinal correctness labels.
// A JavaScript port of src/cap_eval/evaluation.py, using the same conventions:
//   * a pair is comparable when its two severities differ;
//   * a win means the more-correct answer scores strictly higher; score ties
//     are counted separately and never as wins;
//   * scores are rounded to SCORE_DECIMALS first so float noise cannot break ties.
// Runs in the browser and in Node (scripts check it against the Python results).

export const SCORE_DECIMALS = 10;
const SCALE = 10 ** SCORE_DECIMALS;

export function roundScores(values) {
  return Float64Array.from(values, (v) => Math.round(v * SCALE) / SCALE);
}

// Number of values in sorted `arr` that are < x (lower) or <= x (upper).
function lowerBound(arr, x) {
  let lo = 0, hi = arr.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (arr[mid] < x) lo = mid + 1; else hi = mid; }
  return lo;
}
function upperBound(arr, x) {
  let lo = 0, hi = arr.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (arr[mid] <= x) lo = mid + 1; else hi = mid; }
  return lo;
}

/** Count (b, w) pairs with b > w, b == w and b < w. */
export function compareGroups(better, worse) {
  if (!better.length || !worse.length) return { wins: 0, ties: 0, losses: 0, pairs: 0 };
  const sorted = Float64Array.from(worse).sort();
  let wins = 0, ties = 0;
  for (const b of better) {
    const below = lowerBound(sorted, b);
    wins += below;
    ties += upperBound(sorted, b) - below;
  }
  const pairs = better.length * worse.length;
  return { wins, ties, losses: pairs - wins - ties, pairs };
}

/** All pairs of examples with different severity. */
export function pairwiseRanking(scores, severity) {
  const levels = [...new Set(severity)].sort((a, b) => a - b);
  const total = { wins: 0, ties: 0, losses: 0, pairs: 0 };
  const lower = [];
  for (const level of levels) {
    const current = [];
    for (let i = 0; i < scores.length; i++) if (severity[i] === level) current.push(scores[i]);
    if (lower.length) {
      const c = compareGroups(current, lower);
      for (const k in total) total[k] += c[k];
    }
    lower.push(...current);
  }
  return total;
}

function averageRanks(values) {
  const order = Array.from(values.keys()).sort((a, b) => values[a] - values[b]);
  const ranks = new Float64Array(values.length);
  for (let i = 0; i < order.length; ) {
    let j = i;
    while (j + 1 < order.length && values[order[j + 1]] === values[order[i]]) j++;
    const rank = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) ranks[order[k]] = rank;
    i = j + 1;
  }
  return ranks;
}

function pearson(x, y) {
  const n = x.length;
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
  mx /= n; my /= n;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i] - mx, dy = y[i] - my;
    sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
  }
  return sxy / Math.sqrt(sxx * syy);
}

export function spearman(scores, severity) {
  return pearson(averageRanks(scores), averageRanks(severity));
}

function tiedPairs(values) {
  const counts = new Map();
  for (const v of values) counts.set(v, (counts.get(v) || 0) + 1);
  let t = 0;
  for (const c of counts.values()) t += (c * (c - 1)) / 2;
  return t;
}

/** Kendall tau-b. Concordant/discordant pairs are exactly the wins/losses of pairwiseRanking. */
export function kendallTauB(scores, severity, ranking = pairwiseRanking(scores, severity)) {
  const n0 = (scores.length * (scores.length - 1)) / 2;
  const denom = Math.sqrt((n0 - tiedPairs(scores)) * (n0 - tiedPairs(severity)));
  return (ranking.wins - ranking.losses) / denom;
}

/** Every statistic the site reports for one metric. */
export function evaluateMetric(rawScores, labels, severityMap, hardPairs = []) {
  const scores = roundScores(rawScores);
  const severity = labels.map((l) => severityMap[l]);
  const ranking = pairwiseRanking(scores, severity);

  const byLabel = {};
  labels.forEach((l, i) => (byLabel[l] ||= []).push(scores[i]));
  const present = Object.keys(severityMap).filter((l) => byLabel[l]);
  const means = Object.fromEntries(present.map((l) => [l, byLabel[l].reduce((a, b) => a + b, 0) / byLabel[l].length]));

  const hard = new Set(hardPairs.map(([b, w]) => `${b}>${w}`));
  const ordered = [...present].sort((a, b) => severityMap[b] - severityMap[a]);
  const classPairs = [];
  for (let i = 0; i < ordered.length; i++) {
    for (let j = i + 1; j < ordered.length; j++) {
      const [better, worse] = [ordered[i], ordered[j]];
      if (severityMap[better] <= severityMap[worse]) continue;
      const c = compareGroups(byLabel[better], byLabel[worse]);
      classPairs.push({
        better, worse, hard: hard.has(`${better}>${worse}`),
        accuracy: c.wins / c.pairs,
        auc: (c.wins + 0.5 * c.ties) / c.pairs,
        meanViolation: means[worse] > means[better],
      });
    }
  }
  return {
    n: scores.length,
    spearman: spearman(scores, severity),
    kendall: kendallTauB(scores, severity, ranking),
    pairwiseAccuracy: ranking.wins / ranking.pairs,
    ranking,
    means,
    classPairs,
    violations: classPairs.filter((p) => p.meanViolation).length,
  };
}
