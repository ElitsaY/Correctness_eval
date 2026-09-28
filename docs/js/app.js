/* ===== CAP site interactivity ===== */
import { evaluateMetric, roundScores } from "./stats.js";
import { hbars, distributionRows, dotRows, legend } from "./charts.js";

/* ---------- constants ---------- */

// CAP carries the primary accent; baselines share one muted grey (identity comes from their labels).
const METRICS = {
  cap: { name: "CAP", color: "var(--m-cap)" },
  bleu: { name: "BLEU", color: "var(--m-base)" },
  rouge_l: { name: "ROUGE-L", color: "var(--m-base)" },
  meteor: { name: "METEOR", color: "var(--m-base)" },
  bertscore: { name: "BERTScore", color: "var(--m-base)" },
  comet: { name: "COMET", color: "var(--m-base)" },
};
const YOURS_COLOR = "var(--m-yours)";
const MODEL_COLORS = ["var(--c-blue)", "var(--c-terra)", "var(--c-teal)"];
const LABEL_SOURCE_COLORS = ["var(--c-terra)", "var(--c-blue)"];

const LABEL_NAMES = {
  exact: "Exact",
  equivalent: "Equivalent",
  alternative_correct: "Alternative correct",
  overinclusive_valid: "Overinclusive valid",
  partial: "Partial",
  overinclusive_invalid: "Overinclusive invalid",
  invalid: "Invalid",
  contradictory: "Contradictory",
};
const LABELS = Object.keys(LABEL_NAMES);

const state = { paper: null, results: null, scores: null, rows: null, yours: null };

/* ---------- helpers ---------- */

const $ = (id) => document.getElementById(id);
const pct = (v, d = 2) => (v == null || Number.isNaN(v) ? "–" : (100 * v).toFixed(d));
const sevColor = (label) => `var(--sev-${state.results.severity[label]})`;
const metricName = (m) => METRICS[m]?.name ?? m;

function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v == null) continue;
    if (k === "text") node.textContent = v;
    else if (k === "class") node.className = v;
    else if (k === "style") for (const [p, val] of Object.entries(v)) node.style.setProperty(p, val);
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of [].concat(children)) if (c != null) node.append(c);
  return node;
}

function table(head, rows, numericCols = []) {
  return el("table", { class: "summary" }, [
    el("thead", {}, el("tr", {}, head.map((h, i) => el("th", { class: numericCols.includes(i) ? "num" : null, text: h })))),
    el("tbody", {}, rows),
  ]);
}

/** Pill toggle group; calls onPick(key) and marks the active pill. */
function pills(box, items, current, onPick) {
  box.replaceChildren(...items.map(([key, label]) => el("button", {
    type: "button", class: "strat-btn" + (key === current ? " active" : ""), text: label, "data-key": key,
    onclick: (e) => {
      for (const b of box.children) b.classList.toggle("active", b === e.currentTarget);
      onPick(key);
    },
  })));
}

function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32);
}

/* ---------- data ---------- */

async function loadJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Could not load ${path} (${res.status})`);
  return res.json();
}

let rowsPromise;
function loadRows() {
  rowsPromise ||= loadJSON("data/rows.json").then((r) => (state.rows = r));
  return rowsPromise;
}

function splitIndex(split) {
  const s = state.scores.split;
  return s.map((v, i) => (split === "all" || v === split ? i : -1)).filter((i) => i >= 0);
}
const pick = (column, idx) => idx.map((i) => column[i]);

/* ---------- hero ---------- */

function initHero() {
  const svg = $("hero-art");
  const NS = "http://www.w3.org/2000/svg";
  const r = rng(7);
  const sev = [6, 6, 6, 5, 4, 2, 1, 0];
  // Faint pairs of statements joined by a line: the premise/hypothesis motif.
  for (let i = 0; i < 46; i++) {
    const x = r() * 1200, y = r() * 300, dx = 30 + r() * 60, dy = (r() - 0.5) * 40;
    const c = `var(--sev-${sev[Math.floor(r() * sev.length)]})`;
    const line = document.createElementNS(NS, "line");
    Object.entries({ x1: x, y1: y, x2: x + dx, y2: y + dy, class: "ln" }).forEach(([k, v]) => line.setAttribute(k, v));
    svg.append(line);
    for (const [cx, cy] of [[x, y], [x + dx, y + dy]]) {
      const dot = document.createElementNS(NS, "circle");
      Object.entries({ cx, cy, r: 2 + r() * 2, class: "pt" }).forEach(([k, v]) => dot.setAttribute(k, v));
      dot.style.fill = c;
      svg.append(dot);
    }
  }
}

function renderAuthors() {
  const { authors, affiliations } = state.paper;
  const a = $("authors");
  authors.forEach((au, i) => {
    a.append(document.createTextNode(au.name));
    a.append(el("sup", { text: au.affiliations.join(",") + (au.equal ? "*" : "") }));
    if (i < authors.length - 1) a.append(document.createTextNode(", "));
  });
  $("affiliations").textContent = affiliations.map((x, i) => `${i + 1} ${x}`).join(" · ") + " · *Equal contribution";
}

/* ---------- taxonomy ---------- */

function renderTaxonomy() {
  const relations = ["≈", "≈", "≥", ">", ">", ">", "≥"]; // Eq. (1)
  $("ordering").replaceChildren(...LABELS.flatMap((l, i) => [
    el("span", { class: "label-pill", style: { "--sev": sevColor(l) } }, [el("i"), LABEL_NAMES[l]]),
    relations[i] ? el("span", { class: "rel", text: relations[i] }) : null,
  ]).filter(Boolean));

  $("taxonomy-cards").replaceChildren(...LABELS.map((label, i) => {
    const ex = state.results.examples[label];
    return el("article", { class: "tax-card", style: { "--sev": sevColor(label) } }, [
      el("h4", {}, [el("span", { class: "rank", text: String(i + 1) }), LABEL_NAMES[label]]),
      el("p", { class: "def", text: state.paper.label_definitions[label] }),
      el("dl", {}, [
        el("dt", { text: "Q" }), el("dd", { text: ex.question }),
        el("dt", { text: "Gold" }), el("dd", { text: ex.gold_answer }),
        el("dt", { text: "Answer" }), el("dd", { text: ex.answer }),
      ]),
    ]);
  }));
}

/* ---------- results: headline ---------- */

/** Reproduced rows (Python bootstrap CIs) plus COMET as reported in the paper. */
function headlineRows() {
  const rows = state.results.correlations.map((c) => {
    const mono = state.results.monotonicity.find((m) => m.metric === c.metric);
    return {
      metric: c.metric, source: "reproduced",
      spearman: c.spearman, spearman_ci: [c.spearman_ci_low, c.spearman_ci_high],
      kendall: c.kendall, kendall_ci: [c.kendall_ci_low, c.kendall_ci_high],
      pairwise_accuracy: c.pairwise_accuracy, pairwise_accuracy_ci: [c.pairwise_accuracy_ci_low, c.pairwise_accuracy_ci_high],
      violations: mono.mean_violations, class_pairs: mono.class_pairs,
    };
  });
  const comet = state.paper.table2.rows.find((r) => r.metric === "comet");
  rows.push({
    metric: "comet", source: "reported",
    spearman: comet.spearman / 100, spearman_ci: comet.spearman_ci.map((v) => v / 100),
    kendall: comet.kendall / 100, kendall_ci: comet.kendall_ci.map((v) => v / 100),
    pairwise_accuracy: comet.pairwise_accuracy / 100, pairwise_accuracy_ci: [null, null],
    violations: comet.violations, class_pairs: 25,
  });
  return rows;
}

const STAT_NAMES = { spearman: "Spearman ρ", kendall: "Kendall τ", pairwise_accuracy: "pairwise accuracy" };

function renderHeadline(stat = "spearman") {
  const rows = headlineRows().sort((a, b) => b[stat] - a[stat]);
  const isAcc = stat === "pairwise_accuracy";
  hbars($("headline-chart"), rows.map((r) => ({
    label: metricName(r.metric) + (r.source === "reported" ? " (reported)" : ""),
    value: r[stat], lo: r[`${stat}_ci`]?.[0], hi: r[`${stat}_ci`]?.[1],
    color: METRICS[r.metric].color, bold: r.metric === "cap", row: r,
  })), {
    domain: isAcc ? [0.4, 0.85] : [0, 0.7],
    ticks: isAcc ? [0.4, 0.5, 0.6, 0.7, 0.8] : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    format: (v) => (100 * v).toFixed(0), valueFormat: (v) => (100 * v).toFixed(1),
    refLine: isAcc ? { value: 0.5, label: "chance" } : null,
    tooltipRows: (r) => [
      { value: pct(r.value), label: STAT_NAMES[stat], color: r.color },
      ...(r.lo != null ? [{ value: `${pct(r.lo)}–${pct(r.hi)}`, label: "95% CI" }] : []),
      { value: `${r.row.violations}/${r.row.class_pairs}`, label: "class pairs out of order" },
    ],
  });
  $("headline-note").textContent =
    `Test split, ${state.results.rows.toLocaleString()} scored examples, ×100, whiskers: 95% bootstrap intervals (${state.results.n_bootstrap.toLocaleString()} resamples). ` +
    "COMET is as reported in the paper; the rest are reproduced from the released scores.";
}

function headlineTable() {
  const paper = Object.fromEntries(state.paper.table2.rows.map((r) => [r.metric, r]));
  const ci = (range) => (range?.[0] == null ? "" : `[${pct(range[0])}, ${pct(range[1])}]`);
  const rows = headlineRows().sort((a, b) => b.spearman - a.spearman).map((r) => el("tr", { class: r.metric === "cap" ? "highlight" : null }, [
    el("td", {}, [el("span", { class: "swatch", style: { background: METRICS[r.metric].color } }), metricName(r.metric)]),
    el("td", { class: "num" }, [pct(r.spearman), el("span", { class: "ci", text: ci(r.spearman_ci) })]),
    el("td", { class: "num" }, [pct(r.kendall), el("span", { class: "ci", text: ci(r.kendall_ci) })]),
    el("td", { class: "num" }, [pct(r.pairwise_accuracy), el("span", { class: "ci", text: ci(r.pairwise_accuracy_ci) })]),
    el("td", { class: "num", text: `${r.violations}/${r.class_pairs}` }),
    el("td", { class: "num muted", text: `${paper[r.metric].spearman.toFixed(2)} / ${paper[r.metric].kendall.toFixed(2)} / ${paper[r.metric].pairwise_accuracy.toFixed(2)}` }),
    el("td", {}, el("span", { class: "tag-pill" + (r.source === "reproduced" ? " good" : ""), text: r.source })),
  ]));
  $("headline-table").replaceChildren(table(["Metric", "Spearman ρ", "Kendall τ", "Pairwise acc.", "Out of order", "Paper ρ / τ / acc.", "Source"], rows, [1, 2, 3, 4, 5]));
}

/* ---------- results: distributions ---------- */

function quantile(sorted, q) {
  const pos = (sorted.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

function classSummary(scores, labels) {
  const by = {};
  labels.forEach((l, i) => (by[l] ||= []).push(scores[i]));
  return LABELS.filter((l) => by[l]).map((l) => {
    const s = Float64Array.from(by[l]).sort();
    return {
      label: l, n: s.length, mean: s.reduce((a, b) => a + b, 0) / s.length,
      p05: quantile(s, 0.05), p25: quantile(s, 0.25), median: quantile(s, 0.5), p75: quantile(s, 0.75), p95: quantile(s, 0.95),
    };
  });
}

function renderDistribution(metric) {
  const idx = splitIndex("test");
  const scores = pick(state.scores[metric], idx), labels = pick(state.scores.label, idx);
  const summary = classSummary(scores, labels);
  distributionRows($("dist-chart"), summary.map((s) => ({ ...s, label: LABEL_NAMES[s.label], color: sevColor(s.label) })));
  const res = evaluateMetric(scores, labels, state.results.severity, state.results.hard_pairs);
  const bad = res.classPairs.filter((p) => p.meanViolation);
  $("dist-violations").textContent = bad.length
    ? `${metricName(metric)} puts ${bad.length} of ${res.classPairs.length} class pairs out of order (the less correct class has the higher mean): ` +
      bad.map((p) => `${LABEL_NAMES[p.better]} < ${LABEL_NAMES[p.worse]}`).join("; ") + "."
    : `${metricName(metric)} keeps every class pair in order.`;
  $("dist-table").replaceChildren(table(["Class", "n", "Mean", "Median", "P25", "P75", "P5", "P95"], summary.map((s) => el("tr", {}, [
    el("td", {}, el("span", { class: "label-pill", style: { "--sev": sevColor(s.label) } }, [el("i"), LABEL_NAMES[s.label]])),
    ...[s.n, s.mean, s.median, s.p25, s.p75, s.p05, s.p95].map((v, i) => el("td", { class: "num", text: i ? v.toFixed(3) : v.toLocaleString() })),
  ])), [1, 2, 3, 4, 5, 6, 7]));
}

/* ---------- results: hard pairs and AUC ---------- */

const DIVERGING_BINS = [
  { max: 0.35, fill: "var(--div-neg-2)", label: "< 35" },
  { max: 0.45, fill: "var(--div-neg-1)", label: "35–45" },
  { max: 0.55, fill: "var(--div-mid)", label: "45–55" },
  { max: 0.65, fill: "var(--div-pos-1)", label: "55–65" },
  { max: Infinity, fill: "var(--div-pos-2)", label: "≥ 65" },
];
const binFill = (v) => DIVERGING_BINS.find((b) => v < b.max).fill;

function renderHardPairs() {
  const pairs = state.results.hard_pairs;
  const byMetric = {};
  for (const p of state.results.class_pairs) if (p.hard_pair) (byMetric[p.metric] ||= {})[`${p.better}>${p.worse}`] = p.accuracy;
  const comet = state.paper.table4.rows.find((r) => r.metric === "comet");
  byMetric.comet = Object.fromEntries(state.paper.table4.pairs.map(([b, w], i) => [`${b}>${w}`, comet.values[i] / 100]));
  const metrics = Object.keys(byMetric).sort((a, b) => (a === "cap") - (b === "cap"));
  const rows = metrics.map((m) => el("tr", { class: m === "cap" ? "highlight" : null }, [
    el("td", { text: metricName(m) + (m === "comet" ? " (reported)" : "") }),
    ...pairs.map(([b, w]) => {
      const v = byMetric[m][`${b}>${w}`];
      return el("td", { class: "heat", style: { background: binFill(v) }, title: `${metricName(m)}: ${pct(v)}% of pairs ordered correctly`, text: pct(v, 1) });
    }),
  ]));
  $("hard-table").replaceChildren(table(["Metric", ...pairs.map(([b, w]) => `${LABEL_NAMES[b]} > ${LABEL_NAMES[w]}`)], rows));
  $("hard-legend").replaceChildren(el("span", { text: "Accuracy:" }),
    ...DIVERGING_BINS.map((b) => el("span", {}, [el("i", { class: "key", style: { background: b.fill, border: "1px solid var(--line)" } }), b.label])));
}

function renderAuc(metric) {
  const sev = state.results.severity;
  const pairs = Object.fromEntries(state.results.class_pairs.filter((p) => p.metric === metric).map((p) => [`${p.better}>${p.worse}`, p]));
  const rows = LABELS.slice(0, -1).map((b, i) => el("tr", {}, [
    el("td", {}, el("span", { class: "label-pill", style: { "--sev": sevColor(b) } }, [el("i"), LABEL_NAMES[b]])),
    ...LABELS.slice(1).map((w, j) => {
      if (j < i) return el("td", {});
      const p = pairs[`${b}>${w}`];
      if (!p) return el("td", { class: "heat muted", text: sev[b] === sev[w] ? "tie" : "" });
      return el("td", { class: "heat", style: { background: binFill(p.auc) }, title: `${LABEL_NAMES[b]} vs ${LABEL_NAMES[w]}: AUC ${p.auc.toFixed(3)}`, text: p.auc.toFixed(2) });
    }),
  ]));
  $("auc-table").replaceChildren(table(["More correct ↓ / less correct →", ...LABELS.slice(1).map((l) => LABEL_NAMES[l])], rows));
}

/* ---------- LLM answers and human labels ---------- */

function renderLlms() {
  const { models, mean_cap, counts, violations } = state.paper.llm_outputs;
  legend($("llm-legend"), models.map((m, i) => ({ name: m, color: MODEL_COLORS[i] })), "dot");
  dotRows($("llm-chart"), LABELS.map((l) => ({
    label: LABEL_NAMES[l],
    points: models.map((m, i) => ({ series: m, value: mean_cap[l][i] / 100, color: MODEL_COLORS[i], note: `n = ${counts[l][i]}` })),
  })));
  $("llm-note").textContent = `Class pairs out of order: ${models.map((m, i) => `${m} ${violations[i]}/25`).join(", ")}. ` +
    "Classes with few answers (e.g. 6 overinclusive-invalid answers from GPT-4o) have noisy means.";
  $("llm-table").replaceChildren(table(["Class", ...models.map((m) => `${m} (n)`)], LABELS.map((l) => el("tr", {}, [
    el("td", { text: LABEL_NAMES[l] }),
    ...models.map((_, i) => el("td", { class: "num", text: `${mean_cap[l][i].toFixed(2)} (${counts[l][i]})` })),
  ])), [1, 2, 3]));

  const h = state.paper.human_labels;
  const series = ["Synthetic labels", "Human labels"];
  legend($("human-legend"), series.map((s, i) => ({ name: s, color: LABEL_SOURCE_COLORS[i] })), "dot");
  dotRows($("human-chart"), LABELS.map((l) => ({
    label: LABEL_NAMES[l],
    points: series.map((s, i) => ({ series: s, value: h.mean_cap[l][i] / 100, color: LABEL_SOURCE_COLORS[i] })),
  })));
}

/* ---------- dataset explorer ---------- */

const explorer = { page: 0, perPage: 10, open: null };

async function renderExplorer() {
  await loadRows();
  const s = state.scores, r = state.rows;
  const q = $("ex-search").value.trim().toLowerCase();
  const label = $("ex-label").value, dataset = $("ex-dataset").value, split = $("ex-split").value;
  const idx = [];
  for (let i = 0; i < s.row_id.length; i++) {
    if (label && s.label[i] !== label) continue;
    if (dataset && s.dataset[i] !== dataset) continue;
    if (split && s.split[i] !== split) continue;
    if (q && !`${r.question[i]} ${r.gold_answer[i]} ${r.answer[i]}`.toLowerCase().includes(q)) continue;
    idx.push(i);
  }

  const pages = Math.max(1, Math.ceil(idx.length / explorer.perPage));
  explorer.page = Math.min(explorer.page, pages - 1);
  const rows = [];
  for (const i of idx.slice(explorer.page * explorer.perPage, (explorer.page + 1) * explorer.perPage)) {
    const toggle = () => { explorer.open = explorer.open === i ? null : i; renderExplorer(); };
    rows.push(el("tr", {
      class: "ex-row", tabindex: "0", "aria-expanded": String(explorer.open === i), onclick: toggle,
      onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); } },
    }, [
      el("td", { class: "num muted", text: String(s.row_id[i]) }),
      el("td", { text: r.question[i] }),
      el("td", { text: r.gold_answer[i] }),
      el("td", { text: r.answer[i] }),
      el("td", {}, el("span", { class: "label-pill", style: { "--sev": sevColor(s.label[i]) } }, [el("i"), LABEL_NAMES[s.label[i]]])),
    ]));
    if (explorer.open === i) {
      rows.push(el("tr", { class: "ex-detail" }, el("td", { colspan: "5" }, el("div", { class: "kv" }, [
        el("b", { text: "Gold statement" }), el("span", { text: r.premise[i] }),
        el("b", { text: "Answer statement" }), el("span", { text: r.hypothesis[i] }),
        el("b", { text: "Source" }), el("span", { text: `${s.dataset[i]} · ${s.split[i]} split` }),
      ]))));
    }
  }
  $("ex-table").replaceChildren(table(["#", "Question", "Gold", "Answer", "Class"], rows, [0]));

  const prev = el("button", { class: "pill-btn", type: "button", text: "← Previous", onclick: () => { explorer.page--; renderExplorer(); } });
  const next = el("button", { class: "pill-btn", type: "button", text: "Next →", onclick: () => { explorer.page++; renderExplorer(); } });
  prev.disabled = explorer.page === 0;
  next.disabled = explorer.page >= pages - 1;
  $("ex-pager").replaceChildren(el("span", { text: `${idx.length.toLocaleString()} examples · page ${explorer.page + 1} of ${pages}` }), prev, next);
}

/* ---------- leaderboard ---------- */

function leaderboardEntries() {
  const hardMean = (metric) => {
    const acc = state.results.class_pairs.filter((p) => p.metric === metric && p.hard_pair).map((p) => p.accuracy);
    return acc.reduce((a, b) => a + b, 0) / acc.length;
  };
  const entries = headlineRows().map((r) => ({
    key: r.metric, name: metricName(r.metric), color: METRICS[r.metric].color,
    spearman: r.spearman, kendall: r.kendall, pairwise_accuracy: r.pairwise_accuracy, violations: r.violations,
    hard: r.metric === "comet"
      ? state.paper.table4.rows.find((x) => x.metric === "comet").values.reduce((a, b) => a + b, 0) / 400
      : hardMean(r.metric),
    source: r.source === "reported" ? "reported in paper" : "reproduced",
  }));
  if (state.yours?.complete) {
    const res = state.yours.result;
    const hard = res.classPairs.filter((p) => p.hard);
    entries.push({
      key: "yours", name: state.yours.name, color: YOURS_COLOR, spearman: res.spearman, kendall: res.kendall,
      pairwise_accuracy: res.pairwiseAccuracy, violations: res.violations,
      hard: hard.reduce((a, p) => a + p.accuracy, 0) / hard.length, source: "your upload",
    });
  }
  return entries.sort((a, b) => b.spearman - a.spearman);
}

function renderLeaderboard() {
  const rows = leaderboardEntries().map((e, i) => el("tr", { class: e.key === "cap" ? "highlight" : e.key === "yours" ? "yours" : null }, [
    el("td", { class: "num", text: String(i + 1) }),
    el("td", {}, [el("span", { class: "swatch", style: { background: e.color } }), e.name]),
    el("td", { class: "num", text: pct(e.spearman) }),
    el("td", { class: "num", text: pct(e.kendall) }),
    el("td", { class: "num", text: pct(e.pairwise_accuracy) }),
    el("td", { class: "num", text: pct(e.hard) }),
    el("td", { class: "num", text: `${e.violations}/25` }),
    el("td", {}, el("span", { class: "tag-pill" + (e.source === "reproduced" ? " good" : e.key === "yours" ? " yours" : ""), text: e.source })),
  ]));
  $("lb-table").replaceChildren(table(["#", "Evaluator", "Spearman ρ", "Kendall τ", "Pairwise acc.", "Hard pairs", "Out of order", "Source"], rows, [0, 2, 3, 4, 5, 6]));
  $("lb-note").textContent = `Test split (${state.results.rows.toLocaleString()} scored examples), ×100. "Hard pairs" is the mean accuracy over the four hard neighbouring pairs. ` +
    "To add your evaluator, evaluate it below and get in touch.";
}

/* ---------- evaluate yours ---------- */

function parseScoresCsv(text) {
  const lines = text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  if (!lines.length) throw new Error("No data found.");
  const sep = lines[0].includes("\t") ? "\t" : lines[0].includes(";") ? ";" : ",";
  const header = lines[0].split(sep).map((h) => h.trim().replace(/^"|"$/g, "").toLowerCase());
  let idCol = header.indexOf("row_id"), scoreCol = header.indexOf("score"), start = 1;
  if (idCol < 0 || scoreCol < 0) {
    if (header.length === 2 && header.every((h) => h !== "" && !Number.isNaN(Number(h)))) { idCol = 0; scoreCol = 1; start = 0; }
    else throw new Error('Expected a header with "row_id" and "score" columns.');
  }
  const scores = new Map();
  for (let n = start; n < lines.length; n++) {
    const cells = lines[n].split(sep).map((c) => c.trim().replace(/^"|"$/g, ""));
    const id = Number(cells[idCol]), score = Number(cells[scoreCol]);
    if (!Number.isInteger(id)) throw new Error(`Line ${n + 1}: row_id "${cells[idCol]}" is not an integer.`);
    if (!Number.isFinite(score)) throw new Error(`Line ${n + 1}: score "${cells[scoreCol]}" is not a number.`);
    if (scores.has(id)) throw new Error(`Line ${n + 1}: row_id ${id} appears twice.`);
    scores.set(id, score);
  }
  return scores;
}

function rankNormalize(values) {
  const order = Array.from(values.keys()).sort((a, b) => values[a] - values[b]);
  const out = new Float64Array(values.length);
  for (let i = 0; i < order.length; ) {
    let j = i;
    while (j + 1 < order.length && values[order[j + 1]] === values[order[i]]) j++;
    for (let k = i; k <= j; k++) out[order[k]] = (i + j) / 2 / Math.max(1, values.length - 1);
    i = j + 1;
  }
  return out;
}

function stat(n, l, d) {
  return el("div", { class: "stat" }, [el("span", { class: "n", text: n }), el("span", { class: "l", text: l }), d ? el("span", { class: "d", text: d }) : null]);
}

function evaluateYours(text, name) {
  const status = $("yours-status");
  $("yours-result").hidden = false;
  let scores;
  try {
    scores = parseScoresCsv(text);
  } catch (err) {
    status.className = "notice warn";
    status.textContent = err.message;
    $("yours-tiles").replaceChildren();
    return;
  }
  const idx = splitIndex("test");
  const matched = idx.filter((i) => scores.has(state.scores.row_id[i]));
  if (matched.length < 10) {
    status.className = "notice warn";
    status.textContent = `Only ${matched.length} of your row_ids are in the test split. Download the test set above for the right ids.`;
    return;
  }
  const values = roundScores(matched.map((i) => scores.get(state.scores.row_id[i])));
  const labels = pick(state.scores.label, matched);
  const result = evaluateMetric(values, labels, state.results.severity, state.results.hard_pairs);
  const complete = matched.length === idx.length;
  state.yours = { name: name || "My evaluator", result, complete, n: matched.length };

  status.className = "notice" + (complete ? "" : " warn");
  status.textContent = complete
    ? `Scored all ${idx.length.toLocaleString()} test examples. "${state.yours.name}" now appears in the leaderboard above (in your browser only).`
    : `Scored ${matched.length.toLocaleString()} of ${idx.length.toLocaleString()} test examples. Results are shown, but only complete submissions are ranked.`;

  const entries = leaderboardEntries();
  const rank = complete ? entries.findIndex((e) => e.key === "yours") + 1 : null;
  const cap = state.results.correlations.find((c) => c.metric === "cap");
  $("yours-tiles").replaceChildren(
    stat(pct(result.spearman), "Spearman ρ", `CAP ${pct(cap.spearman)}`),
    stat(pct(result.kendall), "Kendall τ", `CAP ${pct(cap.kendall)}`),
    stat(pct(result.pairwiseAccuracy), "Pairwise accuracy", `CAP ${pct(cap.pairwise_accuracy)}`),
    stat(`${result.violations}/${result.classPairs.length}`, "Out of order", "CAP 4/25"),
    stat(rank ? `#${rank}` : "–", "Rank", rank ? `of ${entries.length}` : "needs every test row"),
  );

  const capNorm = rankNormalize(pick(state.scores.cap, matched));
  const yoursNorm = rankNormalize(values);
  const mean = (arr, l) => { let s = 0, n = 0; labels.forEach((x, i) => { if (x === l) { s += arr[i]; n++; } }); return n ? s / n : null; };
  legend($("yours-legend"), [{ name: "CAP", color: METRICS.cap.color }, { name: state.yours.name, color: YOURS_COLOR }], "dot");
  dotRows($("yours-chart"), LABELS.filter((l) => labels.includes(l)).map((l) => ({
    label: LABEL_NAMES[l],
    points: [
      { series: "CAP", value: mean(capNorm, l), color: METRICS.cap.color },
      { series: state.yours.name, value: mean(yoursNorm, l), color: YOURS_COLOR },
    ],
  })));

  renderLeaderboard();
}

function download(filename, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = el("a", { href: url, download: filename });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const csvCell = (v) => (/[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v);

async function downloadTemplate() {
  const btn = $("download-template");
  btn.disabled = true;
  btn.textContent = "Preparing…";
  await loadRows();
  const lines = ["row_id,question,gold_answer,answer"];
  for (const i of splitIndex("test")) {
    lines.push([state.scores.row_id[i], state.rows.question[i], state.rows.gold_answer[i], state.rows.answer[i]].map((v) => csvCell(String(v))).join(","));
  }
  download("CAP-Correctness-test.csv", lines.join("\n") + "\n", "text/csv");
  btn.disabled = false;
  btn.textContent = "Download test set CSV";
}

/* ---------- contact ---------- */

/** Embed the Google Form named in #contact[data-form-url], or say it is coming. */
function initContact() {
  const box = $("contact-form");
  const url = $("contact").dataset.formUrl.trim();
  if (!/^https:\/\/docs\.google\.com\/forms\/|^https:\/\/forms\.gle\//.test(url)) {
    box.replaceChildren(el("div", { class: "notice", text: "The contact form is coming soon." }));
    return;
  }
  // Short forms.gle links cannot be embedded; they are shown as a button instead.
  const embed = url.includes("docs.google.com/forms/") ? url.replace(/\/viewform.*$/, "/viewform?embedded=true") : null;
  box.replaceChildren(
    ...(embed ? [el("div", { class: "form-tile" }, el("iframe", { src: embed, title: "Contact form", loading: "lazy" }))] : []),
    el("p", { class: "form-link" }, el("a", { class: embed ? null : "pill-btn primary", href: url, target: "_blank", rel: "noopener", text: embed ? "Open the form in a new tab ↗" : "Open the contact form ↗" })),
  );
}

/* ---------- wiring ---------- */

function initCopyButtons() {
  for (const b of document.querySelectorAll("[data-copy]")) {
    b.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText($(b.dataset.copy).textContent);
        b.textContent = "Copied";
      } catch (e) {
        b.textContent = "Select and copy";
      }
      setTimeout(() => (b.textContent = "Copy"), 1500);
    });
  }
}

async function main() {
  initHero();
  initCopyButtons();
  initContact();
  [state.paper, state.results, state.scores] = await Promise.all([
    loadJSON("data/paper.json"), loadJSON("data/results.json"), loadJSON("data/scores.json"),
  ]);
  const metricPills = state.scores.metrics.map((m) => [m, metricName(m)]);

  renderAuthors();
  renderTaxonomy();

  renderHeadline();
  headlineTable();
  pills($("stat-toggle"), Object.entries({ spearman: "Spearman ρ", kendall: "Kendall τ", pairwise_accuracy: "Pairwise accuracy" }), "spearman", renderHeadline);

  pills($("dist-metric"), metricPills, "cap", renderDistribution);
  renderDistribution("cap");
  renderHardPairs();
  pills($("auc-metric"), metricPills, "cap", renderAuc);
  renderAuc("cap");

  renderLlms();
  renderLeaderboard();

  for (const l of LABELS) $("ex-label").append(el("option", { value: l, text: LABEL_NAMES[l] }));
  for (const d of [...new Set(state.scores.dataset)].sort()) $("ex-dataset").append(el("option", { value: d, text: d }));
  let searchTimer;
  $("ex-search").addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => { explorer.page = 0; renderExplorer(); }, 200); });
  for (const id of ["ex-label", "ex-dataset", "ex-split"]) $(id).addEventListener("change", () => { explorer.page = 0; renderExplorer(); });
  // The dataset table (3.7 MB of text) loads when it comes into view, unless the URL points below it.
  const target = location.hash && document.querySelector(location.hash);
  const belowExplorer = target && $("explore").compareDocumentPosition(target) & Node.DOCUMENT_POSITION_FOLLOWING;
  if (belowExplorer) await renderExplorer();
  else {
    new IntersectionObserver((entries, obs) => {
      if (entries.some((e) => e.isIntersecting)) { obs.disconnect(); renderExplorer(); }
    }, { rootMargin: "400px" }).observe($("explore"));
  }

  $("download-template").addEventListener("click", downloadTemplate);
  $("yours-file").addEventListener("change", async (e) => {
    const f = e.target.files[0];
    if (f) $("yours-text").value = await f.text();
  });
  $("yours-run").addEventListener("click", () => evaluateYours($("yours-text").value, $("yours-name").value.trim()));
  $("yours-demo").addEventListener("click", () => {
    const idx = splitIndex("test");
    $("yours-name").value = "BLEU (demo)";
    $("yours-text").value = "row_id,score\n" + idx.map((i) => `${state.scores.row_id[i]},${state.scores.bleu[i]}`).join("\n");
    evaluateYours($("yours-text").value, "BLEU (demo)");
  });
  $("yours-download").addEventListener("click", () => {
    const y = state.yours;
    if (!y) return;
    const { spearman, kendall, pairwiseAccuracy, violations, means, classPairs, n } = y.result;
    download("cap-leaderboard-result.json", JSON.stringify({ evaluator: y.name, split: "test", n, spearman, kendall, pairwise_accuracy: pairwiseAccuracy, violations, class_means: means, class_pairs: classPairs }, null, 2), "application/json");
  });

  // Sections above were built after the browser jumped to the URL's #anchor; jump again.
  if (target) target.scrollIntoView({ behavior: "instant" });
}

main().catch((err) => {
  console.error(err);
  document.querySelector(".wrap").prepend(el("div", { class: "notice warn", text: `Could not load the site data: ${err.message}` }));
});
