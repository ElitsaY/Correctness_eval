// Minimal SVG charts following the site's mark specs: <=24px bars with a 4px
// rounded data end, hairline grid, value labels at the bar tip, hit targets
// larger than the marks, and a tooltip that only ever uses textContent.

const NS = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs = {}, parent) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    // CSS variables go through style so theme switches repaint the mark.
    if ((k === "fill" || k === "stroke") && String(v).startsWith("var(")) el.style[k] = v;
    else el.setAttribute(k, v);
  }
  if (parent) parent.appendChild(el);
  return el;
}

// ---- Tooltip --------------------------------------------------------------

let tip;
function tooltipEl() {
  if (!tip) {
    tip = document.createElement("div");
    tip.className = "tooltip";
    tip.setAttribute("role", "status");
    document.body.appendChild(tip);
  }
  return tip;
}

/** rows: [{value, label, color?}] - value leads, label follows. */
export function showTooltip(event, title, rows) {
  const t = tooltipEl();
  t.replaceChildren();
  if (title) {
    const h = document.createElement("div");
    h.className = "tl";
    h.textContent = title;
    t.appendChild(h);
  }
  for (const r of rows) {
    const line = document.createElement("div");
    if (r.color) {
      const k = document.createElement("span");
      k.className = "tk";
      k.style.background = r.color;
      line.appendChild(k);
    }
    const v = document.createElement("span");
    v.className = "tv";
    v.textContent = r.value;
    line.appendChild(v);
    if (r.label) {
      const l = document.createElement("span");
      l.className = "tl";
      l.textContent = " " + r.label;
      line.appendChild(l);
    }
    t.appendChild(line);
  }
  const rect = event.currentTarget?.getBoundingClientRect?.();
  const x = event.clientX ?? (rect ? rect.left + rect.width / 2 : 0);
  const y = event.clientY ?? (rect ? rect.top : 0);
  t.classList.add("show");
  const w = t.offsetWidth, h = t.offsetHeight;
  t.style.left = Math.min(window.innerWidth - w - 8, Math.max(8, x + 12)) + "px";
  t.style.top = (y - h - 12 < 8 ? y + 16 : y - h - 12) + "px";
}

export function hideTooltip() {
  tooltipEl().classList.remove("show");
}

function attachHover(hit, mark, title, rows) {
  const on = (e) => { mark?.classList.add("active"); showTooltip(e, title, rows()); };
  const off = () => { mark?.classList.remove("active"); hideTooltip(); };
  hit.addEventListener("pointermove", on);
  hit.addEventListener("pointerleave", off);
  hit.setAttribute("tabindex", "0");
  hit.addEventListener("focus", on);
  hit.addEventListener("blur", off);
}

// ---- Responsive rendering --------------------------------------------------

const specs = new WeakMap();
const observer = new ResizeObserver((entries) => {
  for (const entry of entries) {
    const spec = specs.get(entry.target);
    if (spec && Math.abs(entry.contentRect.width - spec.lastWidth) > 4) spec.render();
  }
});

function mount(el, draw) {
  const render = () => {
    const width = Math.max(280, el.clientWidth || 640);
    spec.lastWidth = width;
    el.replaceChildren();
    draw(width);
  };
  const spec = { render, lastWidth: 0 };
  specs.set(el, spec);
  observer.observe(el);
  render();
}

function roundedBarPath(x0, x1, y, h, r = 4) {
  const w = x1 - x0;
  if (w <= 0) return "";
  const rr = Math.min(r, w, h / 2);
  return `M${x0},${y}H${x1 - rr}Q${x1},${y} ${x1},${y + rr}V${y + h - rr}Q${x1},${y + h} ${x1 - rr},${y + h}H${x0}Z`;
}

function xAxis(svg, x, ticks, top, bottom, left, right, format) {
  const g = svgEl("g", { class: "grid" }, svg);
  const axis = svgEl("g", { class: "tick" }, svg);
  // Thin the ticks on narrow charts so labels never collide (>= ~40px apart).
  const spacing = ticks.length > 1 ? Math.abs(x(ticks[1]) - x(ticks[0])) : Infinity;
  const every = Math.max(1, Math.ceil(40 / spacing));
  ticks = ticks.filter((_, i) => i % every === 0);
  for (const t of ticks) {
    const px = x(t);
    svgEl("line", { x1: px, x2: px, y1: top, y2: bottom }, g);
    const label = svgEl("text", { x: px, y: bottom + 16, "text-anchor": "middle" }, axis);
    label.textContent = format(t);
  }
  svgEl("line", { class: "baseline", x1: left, x2: left, y1: top, y2: bottom }, svg);
  return right;
}

function linear(domain, range) {
  const [d0, d1] = domain, [r0, r1] = range;
  return (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
}

// ---- Horizontal bars (optionally with CI whiskers) ----------------------------

/**
 * rows: [{label, value, lo?, hi?, color, note?}]
 * opts: {domain:[min,max], ticks:[], format(v), valueFormat(v), labelWidth, refLine:{value,label}}
 */
export function hbars(el, rows, opts = {}) {
  const {
    domain = [0, 1], ticks = [0, 0.25, 0.5, 0.75, 1],
    format = (v) => v.toFixed(2), valueFormat = format, labelWidth = 150, refLine,
    rowHeight = 30, barHeight = 16, tooltipTitle = (r) => r.label, tooltipRows,
  } = opts;
  mount(el, (width) => {
    const lw = Math.min(labelWidth, width * 0.38);
    const left = lw, right = width - 56, top = 8;
    const height = top + rows.length * rowHeight + 28;
    const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, role: "img" }, el);
    const x = linear(domain, [left, right]);
    const bottom = top + rows.length * rowHeight;
    xAxis(svg, x, ticks, top, bottom, left, right, format);
    if (refLine) {
      svgEl("line", { class: "ref-line", x1: x(refLine.value), x2: x(refLine.value), y1: top - 4, y2: bottom }, svg);
      const t = svgEl("text", { x: x(refLine.value) + 4, y: top + 6, class: "small" }, svg);
      t.textContent = refLine.label;
      t.style.fontSize = "11px";
    }
    rows.forEach((r, i) => {
      const y = top + i * rowHeight + (rowHeight - barHeight) / 2;
      const label = svgEl("text", { class: "row-label", x: left - 10, y: y + barHeight / 2 + 4, "text-anchor": "end" }, svg);
      label.textContent = r.label;
      if (r.bold) label.style.fontWeight = "700";
      const x0 = x(Math.max(domain[0], 0)), x1 = x(r.value);
      const mark = svgEl("path", { class: "mark", d: roundedBarPath(x0, x1, y, barHeight), fill: r.color }, svg);
      let end = x1;
      if (r.lo != null && r.hi != null) {
        const cy = y + barHeight / 2;
        svgEl("line", { class: "whisker", x1: x(r.lo), x2: x(r.hi), y1: cy, y2: cy }, svg);
        for (const v of [r.lo, r.hi]) svgEl("line", { class: "whisker", x1: x(v), x2: x(v), y1: cy - 4, y2: cy + 4 }, svg);
        end = Math.max(end, x(r.hi));
      }
      const vl = svgEl("text", { class: "value-label", x: end + 6, y: y + barHeight / 2 + 4 }, svg);
      vl.textContent = valueFormat(r.value);
      const hit = svgEl("rect", { class: "hit", x: 0, y: top + i * rowHeight, width, height: rowHeight }, svg);
      svg.insertBefore(hit, mark);
      attachHover(hit, mark, tooltipTitle(r), () =>
        tooltipRows ? tooltipRows(r) : [{ value: valueFormat(r.value), color: r.color }]);
    });
  });
}

// ---- Distribution rows (box: p25-p75, whisker: p5-p95, tick: median, dot: mean) ----

export function distributionRows(el, rows, opts = {}) {
  const { domain = [0, 1], ticks = [0, 0.25, 0.5, 0.75, 1], format = (v) => v.toFixed(2), labelWidth = 170, rowHeight = 30 } = opts;
  mount(el, (width) => {
    const lw = Math.min(labelWidth, width * 0.4);
    const left = lw, right = width - 16, top = 8;
    const bottom = top + rows.length * rowHeight;
    const svg = svgEl("svg", { viewBox: `0 0 ${width} ${bottom + 28}`, role: "img" }, el);
    const x = linear(domain, [left, right]);
    xAxis(svg, x, ticks, top, bottom, left, right, format);
    rows.forEach((r, i) => {
      const cy = top + i * rowHeight + rowHeight / 2;
      const label = svgEl("text", { class: "row-label", x: left - 10, y: cy + 4, "text-anchor": "end" }, svg);
      label.textContent = r.label;
      const g = svgEl("g", { class: "mark" }, svg);
      svgEl("line", { x1: x(r.p05), x2: x(r.p95), y1: cy, y2: cy, stroke: r.color, "stroke-width": 2, "stroke-linecap": "round" }, g);
      const bw = Math.max(2, x(r.p75) - x(r.p25));
      svgEl("rect", { x: x(r.p25), y: cy - 7, width: bw, height: 14, rx: 4, fill: r.color, "fill-opacity": 0.35, stroke: r.color, "stroke-width": 1.5 }, g);
      svgEl("line", { x1: x(r.median), x2: x(r.median), y1: cy - 7, y2: cy + 7, stroke: r.color, "stroke-width": 2.5 }, g);
      svgEl("circle", { cx: x(r.mean), cy, r: 4.5, fill: "var(--surface)", stroke: "var(--ink)", "stroke-width": 1.5 }, g);
      const hit = svgEl("rect", { class: "hit", x: 0, y: cy - rowHeight / 2, width, height: rowHeight }, svg);
      attachHover(hit, g, r.label, () => [
        { value: format(r.mean), label: "mean" },
        { value: format(r.median), label: "median" },
        { value: `${format(r.p25)}–${format(r.p75)}`, label: "middle 50%" },
        { value: `${format(r.p05)}–${format(r.p95)}`, label: "middle 90%" },
        { value: r.n.toLocaleString(), label: "answers" },
      ]);
    });
  });
}

// ---- Dot rows: several series per category on one axis --------------------------

/** rows: [{label, points:[{series, value, color, note?}]}], series: [{name,color}] */
export function dotRows(el, rows, opts = {}) {
  const { domain = [0, 1], ticks = [0, 0.25, 0.5, 0.75, 1], format = (v) => v.toFixed(2), labelWidth = 170, rowHeight = 30 } = opts;
  mount(el, (width) => {
    const lw = Math.min(labelWidth, width * 0.4);
    const left = lw, right = width - 16, top = 8;
    const bottom = top + rows.length * rowHeight;
    const svg = svgEl("svg", { viewBox: `0 0 ${width} ${bottom + 28}`, role: "img" }, el);
    const x = linear(domain, [left, right]);
    xAxis(svg, x, ticks, top, bottom, left, right, format);
    rows.forEach((r, i) => {
      const cy = top + i * rowHeight + rowHeight / 2;
      const label = svgEl("text", { class: "row-label", x: left - 10, y: cy + 4, "text-anchor": "end" }, svg);
      label.textContent = r.label;
      const vals = r.points.map((p) => p.value);
      svgEl("line", { x1: x(Math.min(...vals)), x2: x(Math.max(...vals)), y1: cy, y2: cy, stroke: "var(--axis)", "stroke-width": 2 }, svg);
      const g = svgEl("g", { class: "mark" }, svg);
      for (const p of r.points) {
        svgEl("circle", { cx: x(p.value), cy, r: 5.5, fill: p.color, stroke: "var(--surface)", "stroke-width": 2 }, g);
      }
      const hit = svgEl("rect", { class: "hit", x: 0, y: cy - rowHeight / 2, width, height: rowHeight }, svg);
      attachHover(hit, g, r.label, () =>
        r.points.map((p) => ({ value: format(p.value), label: p.series + (p.note ? ` (${p.note})` : ""), color: p.color })));
    });
  });
}

// ---- Legend -----------------------------------------------------------------

export function legend(el, items, shape = "rect") {
  el.replaceChildren();
  el.classList.add("legend");
  for (const it of items) {
    const span = document.createElement("span");
    const key = document.createElement("i");
    key.className = "key" + (shape === "dot" ? " dot" : "");
    key.style.background = it.color;
    span.appendChild(key);
    span.appendChild(document.createTextNode(it.name));
    el.appendChild(span);
  }
}
