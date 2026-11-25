"use strict";
const $ = id => document.getElementById(id);
const pct = value => `${(100 * value).toFixed(1)}%`;
let data;
let trial = 0;
let budgetIndex = 0;
const palette = () => {
  const style = getComputedStyle(document.documentElement);
  return Object.fromEntries(["accent", "muted", "line", "warm", "bg", "text"].map(k => [k, style.getPropertyValue(`--${k}`).trim()]));
};

function showChart() {
  const colors = palette();
  const points = data.curves;
  const left = 48, right = 598, top = 22, bottom = 260;
  const x = i => left + i * (right - left) / Math.max(1, points.length - 1);
  const y = score => bottom - score * (bottom - top);
  const line = (x1, y1, x2, y2, color, extra = "") => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" ${extra}/>`;
  let svg = "";
  for (const tick of [0, .25, .5, .75, 1]) {
    svg += line(left, y(tick), right, y(tick), colors.line);
    svg += `<text x="${left - 10}" y="${y(tick) + 4}" text-anchor="end" fill="${colors.muted}" font-size="16">${tick * 100}%</text>`;
  }
  svg += line(left, y(data.chance), right, y(data.chance), colors.muted, 'stroke-dasharray="5 5"');
  svg += line(left, y(data.shuffled_mean), right, y(data.shuffled_mean), colors.warm, 'stroke-dasharray="2 5"');
  points.forEach((p, i) => {
    svg += line(x(i), y(p.min), x(i), y(p.max), colors.accent);
    svg += line(x(i) - 4, y(p.min), x(i) + 4, y(p.min), colors.accent);
    svg += line(x(i) - 4, y(p.max), x(i) + 4, y(p.max), colors.accent);
  });
  svg += `<polyline points="${points.map((p, i) => `${x(i)},${y(p.mean)}`).join(" ")}" fill="none" stroke="${colors.accent}" stroke-width="2.5"/>`;
  for (let i = 0; i < points.length; i++) {
    svg += `<circle cx="${x(i)}" cy="${y(points[i].mean)}" r="${i === budgetIndex ? 7 : 3.5}" fill="${colors.accent}"/>`;
    svg += `<text x="${x(i)}" y="${bottom + 24}" text-anchor="middle" fill="${colors.muted}" font-size="16">${points[i].cells}</text>`;
  }
  svg += `<text x="${(left + right) / 2}" y="306" text-anchor="middle" fill="${colors.muted}" font-size="16">Recorded cells</text>`;
  $("curve").innerHTML = svg;
}

function showSignals(detail) {
  const canvas = $("signals");
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.getBoundingClientRect().width;
  canvas.width = Math.max(1, Math.round(width * ratio));
  canvas.height = 105 * ratio;
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const colors = palette();
  ctx.fillStyle = colors.bg;
  ctx.fillRect(0, 0, width, 105);
  const lookup = new Map(data.unit_ids.map((id, i) => [id, i]));
  const counts = detail.cells.map(id => data.test_counts[trial][lookup.get(id)]);
  // Keep the vertical scale fixed across trials and cell budgets.
  const maximum = data.test_counts.reduce((max, row) => row.reduce((m, value) => Math.max(m, value), max), 1);
  const slot = width / counts.length;
  ctx.fillStyle = colors.accent;
  counts.forEach((count, i) => {
    const height = count / maximum * 85;
    ctx.fillRect(i * slot, 95 - height, Math.max(.5, slot - 1), Math.max(1, height));
  });
  $("signalCaption").textContent = `${counts.reduce((a, b) => a + b, 0)} spikes from ${counts.length} cells. Each bar shows one cell\'s count. The scale stays the same as you change trials or cells.`;
}

function showMatrix(detail) {
  const matrix = $("confusion");
  matrix.replaceChildren();
  matrix.style.gridTemplateColumns = `repeat(${data.classes.length + 1}, minmax(0,1fr))`;
  const cell = (label, className) => {
    const node = document.createElement("div");
    node.textContent = label;
    node.className = className;
    matrix.append(node);
  };
  cell("Shown / predicted", "matrix-head");
  for (const id of data.classes) cell(`#${id}`, "matrix-head");
  data.classes.forEach((id, i) => {
    cell(`#${id}`, "matrix-head");
    detail.confusion[i].forEach((count, j) => cell(count, `matrix-cell ${i === j ? "diagonal" : ""}`));
  });
}

function render() {
  const budget = data.curves[budgetIndex].cells;
  const detail = data.detail[String(budget)];
  const truth = data.test_labels[trial];
  const predicted = detail.predictions[trial];
  $("budgetValue").textContent = budget;
  $("budget").setAttribute("aria-valuetext", `${budget} cells`);
  $("currentScore").textContent = pct(data.curves[budgetIndex].mean);
  $("trial").value = String(trial);
  $("previous").disabled = trial === 0;
  $("next").disabled = trial === data.test_labels.length - 1;
  $("trialMeta").textContent = `${(data.test_starts[trial] / 60).toFixed(1)} minutes into the recording (presentation ${data.test_trial_ids[trial]}).`;
  $("outcome").textContent = truth === predicted ? "Correct" : "Incorrect";
  $("outcome").className = `badge ${truth === predicted ? "correct" : "wrong"}`;
  const gallery = $("gallery");
  gallery.replaceChildren();
  data.classes.forEach((frame, j) => {
    const card = document.createElement("div");
    card.className = `image-card ${frame === truth ? "true-image" : ""}`;
    const image = document.createElement("img");
    image.src = window.NEURAL_IMAGES?.[frame] || `images/${frame}.png`;
    image.alt = `Allen natural scene ${frame}`;
    const label = document.createElement("div");
    label.className = "card-label";
    const name = document.createElement("span"); name.textContent = `Image ${frame}`;
    const probability = document.createElement("span"); probability.textContent = pct(detail.probabilities[trial][j]);
    label.append(name, probability);
    const track = document.createElement("div"); track.className = "bar-track";
    const bar = document.createElement("div"); bar.className = "bar"; bar.style.width = `${100 * detail.probabilities[trial][j]}%`; track.append(bar);
    const tags = document.createElement("div"); tags.className = "tags";
    tags.textContent = [frame === truth ? "Shown" : "", frame === predicted ? "Predicted" : ""].filter(Boolean).join(" / ");
    card.append(image, label, track, tags);
    gallery.append(card);
  });
  showChart(); showMatrix(detail); showSignals(detail);
}

async function start() {
  if (window.NEURAL_RESULTS) {
    data = window.NEURAL_RESULTS;
  } else {
    const response = await fetch("results.json");
    if (!response.ok) throw new Error(`Results could not load (HTTP ${response.status})`);
    data = await response.json();
  }
  if (!data.curves?.length || !data.test_labels?.length) throw new Error("The results file has no test predictions");
  $("session").textContent = data.metadata.session_id;
  $("totalCells").textContent = data.metadata.units;
  $("trainTrials").textContent = data.metadata.train_trials;
  $("testTrials").textContent = data.metadata.test_trials;
  $("budgetMax").textContent = `${data.metadata.units} cells`;
  $("budget").max = data.curves.length - 1;
  budgetIndex = data.curves.length - 1;
  $("budget").value = budgetIndex;
  data.test_labels.forEach((label, i) => {
    const option = document.createElement("option"); option.value = i; option.textContent = `Trial ${i + 1} of ${data.test_labels.length}`; $("trial").append(option);
  });
  for (const row of data.region_results) {
    const tr = document.createElement("tr");
    for (const value of [row.region, row.removed, pct(row.score), `${pct(row.random_mean)} (${pct(row.random_min)}–${pct(row.random_max)})`]) {
      const td = document.createElement("td"); td.textContent = value; tr.append(td);
    }
    $("regions").append(tr);
  }
  $("splitInfo").textContent = `The model trained on blocks ${data.metadata.train_blocks.join(", ")} and was tested on block ${data.metadata.test_block}. Only cells that passed these quality checks were included: ${data.metadata.unit_filter}.`;
  $("controlInfo").textContent = `As a control, we shuffled the training labels within each block. Across ${data.shuffled_scores.length} fits, average test accuracy was ${pct(data.shuffled_mean)}. Random guessing would score ${pct(data.chance)}. Each larger cell selection keeps the cells from the smaller selection.`;
  $("limits").textContent = `The data comes from one mouse, recorded in a single session. The test uses later repeats of ${data.classes.length} pictures used during training. We haven\'t tested the model on new pictures or other mice. Its probabilities haven\'t been checked against how often predictions are correct. Leaving a region out only changes which recordings the model uses.`;
  $("curveNote").textContent = `The line averages scores from ${data.seeds} random cell selections. The bars show the lowest and highest scores across those selections. They aren\'t confidence intervals. The trial viewer uses the first selection.`;
  if (window.NEURAL_RESULTS) {
    for (const [id, content] of [["fullDownload", data], ["summaryDownload", {metadata: data.metadata, curves: data.curves, region_results: data.region_results}]]) {
      $(id).href = URL.createObjectURL(new Blob([JSON.stringify(content, null, 2)], {type:"application/json"}));
    }
  }
  $("status").hidden = true;
  $("experiment").hidden = false;
  $("budget").addEventListener("input", event => { budgetIndex = Number(event.target.value); render(); });
  $("trial").addEventListener("change", event => { trial = Number(event.target.value); render(); });
  $("previous").addEventListener("click", () => { if (trial > 0) trial--; render(); });
  $("next").addEventListener("click", () => { if (trial < data.test_labels.length - 1) trial++; render(); });
  window.addEventListener("resize", () => showSignals(data.detail[String(data.curves[budgetIndex].cells)]));
  render();
}
start().catch(error => { $("status").textContent = `The results couldn\'t load. ${error.message}.`; });
