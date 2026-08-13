"use strict";

const tracks = {
  coco: {
    scope: "1,000 images · 5,003 captions",
    title: "Image-only Text → Image",
    delta: "+2.89",
    ci: "95% CI [+2.00, +3.92] · paired image-cluster bootstrap",
    metrics: [
      { label: "R@1", baseline: 50.19, trained: 53.09 },
      { label: "R@10", baseline: 89.31, trained: 90.67 },
      { label: "MRR", baseline: 63.51, trained: 65.80 },
    ],
  },
  video: {
    scope: "1,000 videos · one-caption 1K-A",
    title: "Text → Video / Video → Text",
    delta: "+7.90",
    ci: "T2V R@10 95% CI [+5.60, +10.40] · paired video-cluster bootstrap",
    metrics: [
      { label: "T2V R@1", baseline: 30.4, trained: 33.5 },
      { label: "T2V R@10", baseline: 63.1, trained: 71.0 },
      { label: "V2T R@10", baseline: 61.0, trained: 70.8 },
    ],
  },
};

const fixtures = {
  runner: {
    gates: [0.54, 0.46], latency: "0.38 ms · numpy exact",
    rows: [
      ["Open field motion", 0.912, ["#071d1a", "#2ce6c5", "#b9ffe9"]],
      ["Track silhouette", 0.874, ["#17142a", "#6d8cff", "#d6dcff"]],
      ["Sunlit landscape", 0.829, ["#26170c", "#ff8d3a", "#ffe0b8"]],
    ],
  },
  city: {
    gates: [0.47, 0.53], latency: "0.41 ms · numpy exact",
    rows: [
      ["Neon street flow", 0.931, ["#0b1024", "#6d8cff", "#2ce6c5"]],
      ["Urban light trails", 0.886, ["#201026", "#ff4f9a", "#6d8cff"]],
      ["Night intersection", 0.842, ["#10171e", "#2ce6c5", "#ff8d3a"]],
    ],
  },
  ocean: {
    gates: [0.61, 0.39], latency: "0.36 ms · numpy exact",
    rows: [
      ["Calm water vessel", 0.904, ["#081925", "#2d91d1", "#c6f2ff"]],
      ["Blue horizon", 0.861, ["#0c2230", "#2ce6c5", "#8ac9ff"]],
      ["Coastal geometry", 0.817, ["#181e24", "#6d8cff", "#e8f4ff"]],
    ],
  },
};

let mode = "full";

function renderTrack(name) {
  const track = tracks[name];
  document.querySelector("#metric-scope").textContent = track.scope;
  document.querySelector("#metric-title").textContent = track.title;
  document.querySelector("#metric-delta").textContent = track.delta;
  document.querySelector("#metric-ci").textContent = track.ci;
  document.querySelector("#metric-chart").innerHTML = track.metrics.map((metric) => `
    <div class="bar-group">
      <span class="bar-label">${metric.label}</span>
      <div class="bars">
        <div class="bar baseline" style="width:${metric.baseline}%">${metric.baseline.toFixed(2)}</div>
        <div class="bar trained" style="width:${metric.trained}%">${metric.trained.toFixed(2)}</div>
      </div>
    </div>`).join("");
  document.querySelectorAll(".track-tab").forEach((button) => {
    const active = button.dataset.track === name;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", String(active));
  });
}

function modeGates(gates) {
  if (mode === "image") return [1, 0];
  if (mode === "text") return [0, 1];
  return gates;
}

function renderFixture() {
  const fixture = fixtures[document.querySelector("#query").value];
  const gates = modeGates(fixture.gates);
  document.querySelector("#image-gate").textContent = gates[0].toFixed(2);
  document.querySelector("#text-gate").textContent = gates[1].toFixed(2);
  document.querySelector("#image-gate-bar").style.width = `${gates[0] * 100}%`;
  document.querySelector("#text-gate-bar").style.width = `${gates[1] * 100}%`;
  document.querySelector("#latency").textContent = fixture.latency;
  const penalty = mode === "full" ? 0 : mode === "image" ? 0.027 : 0.019;
  document.querySelector("#synthetic-results").innerHTML = fixture.rows.map((row, index) => `
    <article class="synthetic-card">
      <div class="synthetic-art" style="--art-bg:${row[2][0]};--art-a:${row[2][1]};--art-b:${row[2][2]}"></div>
      <div class="synthetic-body">
        <div class="synthetic-meta"><span>#${index + 1}</span><strong>cos ${(row[1] - penalty * index).toFixed(3)}</strong></div>
        <p>${row[0]}</p>
      </div>
    </article>`).join("");
}

document.querySelectorAll(".track-tab").forEach((button) => button.addEventListener("click", () => renderTrack(button.dataset.track)));
document.querySelector("#query").addEventListener("change", renderFixture);
document.querySelectorAll(".mode").forEach((button) => button.addEventListener("click", () => {
  mode = button.dataset.mode;
  document.querySelectorAll(".mode").forEach((item) => item.classList.toggle("is-active", item === button));
  renderFixture();
}));

renderTrack("coco");
renderFixture();
