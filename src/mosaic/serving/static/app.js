const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (char) => ({
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  "\"": "&quot;",
  "'": "&#39;",
})[char]);

const formatPercent = (value) => Number(value).toFixed(3);
const formatPp = (value) => `${value >= 0 ? "+" : ""}${(Number(value) * 100).toFixed(1)} pp`;
const formatCi = (row) => `[${formatPp(row.lower)}, ${formatPp(row.upper)}]`;
const formatBytes = (bytes) => `${(Number(bytes) / 1024).toFixed(0)} KiB`;
const shortHash = (value) => value ? `${String(value).slice(0, 10)}…${String(value).slice(-8)}` : "unavailable";

let videoEvidence = null;

function setView(name, { updateHash = true } = {}) {
  const selected = name === "video" ? "video" : "coco";
  $$("[data-view]").forEach((button) => {
    const active = button.dataset.view === selected;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", String(active));
  });
  $$("[data-view-panel]").forEach((panel) => {
    const active = panel.dataset.viewPanel === selected;
    panel.classList.toggle("is-active", active);
    panel.hidden = !active;
  });
  if (selected !== "video") {
    $$("#video-samples video").forEach((video) => video.pause());
  } else if (!videoEvidence) {
    loadVideoEvidence();
  }
  if (updateHash) window.history.replaceState(null, "", `#${selected}`);
}

async function health() {
  try {
    const state = await fetch("/api/health").then((response) => response.json());
    const videoState = state.video_evidence_ready ? "video evidence ready" : "video evidence unavailable";
    $("#status").classList.add("is-ready");
    $("#status span:last-child").textContent = `${state.items.toLocaleString()} images · ${videoState}`;
    $("#index").textContent = state.index_ready
      ? `${state.index_backend} · ${state.items.toLocaleString()} × ${state.dimension}d`
      : "Image index not built";
    const models = await fetch("/api/models").then((response) => response.json());
    $("#model").textContent = `${models.active_model} · ${models.display_scope || models.trainable_scope}`;
  } catch (_) {
    $("#status").classList.add("is-error");
    $("#status span:last-child").textContent = "Service offline";
  }
}

async function search() {
  const query = $("#query").value.trim();
  if (!query) return;
  $("#run").disabled = true;
  $("#results").innerHTML = '<div class="loading-state">正在编码查询并检索完整目录…</div>';
  try {
    const response = await fetch("/api/search/text", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query, top_k: 10 }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error?.message || "search failed");
    $("#results").innerHTML = payload.results.map((item) => `
      <article class="result">
        ${item.image_url ? `<img loading="lazy" src="${escapeHtml(item.image_url)}" alt="COCO result ${item.content_id}">` : ""}
        <div class="result-body">
          <div class="result-meta">
            <span class="rank">#${item.rank}</span>
            <span class="score">cos ${item.score.toFixed(4)}</span>
          </div>
          <p>${escapeHtml(item.preview_caption || `content_id: ${item.content_id}`)}</p>
          <small>content_id ${item.content_id}</small>
        </div>
      </article>`).join("");
  } catch (error) {
    $("#results").innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
  } finally {
    $("#run").disabled = false;
  }
}

function renderVideoHeadline(payload) {
  const trained = payload.models.find((model) => model.id === "mosaic_trained_temporal_attention");
  const metrics = [
    ["Text → Video R@1", trained.text_to_video.r1, `mean baseline ${formatPercent(payload.models[0].text_to_video.r1)}`],
    ["Text → Video R@10", trained.text_to_video.r10, `mean baseline ${formatPercent(payload.models[0].text_to_video.r10)}`],
    ["Video → Text R@1", trained.video_to_text.r1, `mean baseline ${formatPercent(payload.models[0].video_to_text.r1)}`],
    ["Video → Text R@10", trained.video_to_text.r10, `mean baseline ${formatPercent(payload.models[0].video_to_text.r10)}`],
  ];
  $("#video-headline").innerHTML = metrics.map(([label, value, detail]) => `
    <div class="metric-cell">
      <span class="metric-label">${escapeHtml(label)}</span>
      <strong>${formatPercent(value)}</strong>
      <span>${escapeHtml(detail)}</span>
    </div>`).join("");
}

function renderVideoModels(payload) {
  const trained = payload.models.find((model) => model.trainable);
  $("#video-models").innerHTML = payload.models.map((model) => `
    <tr class="${model.trainable ? "is-primary" : ""}">
      <td>
        <span class="model-name">${escapeHtml(model.label)}</span>
        <span class="model-kind">${model.trainable ? "666,628 trainable params" : "frozen baseline"}</span>
      </td>
      <td class="${model === trained ? "best-value" : ""}">${formatPercent(model.text_to_video.r1)}</td>
      <td class="${model === trained ? "best-value" : ""}">${formatPercent(model.text_to_video.r10)}</td>
      <td class="${model === trained ? "best-value" : ""}">${formatPercent(model.video_to_text.r1)}</td>
      <td class="${model === trained ? "best-value" : ""}">${formatPercent(model.video_to_text.r10)}</td>
      <td class="${model === trained ? "best-value" : ""}">${formatPercent(model.text_to_video.mrr)}</td>
    </tr>`).join("");
}

function renderDeltas(payload) {
  const rows = [
    ["T2V R@1", payload.trained_vs_mean.text_to_video.r1],
    ["T2V R@10", payload.trained_vs_mean.text_to_video.r10],
    ["V2T R@1", payload.trained_vs_mean.video_to_text.r1],
    ["V2T R@10", payload.trained_vs_mean.video_to_text.r10],
  ];
  $("#video-deltas").innerHTML = rows.map(([label, row]) => `
    <div class="delta-item">
      <span>${escapeHtml(label)}</span>
      <strong>${formatPp(row.delta)}</strong>
      <small>95% CI ${escapeHtml(formatCi(row))}</small>
      <div class="ci-track" aria-hidden="true"><span style="--bar-width: ${Math.min(100, row.delta * 900)}%"></span></div>
    </div>`).join("");
}

function renderSamples(payload) {
  $("#sample-count").textContent = payload.samples.length
    ? `${payload.samples.length} fixed local samples`
    : "Local MSR-VTT media not installed";
  if (!payload.samples.length) {
    $("#video-samples").innerHTML = '<div class="empty-state">冻结指标仍可核对；本机没有可播放的白名单视频文件。</div>';
    return;
  }
  $("#video-samples").innerHTML = payload.samples.map((sample) => `
    <article class="video-card">
      <video controls preload="metadata" playsinline src="${escapeHtml(sample.stream_url)}#t=0.1"></video>
      <div class="video-copy">
        <div class="video-title-row">
          <strong>${escapeHtml(sample.label)}</strong>
          <span>${escapeHtml(sample.video_id)}</span>
        </div>
        <p>${escapeHtml(sample.caption)}</p>
        <small>${formatBytes(sample.bytes)} · fixed allowlist</small>
      </div>
    </article>`).join("");
}

function renderVideoEvidence(payload) {
  if (payload.status !== "ready") {
    $("#ranking-boundary").textContent = payload.error || "Frozen video report is unavailable.";
    $("#video-models").innerHTML = '<tr><td colspan="6">Video evidence is not ready.</td></tr>';
    renderSamples(payload);
    return;
  }
  $("#ranking-boundary").textContent = payload.ranking.reason;
  $("#video-coverage").textContent = `${payload.coverage.videos.toLocaleString()} videos · ${payload.coverage.captions.toLocaleString()} queries · epoch ${payload.coverage.checkpoint_epoch}`;
  $("#protocol-query").textContent = payload.protocol.query_policy;
  $("#audit-state").textContent = payload.evidence.audit_matches_report ? "Report-bound / complete" : payload.evidence.audit_status;
  $("#audit-digest").textContent = `evaluation ${shortHash(payload.evidence.evaluation_sha256)}`;
  renderVideoHeadline(payload);
  renderVideoModels(payload);
  renderDeltas(payload);
  renderSamples(payload);
}

async function loadVideoEvidence() {
  try {
    const response = await fetch("/api/video/evidence");
    const payload = await response.json();
    if (!response.ok) throw new Error("video evidence request failed");
    videoEvidence = payload;
    renderVideoEvidence(payload);
  } catch (error) {
    renderVideoEvidence({
      status: "not_ready",
      error: error.message,
      samples: [],
      ranking: { available: false },
    });
  }
}

$$('[data-view]').forEach((button) => button.addEventListener("click", () => setView(button.dataset.view)));
$("#run").addEventListener("click", search);
$("#query").addEventListener("keydown", (event) => { if (event.key === "Enter") search(); });
window.addEventListener("hashchange", () => setView(window.location.hash.slice(1), { updateHash: false }));

health();
const initialQuery = new URLSearchParams(window.location.search).get("q");
if (initialQuery) {
  $("#query").value = initialQuery;
  search();
}
setView(window.location.hash.slice(1), { updateHash: false });
