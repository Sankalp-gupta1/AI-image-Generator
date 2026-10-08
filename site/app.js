const API_BASE = "https://aihorde.net/api/v2";
const ANON_KEY = "0000000000";
const CLIENT_AGENT = "GenVisionStudio:1.0:https://github.com/Sankalp-gupta1/AI-image-Generator";

const STYLE_SUFFIXES = {
  "Photorealistic": "photorealistic professional photography, natural lighting, realistic textures, high detail, balanced composition",
  "Cinematic": "cinematic film still, dramatic natural lighting, atmospheric depth, professional color grading, detailed composition",
  "Product": "premium product photography, studio lighting, commercial composition, crisp material texture, clean background",
  "Anime": "high quality anime illustration, clean linework, expressive composition, detailed environment",
  "Concept Art": "high detail concept art, imaginative worldbuilding, polished digital painting, strong composition",
  "Minimal": "minimal visual design, refined composition, clean geometry, elegant negative space"
};

const SAMPLE_PROMPTS = [
  "A futuristic Indian metro station at night, rain reflections, cinematic architecture photography",
  "A premium transparent wireless speaker floating above a black glass pedestal, studio product photography",
  "An astronaut botanist tending a greenhouse on Mars, realistic documentary photography",
  "A floating eco-city above the clouds, solar architecture, hanging gardens, cinematic concept art",
  "A cozy AI research lab at midnight, holographic displays, warm practical lighting, photorealistic"
];

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const promptEl = $("prompt");
const promptCount = $("promptCount");
const stepsEl = $("steps");
const cfgEl = $("cfg");
const stepsValue = $("stepsValue");
const cfgValue = $("cfgValue");
const generateBtn = $("generateBtn");
const engineBadge = $("engineBadge");
const emptyState = $("emptyState");
const loadingState = $("loadingState");
const resultsEl = $("results");
const queueHeadline = $("queueHeadline");
const queueDetail = $("queueDetail");
const progressBar = $("progressBar");
const historyEl = $("history");

let selectedStyle = "Photorealistic";
let startedAt = 0;
let toastTimer = null;

function updatePromptCount() {
  promptCount.textContent = promptEl.value.length;
}

function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 2600);
}

function setStage(stage, headline, detail, progress) {
  if (stage === "idle") {
    engineBadge.className = "engine-badge";
    engineBadge.textContent = "READY";
    return;
  }

  if (stage === "busy") {
    engineBadge.className = "engine-badge busy";
    engineBadge.textContent = "GENERATING";
  } else if (stage === "error") {
    engineBadge.className = "engine-badge error";
    engineBadge.textContent = "ERROR";
  } else {
    engineBadge.className = "engine-badge";
    engineBadge.textContent = "COMPLETE";
  }

  if (headline) queueHeadline.textContent = headline;
  if (detail) queueDetail.textContent = detail;
  if (typeof progress === "number") {
    progressBar.style.width = Math.max(6, Math.min(96, progress)) + "%";
  }
}

function getDimensions() {
  const parts = $("aspect").value.split("x").map(Number);
  return { width: parts[0], height: parts[1] };
}

function buildPrompt() {
  const base = promptEl.value.trim();
  if (base.length < 3) throw new Error("Please enter a meaningful prompt.");
  const style = STYLE_SUFFIXES[selectedStyle] || "";
  return style ? base + ", " + style : base;
}

function resolveSeed() {
  const raw = Number($("seed").value);
  if (Number.isFinite(raw) && raw >= 0) return Math.floor(raw);
  return Math.floor(Math.random() * 2147483646) + 1;
}

function normalizeImageSource(value) {
  if (!value) return "";
  if (/^https?:\/\//i.test(value) || /^data:image\//i.test(value)) return value;
  return "data:image/webp;base64," + value;
}

async function apiFetch(path, options) {
  const response = await fetch(API_BASE + path, options);
  const text = await response.text();
  let body = {};
  try { body = text ? JSON.parse(text) : {}; } catch (_) { body = { message: text }; }

  if (!response.ok) {
    const message = body.message || body.rc || ("Request failed with HTTP " + response.status);
    throw new Error(message);
  }
  return body;
}

async function submitGeneration() {
  const positive = buildPrompt();
  const negative = $("negativePrompt").value.trim();
  const promptForApi = negative ? positive + " ### " + negative : positive;
  const dims = getDimensions();
  const seed = resolveSeed();
  const model = $("model").value;
  const n = Number($("count").value);
  const steps = Number(stepsEl.value);
  const cfg = Number(cfgEl.value);

  const payload = {
    prompt: promptForApi,
    params: {
      sampler_name: "k_euler_a",
      cfg_scale: cfg,
      width: dims.width,
      height: dims.height,
      steps: steps,
      n: n,
      seed: String(seed),
      karras: true,
      hires_fix: false,
      clip_skip: 1
    },
    allow_downgrade: true,
    nsfw: false,
    censor_nsfw: true,
    trusted_workers: false,
    slow_workers: true,
    extra_slow_workers: false,
    r2: true,
    shared: false
  };

  if (model) payload.models = [model];

  const response = await apiFetch("/generate/async", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "apikey": ANON_KEY,
      "Client-Agent": CLIENT_AGENT
    },
    body: JSON.stringify(payload)
  });

  if (!response.id) throw new Error("The generation service did not return a request ID.");

  return {
    id: response.id,
    seed,
    dims,
    positive,
    negative,
    requestedModel: model || "Auto",
    count: n,
    steps,
    cfg
  };
}

async function waitForGeneration(job) {
  const timeoutAt = Date.now() + 10 * 60 * 1000;
  let ticks = 0;

  while (Date.now() < timeoutAt) {
    const check = await apiFetch("/generate/check/" + encodeURIComponent(job.id), {
      headers: { "Client-Agent": CLIENT_AGENT }
    });

    ticks += 1;
    if (check.faulted) throw new Error("The distributed generation job faulted. Please try again.");

    const waiting = Number(check.waiting || 0);
    const processing = Number(check.processing || 0);
    const finished = Number(check.finished || 0);
    const queuePosition = check.queue_position;
    const waitTime = check.wait_time;

    if (check.done) break;

    let detail = "Waiting for an available GPU worker";
    if (processing > 0) detail = "A worker is rendering your image";
    else if (queuePosition !== undefined && queuePosition !== null) detail = "Queue position: " + queuePosition;
    if (waitTime && !processing) detail += " · estimated wait " + waitTime + "s";

    const progress = processing > 0 ? 72 + Math.min(18, ticks) : Math.min(62, 12 + ticks * 3);
    setStage("busy", processing > 0 ? "Diffusion inference in progress..." : "Waiting in the GPU queue...", detail, progress);

    await sleep(2200);
  }

  if (Date.now() >= timeoutAt) throw new Error("Generation timed out after 10 minutes. Anonymous requests can be slow during peak load.");

  setStage("busy", "Finalizing generated image...", "Retrieving result and metadata", 94);

  return apiFetch("/generate/status/" + encodeURIComponent(job.id), {
    headers: { "Client-Agent": CLIENT_AGENT }
  });
}

function renderResults(status, job) {
  const generations = Array.isArray(status.generations) ? status.generations : [];
  if (!generations.length) throw new Error("Generation completed, but no image result was returned.");

  resultsEl.innerHTML = "";
  resultsEl.className = "results" + (generations.length > 1 ? " cols-2" : "");

  generations.forEach((gen, index) => {
    const src = normalizeImageSource(gen.img);
    const card = document.createElement("article");
    card.className = "result-card";

    const image = document.createElement("img");
    image.src = src;
    image.alt = "Generated AI image " + (index + 1);

    const actions = document.createElement("div");
    actions.className = "result-actions";

    const open = document.createElement("a");
    open.href = src;
    open.target = "_blank";
    open.rel = "noreferrer";
    open.textContent = "Open full ↗";

    const copy = document.createElement("button");
    copy.type = "button";
    copy.textContent = "Copy prompt";
    copy.onclick = async () => {
      try {
        await navigator.clipboard.writeText(job.positive);
        toast("Prompt copied");
      } catch (_) {
        toast("Copy is not available in this browser");
      }
    };

    actions.append(open, copy);
    card.append(image, actions);
    resultsEl.appendChild(card);
  });

  emptyState.classList.add("hidden");
  loadingState.classList.add("hidden");
  resultsEl.classList.remove("hidden");

  const first = generations[0] || {};
  const elapsed = ((Date.now() - startedAt) / 1000).toFixed(1);
  $("metaModel").textContent = first.model || job.requestedModel || "Distributed";
  $("metaSeed").textContent = first.seed || job.seed;
  $("metaSize").textContent = job.dims.width + "×" + job.dims.height;
  $("metaTime").textContent = elapsed + "s";
  setStage("done", "", "", 100);

  const historyItem = {
    image: normalizeImageSource(first.img),
    prompt: promptEl.value.trim(),
    model: first.model || job.requestedModel,
    seed: first.seed || job.seed,
    at: Date.now()
  };
  saveHistory(historyItem);
}

function setLoadingVisible() {
  emptyState.classList.add("hidden");
  resultsEl.classList.add("hidden");
  loadingState.classList.remove("hidden");
  progressBar.style.width = "7%";
}

function restoreCanvasAfterError() {
  loadingState.classList.add("hidden");
  if (!resultsEl.children.length) emptyState.classList.remove("hidden");
}

async function generate() {
  if (generateBtn.disabled) return;

  try {
    generateBtn.disabled = true;
    startedAt = Date.now();
    setLoadingVisible();
    setStage("busy", "Preparing generation request...", "Validating prompt and sampling controls", 8);

    const job = await submitGeneration();
    $("seed").value = job.seed;
    $("metaSeed").textContent = job.seed;
    $("metaSize").textContent = job.dims.width + "×" + job.dims.height;

    setStage("busy", "Generation accepted", "Request " + job.id.slice(0, 8) + "… queued for a GPU worker", 14);
    const status = await waitForGeneration(job);
    renderResults(status, job);
    toast("Generation complete");
  } catch (error) {
    console.error(error);
    setStage("error", "Generation could not complete", error.message || "Unknown error", 8);
    queueHeadline.textContent = "Generation could not complete";
    queueDetail.textContent = error.message || "Please try again.";
    setTimeout(restoreCanvasAfterError, 3800);
    toast(error.message || "Generation failed");
  } finally {
    generateBtn.disabled = false;
  }
}

function getHistory() {
  try {
    return JSON.parse(localStorage.getItem("genvision_history") || "[]");
  } catch (_) {
    return [];
  }
}

function saveHistory(item) {
  const history = getHistory();
  history.unshift(item);
  localStorage.setItem("genvision_history", JSON.stringify(history.slice(0, 8)));
  renderHistory();
}

function renderHistory() {
  const history = getHistory();
  historyEl.innerHTML = "";

  if (!history.length) {
    const empty = document.createElement("p");
    empty.className = "community-note";
    empty.textContent = "No generations in this browser session yet.";
    historyEl.appendChild(empty);
    return;
  }

  history.forEach((item) => {
    const card = document.createElement("article");
    card.className = "history-card";

    const img = document.createElement("img");
    img.src = item.image;
    img.alt = "Previous generated image";

    const text = document.createElement("div");
    const title = document.createElement("b");
    title.textContent = item.prompt;
    const meta = document.createElement("small");
    meta.textContent = (item.model || "AI Horde") + " · seed " + item.seed;

    text.append(title, meta);
    card.append(img, text);
    historyEl.appendChild(card);
  });
}

document.querySelectorAll(".chip").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".chip").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    selectedStyle = button.dataset.style;
  });
});

promptEl.addEventListener("input", updatePromptCount);
stepsEl.addEventListener("input", () => stepsValue.textContent = stepsEl.value);
cfgEl.addEventListener("input", () => cfgValue.textContent = Number(cfgEl.value).toFixed(1));

$("seedBtn").addEventListener("click", () => {
  $("seed").value = Math.floor(Math.random() * 2147483646) + 1;
});

$("randomPromptBtn").addEventListener("click", () => {
  promptEl.value = SAMPLE_PROMPTS[Math.floor(Math.random() * SAMPLE_PROMPTS.length)];
  updatePromptCount();
  toast("Sample prompt loaded");
});

$("clearHistory").addEventListener("click", () => {
  localStorage.removeItem("genvision_history");
  renderHistory();
  toast("History cleared");
});

generateBtn.addEventListener("click", generate);
promptEl.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") generate();
});

updatePromptCount();
renderHistory();
setStage("idle");
