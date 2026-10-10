// app.js — Interface du routeur de demandes étudiantes.
// Ne suppose JAMAIS quel modèle a gagné la comparaison : lit manifest.json au
// démarrage et charge le bon pipeline de vectorisation (tfidf.js ou
// bilstm_tokenizer.js) en conséquence. Si le pipeline Python re-sélectionne un
// autre modèle plus tard (nouveau corpus.csv), il suffit de republier le
// dossier exporté par export_web.py : cette page s'adapte automatiquement.

// app.js — Interface du routeur de demandes étudiantes.
// Ne suppose JAMAIS quel modèle a gagné la comparaison : lit manifest.json au
// démarrage et charge le bon pipeline de vectorisation (tfidf.js ou
// bilstm_tokenizer.js) en conséquence. Si le pipeline Python re-sélectionne un
// autre modèle plus tard (nouveau corpus.csv), il suffit de republier le
// dossier exporté par export_web.py : cette page s'adapte automatiquement.

const CACHE_NAME = "routeur-nlp-v4";
let session = null;
let manifest = null;
let vectorizeFn = null; // (text) -> Float32Array, injectée selon la famille du modèle (tfidf/bilstm)
let transformerClassifier = null; // pipeline Transformers.js, uniquement pour les transformers

async function cachedFetch(url) {
  if ("caches" in window) {
    const cache = await caches.open(CACHE_NAME);
    const cached = await cache.match(url);
    if (cached) return { response: cached, fromCache: true };
    const response = await fetch(url);
    if (response.ok) await cache.put(url, response.clone());
    return { response, fromCache: false };
  }
  return { response: await fetch(url), fromCache: false };
}

function setStatus(state, text) {
  const el = document.getElementById("model-status");
  el.className = `model-status ${state}`;
  el.querySelector(".status-text").textContent = text;
}

// Fonction utilitaire pour vérifier si le modèle est un transformer
function isTransformerModel(manifest) {
  const name = manifest.model_name.toLowerCase();
  return manifest.family === "transformer" || name.includes("distilcamembert") || name.includes("camembert");
}

async function initModel() {
  setStatus("loading", "Chargement du manifeste...");
  try {
    const manifestResult = await cachedFetch("manifest.json");
    manifest = await manifestResult.response.json();

    setStatus("loading", `Chargement du modèle (${manifest.model_name})...`);

    if (isTransformerModel(manifest)) {
      // CamemBERT / DistilCamemBERT, via Transformers.js
      const { pipeline, env } = await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.0.1");
      env.allowLocalModels = true;
      env.allowRemoteModels = false;
      env.localModelPath = "./";
      env.useBrowserCache = false; // Désactivé pour forcer le chargement du modèle fusionné autonome
      
      const modelDir = (manifest.files && manifest.files.model_dir) || ".";
      
      transformerClassifier = await pipeline("text-classification", modelDir, {
        quantized: false, // Important : force la lecture de model.onnx standard
        progress_callback: (p) => {
          if (p.status === "progress" && p.total) {
            setStatus("loading", `Téléchargement du modèle... ${Math.round((p.loaded / p.total) * 100)}%`);
          }
        },
      });
      setStatus("ready", `${manifest.model_name} — chargé via Transformers.js`);
    } else {
      // Logique existante pour TF-IDF et BiLSTM (ONNX Runtime classique)
      const [onnxResult, vocabResult] = await Promise.all([
        cachedFetch(manifest.files.onnx),
        cachedFetch(manifest.files.vocab),
      ]);
      const onnxBuffer = await onnxResult.response.arrayBuffer();
      const vocabJson = await vocabResult.response.json();

      if (manifest.family === "tfidf") {
        setVocab(vocabJson); // depuis tfidf.js
        vectorizeFn = vectorize; // depuis tfidf.js
      } else if (manifest.family === "bilstm") {
        setTokenizerData(vocabJson); // depuis bilstm_tokenizer.js
        vectorizeFn = vectorizeForBilstm; // depuis bilstm_tokenizer.js
      } else {
        throw new Error(`Famille de modèle non supportée par cette page : ${manifest.family}`);
      }

      session = await ort.InferenceSession.create(new Uint8Array(onnxBuffer));

      const cached = onnxResult.fromCache && vocabResult.fromCache;
      setStatus("ready", `${manifest.model_name} — ${cached ? "en cache" : "téléchargé"} (${(onnxBuffer.byteLength / 1024).toFixed(0)} Ko)`);
    }

    document.getElementById("single-submit").disabled = false;
    document.getElementById("batch-submit").disabled = false;
  } catch (err) {
    console.error(err);
    setStatus("error", "Erreur de chargement du modèle — voir la console");
  }
}

async function classifyOne(text, topK = 3) {
  if (isTransformerModel(manifest)) {
    // On passe à la fois topk et top_k avec le nombre total de classes pour
    // couvrir toutes les versions de l'API Transformers.js.
    const output = await transformerClassifier(text, { 
        topk: manifest.classes.length,
        top_k: manifest.classes.length 
    });
    
    // On s'assure d'avoir un tableau, puis on trie et on coupe aux 3 meilleurs
    const scores = Array.isArray(output) ? output : [output];
    return scores
      .map((s) => ({ service: s.label, confidence: s.score }))
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, topK);
  }

  // Logique TF-IDF / BiLSTM
  const vec = vectorizeFn(text);
  const tensor = new ort.Tensor("float32", vec, [1, vec.length]);
  const results = await session.run({ [session.inputNames[0]]: tensor });
  const outputName = session.outputNames[session.outputNames.length - 1];
  const proba = Array.from(results[outputName].data);
  return proba
    .map((p, i) => ({ service: manifest.classes[i], confidence: p }))
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, topK);
}

async function classifyBatch(texts, topK = 3) {
  const out = [];
  for (const t of texts) out.push(await classifyOne(t, topK));
  return out;
}

function renderSingleResult(ranked) {
  const board = document.getElementById("single-board");
  board.innerHTML = "";
  ranked.forEach((r, i) => {
    const row = document.createElement("div");
    row.className = `flap-row ${i === 0 ? "primary" : ""} animate`;
    row.style.animationDelay = `${i * 60}ms`;
    row.innerHTML = `
      <span class="rank">${String(i + 1).padStart(2, "0")}</span>
      <span class="service">${r.service}</span>
      <span class="confidence">${(r.confidence * 100).toFixed(1)}%</span>`;
    board.appendChild(row);
  });
  board.setAttribute("aria-live", "polite");
}

function parseFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const content = reader.result;
      if (file.name.toLowerCase().endsWith(".csv")) {
        resolve(parseCsv(content));
      } else {
        resolve(content.split(/\r?\n/).map((l) => l.trim()).filter((l) => l.length > 0));
      }
    };
    reader.onerror = reject;
    reader.readAsText(file, "utf-8");
  });
}

function parseCsv(content) {
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;
  for (let i = 0; i < content.length; i++) {
    const c = content[i];
    if (inQuotes) {
      if (c === '"' && content[i + 1] === '"') { field += '"'; i++; }
      else if (c === '"') { inQuotes = false; }
      else field += c;
    } else if (c === '"') { inQuotes = true; }
    else if (c === ",") { row.push(field); field = ""; }
    else if (c === "\n" || c === "\r") {
      if (field.length || row.length) { row.push(field); rows.push(row); }
      field = ""; row = [];
      if (c === "\r" && content[i + 1] === "\n") i++;
    } else field += c;
  }
  if (field.length || row.length) { row.push(field); rows.push(row); }
  if (!rows.length) return [];

  const header = rows[0].map((h) => h.trim().toLowerCase());
  let col = header.indexOf("text");
  if (col === -1) col = header.indexOf("message");
  const dataRows = col !== -1 ? rows.slice(1) : rows;
  const useCol = col !== -1 ? col : 0;
  return dataRows.map((r) => (r[useCol] || "").trim()).filter((t) => t.length > 0);
}

function csvEscape(value) {
  const s = String(value);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function resultsToCsv(messages, results) {
  const header = ["text", "service_rang1", "confiance_rang1", "service_rang2", "confiance_rang2", "service_rang3", "confiance_rang3"];
  const lines = [header.join(",")];
  messages.forEach((msg, i) => {
    const r = results[i];
    const cells = [msg];
    for (let k = 0; k < 3; k++) {
      cells.push(r[k] ? r[k].service : "", r[k] ? r[k].confidence.toFixed(4) : "");
    }
    lines.push(cells.map(csvEscape).join(","));
  });
  return lines.join("\n");
}

function downloadCsv(csvText, filename) {
  const blob = new Blob(["\uFEFF" + csvText], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function renderBatchTable(messages, results) {
  const container = document.getElementById("batch-results");
  if (!messages.length) {
    container.innerHTML = '<p class="empty-state">Aucun résultat pour le moment.</p>';
    return;
  }
  let html = '<div class="table-scroll"><table class="results"><thead><tr>' +
    "<th>Message</th><th>Service identifié</th><th>Confiance</th><th>2e service</th><th>3e service</th>" +
    "</tr></thead><tbody>";
  messages.forEach((msg, i) => {
    const r = results[i];
    html += `<tr>
      <td class="msg-cell" title="${msg.replace(/"/g, "&quot;")}">${msg}</td>
      <td class="service-cell">${r[0].service}</td>
      <td class="conf-cell">${(r[0].confidence * 100).toFixed(1)}%</td>
      <td class="conf-cell">${r[1] ? r[1].service : "–"}</td>
      <td class="conf-cell">${r[2] ? r[2].service : "–"}</td>
    </tr>`;
  });
  html += "</tbody></table></div>";
  container.innerHTML = html;
}

// ---------- Wiring UI ----------

document.addEventListener("DOMContentLoaded", () => {
  initModel();

  // Tabs
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.setAttribute("aria-selected", "false"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.setAttribute("aria-selected", "true");
      document.getElementById(btn.dataset.target).classList.add("active");
    });
  });

  // Mode unitaire
  document.getElementById("single-submit").addEventListener("click", async () => {
    const text = document.getElementById("single-input").value.trim();
    // CORRECTION : Accepte l'exécution si on a soit session (TF-IDF/BiLSTM) soit transformerClassifier
    if (!text || (!session && !transformerClassifier)) return;
    
    const btn = document.getElementById("single-submit");
    btn.disabled = true;
    btn.textContent = "Analyse...";
    try {
      const ranked = await classifyOne(text);
      renderSingleResult(ranked);
    } finally {
      btn.disabled = false;
      btn.textContent = "Identifier le service";
    }
  });

  // Mode lot
  let batchFile = null;
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragover"); });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length) {
      batchFile = e.dataTransfer.files[0];
      document.querySelector(".dropzone .filename").textContent = batchFile.name;
    }
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) {
      batchFile = fileInput.files[0];
      document.querySelector(".dropzone .filename").textContent = batchFile.name;
    }
  });

  let lastMessages = [];
  let lastResults = [];

  document.getElementById("batch-submit").addEventListener("click", async () => {
    // CORRECTION : Accepte l'exécution si on a soit session (TF-IDF/BiLSTM) soit transformerClassifier
    if (!batchFile || (!session && !transformerClassifier)) return;
    
    const btn = document.getElementById("batch-submit");
    btn.disabled = true;
    const progressWrap = document.getElementById("progress-wrap");
    const bar = document.getElementById("progress-bar");
    progressWrap.style.display = "block";
    bar.style.width = "0%";

    try {
      const messages = await parseFile(batchFile);
      const results = [];
      for (let i = 0; i < messages.length; i++) {
        results.push(await classifyOne(messages[i]));
        bar.style.width = `${Math.round(((i + 1) / messages.length) * 100)}%`;
      }
      lastMessages = messages;
      lastResults = results;
      renderBatchTable(messages, results);
      document.getElementById("export-btn").disabled = messages.length === 0;
    } finally {
      btn.disabled = false;
      setTimeout(() => { progressWrap.style.display = "none"; }, 400);
    }
  });

  document.getElementById("export-btn").addEventListener("click", () => {
    if (!lastMessages.length) return;
    const csv = resultsToCsv(lastMessages, lastResults);
    downloadCsv(csv, "resultats_routeur.csv");
  });
});