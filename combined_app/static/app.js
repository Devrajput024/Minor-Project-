const $ = (s) => document.querySelector(s);
const state = { imageFile: null, imageVerified: false, imageObjectUrl: null, omicsFile: null, omicsResult: null };
let imagingOverview = null;
let imagingEvaluations = [];
let omicsSort = { key: "sample", direction: 1 };
let omicsPage = 0;
const OMICS_PAGE_SIZE = 25;

function status(el, message = "", kind = "") {
  el.textContent = message;
  el.className = `inline-status ${kind}`;
}
function pct(value) { return `${(Number(value) * 100).toFixed(1)}%`; }
function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}
async function jsonRequest(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(data.detail || data.error || `Request failed (${response.status})`);
    error.payload = data;
    throw error;
  }
  return data;
}

function renderImageMetrics(info) {
  const root = $("#image-metrics");
  const available = (info.modalities || []).filter((m) => m.status === "available");
  const items = available.map((m) => {
    const metric = m.metrics || {};
    return `<div class="metric-item"><span>${escapeHtml(m.name)} · test n=${metric.n_test ?? "—"}</span><strong>${pct(metric.test_accuracy)}</strong><small>accuracy&nbsp; / &nbsp;ROC-AUC ${Number(metric.roc_auc || 0).toFixed(3)}</small></div>`;
  }).join("");
  root.innerHTML = `<div class="metric-intro"><span class="step">PRESERVED BASELINES</span><strong>Existing test results</strong><small>Source project evaluation</small></div>${items || '<div class="loading">Imaging model information unavailable.</div>'}`;
}

function currentModalityInfo() {
  return imagingOverview?.modalities?.find((item) => item.key === $("#modality").value);
}

function renderImageEvaluation() {
  const entry = imagingEvaluations.find((item) => item.id === $("#evaluation-select").value);
  const metrics = entry?.metrics;
  const target = $("#image-evaluation-metrics");
  if (!entry || !metrics) {
    target.innerHTML = '<p class="detail-placeholder">Held-out evaluation metrics are unavailable for this modality.</p>';
    $("#confusion-matrix").innerHTML = "";
    $("#roc-note").textContent = "ROC curve: Not available for this selected checkpoint.";
    $("#evaluation-curves").innerHTML = "";
    return;
  }
  const values = [
    ["Accuracy", metrics.accuracy], ["Precision · positive class", metrics.precision],
    ["Recall / sensitivity · positive class", metrics.recall_sensitivity], ["Specificity", metrics.specificity],
    ["F1-score · positive class", metrics.f1_score], ["Macro F1", metrics.macro_f1], ["ROC-AUC", metrics.roc_auc],
  ];
  target.innerHTML = values.map(([label, value]) => `<div class="evaluation-metric"><span>${escapeHtml(label)}</span><strong>${value == null ? "Not available" : pct(value)}</strong></div>`).join("");
  const matrix = metrics.confusion_matrix;
  if (matrix?.length === 2 && matrix[0]?.length === 2) {
    const labels = entry.class_mapping && typeof entry.class_mapping === "object" ? Object.keys(entry.class_mapping) : ["class 0", "class 1"];
    $("#confusion-matrix").innerHTML = `<table><thead><tr><th></th><th>Pred. ${escapeHtml(labels[0])}</th><th>Pred. ${escapeHtml(labels[1])}</th></tr></thead><tbody><tr><th>Actual ${escapeHtml(labels[0])}</th><td>${matrix[0][0]}</td><td>${matrix[0][1]}</td></tr><tr><th>Actual ${escapeHtml(labels[1])}</th><td>${matrix[1][0]}</td><td>${matrix[1][1]}</td></tr></tbody></table>`;
  } else {
    $("#confusion-matrix").innerHTML = '<p class="detail-placeholder">Confusion matrix is not available.</p>';
  }
  const historical = !entry.active_for_inference;
  $("#evaluation-caption").textContent = `${entry.modality_label} · ${entry.experiment_identifier} · ${entry.evaluation_status} · ${entry.split} · N=${entry.test_samples ?? "Not available"}`;
  $("#roc-note").innerHTML = `<strong>ROC-AUC ${metrics.roc_auc == null ? "Not available" : Number(metrics.roc_auc).toFixed(3)}</strong><p>${entry.curve_urls?.roc ? "ROC curve is from this saved experiment and split." : "AUC is recorded, but a ROC curve matched to this checkpoint and split is not available."}<br>${entry.curve_note ? escapeHtml(entry.curve_note) : "PR curve artifact is unavailable for this exact run."}</p>${entry.provenance_note ? `<p class="provenance-warning">${escapeHtml(entry.provenance_note)}</p>` : ""}${historical ? '<span class="historical-banner">HISTORICAL RESULT · NOT THE LIVE PREDICTION CHECKPOINT</span>' : ""}`;
  const curves = [["roc", "ROC curve"], ["pr", "Precision–recall curve"]].filter(([kind]) => entry.curve_urls?.[kind]);
  $("#evaluation-curves").innerHTML = curves.length ? `<div class="curve-grid">${curves.map(([kind, label]) => `<figure><img src="/imaging${entry.curve_urls[kind]}" alt="${escapeHtml(label)} for ${escapeHtml(entry.experiment_identifier)}"><figcaption>${escapeHtml(label)} · ${escapeHtml(entry.experiment_identifier)}</figcaption></figure>`).join("")}</div>` : '<p class="curve-unavailable">No matching ROC or PR curve image is available for this checkpoint and split. Curves from other runs are kept separate.</p>';
  $("#download-evaluation-report").href = `/imaging/api/reports/evaluation/${encodeURIComponent(entry.id)}.pdf`;
}

function renderVerification({ file, inspected = null, imageInfo = null, statusText = "REJECTED", validation = "Passed image decoding" }) {
  const detected = inspected?.detected_modality || inspected?.raw_modality || "Not verified";
  const confidence = inspected?.confidence;
  const width = imageInfo?.width || inspected?.width;
  const height = imageInfo?.height || inspected?.height;
  const format = imageInfo?.format || inspected?.format || file?.type?.split("/").pop() || file?.name?.split(".").pop();
  $("#verification-details").innerHTML = `<dl class="detail-list">
    <div><dt>Filename</dt><dd>${escapeHtml(file?.name || "Not available")}</dd></div>
    <div><dt>Image dimensions</dt><dd>${width && height ? `${width} × ${height} px` : "Not available"}</dd></div>
    <div><dt>Image format</dt><dd>${escapeHtml(format ? format.toUpperCase() : "Not available")}</dd></div>
    <div><dt>Expected modality</dt><dd>${escapeHtml($("#modality").value)}</dd></div>
    <div><dt>Detected modality</dt><dd>${escapeHtml(detected)}</dd></div>
    <div><dt>Detector confidence</dt><dd>${confidence == null ? "Not available" : pct(confidence)} <small>(separate from classifier scores)</small></dd></div>
    <div><dt>Image validation</dt><dd>${escapeHtml(validation)}</dd></div>
    <div><dt>Verification status</dt><dd><span class="verification-pill ${statusText === "PASSED" ? "pass" : "reject"}">${escapeHtml(statusText)}</span></dd></div>
  </dl>`;
}

function renderInterpretation(result) {
  const scores = result.class_probabilities || {};
  const benign = Number(scores.benign);
  const malignant = Number(scores.malignant);
  const threshold = Number(result.classification_threshold ?? 0.5);
  const winner = result.predicted_class;
  const winnerScore = Number(scores[winner]);
  const gap = Math.abs(benign - malignant);
  const distance = Math.abs(malignant - threshold);
  const nearBoundary = distance <= 0.05;
  const statement = winner === "benign"
    ? "The model assigned a higher score to the benign class than to the malignant class. Therefore, its predicted class is benign."
    : "The model assigned a higher score to the malignant class than to the benign class. Therefore, its predicted class is malignant.";
  $("#prediction-interpretation").innerHTML = `<p class="interpretation-copy">${statement}</p>
    <div class="interpretation-stats"><div><span>Score difference</span><strong>${(gap * 100).toFixed(1)} pp</strong></div><div><span>Distance from 0.50 boundary</span><strong>${(distance * 100).toFixed(1)} pp</strong></div><div><span>Decision threshold</span><strong>${threshold.toFixed(2)}</strong></div></div>
    <p class="uncertainty-note ${nearBoundary ? "near" : ""}">${nearBoundary ? "The malignant score is within 5 percentage points of the 0.50 decision boundary. Treat this output as close to the cutoff." : "The score is outside the display's near-boundary band; this does not mean the model is certain."} The 5-point band is a visual cue only and does not alter the classifier decision.</p>
    <small class="score-caveat">${escapeHtml(result.score_type || "Softmax model scores; not calibrated probabilities.")}</small>`;
}

function renderModelInformation(info) {
  const details = info || {};
  const rows = [
    ["Architecture", details.architecture], ["Checkpoint", details.checkpoint_filename], ["Framework", details.framework],
    ["Input dimensions", details.input_dimensions], ["Preprocessing", details.preprocessing],
    ["Normalization mean", JSON.stringify(details.normalization_mean)], ["Normalization standard deviation", JSON.stringify(details.normalization_std)],
    ["Class mapping", JSON.stringify(details.class_mapping)], ["Inference device", details.device],
    ["Model version", details.model_version], ["Experiment identifier", details.experiment_identifier], ["Decision rule", details.classification_rule],
  ];
  $("#model-information").innerHTML = rows.map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value == null || value === "" ? "Not available" : value)}</dd></div>`).join("");
}

function renderGradcam(result) {
  const container = $("#gradcam-content");
  if (!result?.available) {
    container.innerHTML = `<p class="gradcam-unavailable">${escapeHtml(result?.explanation || "Grad-CAM is not available for this architecture or prediction. The classifier output is unaffected.")}</p>`;
    return;
  }
  container.innerHTML = `<div class="gradcam-gallery">
    <figure><img src="${state.imageObjectUrl}" alt="Original uploaded image"><figcaption>Original image</figcaption></figure>
    <figure><img src="${result.heatmap}" alt="Grad-CAM heatmap for predicted class"><figcaption>Grad-CAM heatmap</figcaption></figure>
    <figure><img src="${result.overlay}" alt="Grad-CAM overlay on uploaded image"><figcaption>Heatmap overlay</figcaption></figure>
  </div><p class="gradcam-caption">${escapeHtml(result.explanation)}</p>`;
}

function renderImagePrediction(result) {
  state.latestPrediction = result;
  const scores = result.class_probabilities || {};
  const predicted = result.predicted_class;
  const predictedScore = Number(scores[predicted] ?? result.model_confidence);
  const labelClass = predicted === "malignant" ? "malignant" : "benign";
  const scoreCards = ["benign", "malignant"].map((label) => {
    const value = Number(scores[label] || 0);
    return `<div class="class-score ${label}"><div><span>${label.toUpperCase()}</span><strong>${pct(value)}</strong></div><div class="score-track"><i style="width:${Math.max(0, Math.min(100, value * 100))}%"></i></div></div>`;
  }).join("");
  $("#image-result").innerHTML = `<div class="result-summary ${labelClass}"><div class="result-image"><img src="${state.imageObjectUrl}" alt="Uploaded medical image"></div><div class="result-primary"><span class="prediction-badge">PREDICTION STATUS · COMPLETE</span><div class="prediction-label ${labelClass}">${escapeHtml(predicted.toUpperCase())}</div><div class="confidence">Predicted class score · <strong>${pct(predictedScore)}</strong></div></div></div>
    <div class="class-score-list">${scoreCards}</div><p class="result-score-note">${escapeHtml(result.score_type || "Softmax model scores; not calibrated probabilities.")}</p><button class="report-button individual-report-button" id="download-prediction-report" type="button">Download individual prediction PDF ↓</button>`;
  $("#image-empty").classList.add("hidden");
  $("#image-result").classList.remove("hidden");
  renderVerification({ file: state.imageFile, inspected: state.latestInspection, imageInfo: result.image, statusText: "PASSED", validation: "Passed format and image decoding checks" });
  renderInterpretation(result);
  renderGradcam(result.gradcam);
  renderModelInformation(result.model_information);
  $("#download-prediction-report").addEventListener("click", downloadPredictionReport);
}

async function downloadPredictionReport() {
  if (!state.latestPrediction || !state.imageFile) return;
  const button = $("#download-prediction-report");
  button.disabled = true;
  button.textContent = "Preparing PDF…";
  const body = new FormData();
  body.append("file", state.imageFile);
  body.append("result_json", JSON.stringify(state.latestPrediction));
  try {
    const response = await fetch("/imaging/api/reports/prediction.pdf", { method: "POST", body });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.detail || `Report generation failed (${response.status})`);
    }
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement("a");
    link.href = url; link.download = "individual_imaging_prediction_report.pdf"; link.click();
    URL.revokeObjectURL(url);
  } catch (error) {
    status($("#image-status"), `PDF report failed: ${error.message}`, "error");
  } finally {
    button.disabled = false;
    button.textContent = "Download individual prediction PDF ↓";
  }
}

function renderOmicsMetrics(omics) {
  const body = $("#omics-metrics tbody");
  const labels = { cnv: "CNV · Logistic Regression", rna: "RNA · Random Forest", mutation: "Mutation · Logistic Regression", protein: "Protein · Random Forest", integrated: "Integrated · Random Forest" };
  body.innerHTML = Object.entries(omics.test_metrics || {}).map(([scope, m]) => {
    const winner = scope === "rna" ? '<span class="winner-tag">CV WINNER</span>' : "";
    return `<tr><td>${escapeHtml(labels[scope] || scope)}${winner}</td><td>${pct(m.accuracy)}</td><td>${pct(m.balanced_accuracy)}</td><td>${pct(m.deceased_recall)}</td><td>${Number(m.roc_auc).toFixed(3)}</td><td>${Number(m.pr_auc).toFixed(3)}</td></tr>`;
  }).join("") || '<tr><td colspan="6">Saved metrics are unavailable.</td></tr>';
}

function omicsSampleLabel(row, result = state.omicsResult) {
  return row.sample_identifier == null || row.sample_identifier === ""
    ? `Row ${row.row}`
    : String(row.sample_identifier);
}

function renderOmicsTable() {
  const result = state.omicsResult;
  if (!result) return;
  const query = $("#omics-table-search").value.trim().toLocaleLowerCase();
  const filtered = result.results.filter((row) => `${omicsSampleLabel(row)} ${row.predicted_vital_status}`.toLocaleLowerCase().includes(query));
  const key = omicsSort.key;
  filtered.sort((left, right) => {
    let a;
    let b;
    if (key === "sample") {
      a = omicsSampleLabel(left).toLocaleLowerCase();
      b = omicsSampleLabel(right).toLocaleLowerCase();
    } else if (key === "predicted_class") {
      a = left.predicted_class;
      b = right.predicted_class;
    } else if (key === "difference") {
      a = Math.abs(Number(left.class_probabilities?.alive || 0) - Number(left.class_probabilities?.deceased || 0));
      b = Math.abs(Number(right.class_probabilities?.alive || 0) - Number(right.class_probabilities?.deceased || 0));
    } else {
      a = Number(left.class_probabilities?.[key] || 0);
      b = Number(right.class_probabilities?.[key] || 0);
    }
    return (a < b ? -1 : a > b ? 1 : 0) * omicsSort.direction;
  });
  const pageCount = Math.max(1, Math.ceil(filtered.length / OMICS_PAGE_SIZE));
  omicsPage = Math.min(omicsPage, pageCount - 1);
  const visible = filtered.slice(omicsPage * OMICS_PAGE_SIZE, (omicsPage + 1) * OMICS_PAGE_SIZE);
  const identifierHeading = result.sample_identifier_column
    ? `SAMPLE IDENTIFIER · ${escapeHtml(result.sample_identifier_column)}`
    : "SAMPLE IDENTIFIER · ROW NUMBER";
  $("#omics-table-body").innerHTML = visible.map((row) => {
    const alive = Number(row.class_probabilities?.alive || 0);
    const deceased = Number(row.class_probabilities?.deceased || 0);
    return `<tr><td>${escapeHtml(omicsSampleLabel(row))}</td><td class="${row.predicted_class === 1 ? "class-deceased" : "class-alive"}">${escapeHtml(row.predicted_vital_status)}</td><td>${pct(alive)}</td><td>${pct(deceased)}</td><td>${pct(Math.abs(alive - deceased))}</td></tr>`;
  }).join("") || '<tr><td colspan="5">No samples match this search.</td></tr>';
  $("#omics-table-count").textContent = `Showing ${filtered.length ? omicsPage * OMICS_PAGE_SIZE + 1 : 0}–${Math.min((omicsPage + 1) * OMICS_PAGE_SIZE, filtered.length)} of ${filtered.length} matching samples`;
  $("#omics-previous-page").disabled = omicsPage === 0;
  $("#omics-next-page").disabled = omicsPage >= pageCount - 1;
  $("#omics-dashboard-caption").textContent = `${result.filename} · ${result.rows} samples · ${result.used_features} required features validated`;
  document.querySelector(".omics-detail-table thead th:first-child button").innerHTML = `${identifierHeading} <span aria-hidden="true">↕</span>`;
}

function renderOmicsDashboard(result) {
  if (!result.model_details?.algorithm || !result.test_performance?.confusion_matrix) {
    throw new Error("The dashboard API is out of date. Restart the FastAPI server to load the latest multi-omics metadata and report routes, then run the analysis again.");
  }
  state.omicsResult = result;
  omicsPage = 0;
  omicsSort = { key: "sample", direction: 1 };
  const aliveCount = result.results.filter((row) => row.predicted_class === 0).length;
  const deceasedCount = result.results.length - aliveCount;
  const alivePercent = result.rows ? aliveCount / result.rows * 100 : 0;
  const deceasedPercent = result.rows ? deceasedCount / result.rows * 100 : 0;
  const details = result.model_details;
  const metrics = result.test_performance;
  const cards = [
    ["Total samples analysed", result.rows, "Uploaded CSV rows"],
    ["Predicted Alive", aliveCount, `${alivePercent.toFixed(1)}% of samples`],
    ["Predicted Deceased", deceasedCount, `${deceasedPercent.toFixed(1)}% of samples`],
    ["Selected model", result.model, details.algorithm],
  ];
  $("#omics-summary-cards").innerHTML = cards.map(([label, value, note]) => `<article class="omics-summary-card"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><small>${escapeHtml(note)}</small></article>`).join("");
  $("#omics-distribution").innerHTML = `
    <div class="distribution-row"><div class="distribution-label"><strong>Alive</strong><span>${aliveCount} samples · ${alivePercent.toFixed(1)}%</span></div><div class="distribution-track"><i class="alive-bar" style="width:${alivePercent}%"></i></div></div>
    <div class="distribution-row"><div class="distribution-label"><strong>Deceased</strong><span>${deceasedCount} samples · ${deceasedPercent.toFixed(1)}%</span></div><div class="distribution-track"><i class="deceased-bar" style="width:${deceasedPercent}%"></i></div></div>
    <p class="distribution-note">Counts reflect model predictions for this upload only.</p>`;
  const inputItems = [
    ["Uploaded filename", result.filename],
    ["Sample count", result.rows],
    ["Input features used", `${result.input_feature_count} of ${result.input_feature_count} required`],
    ["Required feature validation", result.required_feature_validation],
    ["Missing required features", result.missing_required_features.length ? result.missing_required_features.join(", ") : "None"],
    ["Analysis completion", "Complete"],
  ];
  $("#omics-input-information").innerHTML = inputItems.map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("");
  const modelItems = [
    ["Feature group", details.feature_group],
    ["Model algorithm", details.algorithm],
    ["Checkpoint filename", details.checkpoint_filename],
    ["Number of input features", details.input_features],
    ["Target variable", `${details.target_variable} (0=Alive, 1=Deceased)`],
    ["Class mapping", "0 = Alive · 1 = Deceased"],
    ["Samples analysed", details.samples_analysed],
  ];
  const preprocessing = Object.entries(details.preprocessing || {}).map(([name, value]) => `${name}: ${value}`).join("\n");
  modelItems.push(["Saved preprocessing pipeline", preprocessing || "Not recorded in saved metadata"]);
  $("#omics-model-information").innerHTML = modelItems.map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("");
  const performanceItems = [
    ["Accuracy", metrics.accuracy], ["Precision · Deceased", metrics.precision],
    ["Recall · Deceased", metrics.recall], ["Specificity · Alive", metrics.specificity],
    ["Macro-F1", metrics.macro_f1], ["ROC-AUC", metrics.roc_auc], ["PR-AUC", metrics.pr_auc],
  ];
  $("#omics-performance-metrics").innerHTML = performanceItems.map(([label, value]) => `<div><span>${escapeHtml(label)}</span><strong>${pct(value)}</strong></div>`).join("");
  const matrix = metrics.confusion_matrix;
  $("#omics-confusion-matrix").innerHTML = `<table><thead><tr><th></th><th>Predicted Alive</th><th>Predicted Deceased</th></tr></thead><tbody><tr><th>Actual Alive</th><td>${matrix[0][0]}</td><td>${matrix[0][1]}</td></tr><tr><th>Actual Deceased</th><td>${matrix[1][0]}</td><td>${matrix[1][1]}</td></tr></tbody></table><small>Saved held-out test set; not computed from uploaded samples.</small>`;
  $("#omics-interpretation-text").textContent = `This model classifies Dataset C recorded vital-status labels: 0 = Alive and 1 = Deceased. In this upload, it predicted ${aliveCount} Alive and ${deceasedCount} Deceased labels. These outputs do not diagnose cancer or predict survival time.`;
  $("#omics-disclaimer").textContent = result.disclaimer;
  $("#omics-dashboard").classList.remove("hidden");
  renderOmicsTable();
}

function downloadOmicsCsv() {
  const result = state.omicsResult;
  if (!result) return;
  const identifierHeader = result.sample_identifier_column || "Row number";
  const csvCell = (value) => `"${String(value ?? "").replace(/"/g, '""')}"`;
  const lines = [
    [identifierHeader, "Predicted vital status", "Predicted class (0=Alive, 1=Deceased)", "Alive probability", "Deceased probability", "Probability difference"].map(csvCell).join(","),
    ...result.results.map((row) => {
      const alive = Number(row.class_probabilities?.alive || 0);
      const deceased = Number(row.class_probabilities?.deceased || 0);
      return [
        row.sample_identifier == null ? `Row ${row.row}` : row.sample_identifier,
        row.predicted_vital_status,
        row.predicted_class,
        alive,
        deceased,
        Math.abs(alive - deceased),
      ].map(csvCell).join(",");
    }),
  ];
  const url = URL.createObjectURL(new Blob([lines.join("\r\n")], { type: "text/csv;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `${result.scope}_predictions.csv`;
  link.click();
  URL.revokeObjectURL(url);
}

async function downloadOmicsPdf() {
  if (!state.omicsResult) return;
  const button = $("#download-omics-pdf");
  button.disabled = true;
  button.textContent = "Preparing PDF…";
  try {
    const response = await fetch("/api/omics/report.pdf", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state.omicsResult),
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.detail || `PDF generation failed (${response.status})`);
    }
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement("a");
    link.href = url;
    link.download = `${state.omicsResult.scope}_detailed_analysis.pdf`;
    link.click();
    URL.revokeObjectURL(url);
  } catch (error) {
    status($("#omics-status"), `PDF report failed: ${error.message}`, "error");
  } finally {
    button.disabled = false;
    button.textContent = "Download Detailed Analysis PDF";
  }
}

async function loadOverview() {
  try {
    const [data, evaluations] = await Promise.all([
      jsonRequest("/api/overview"),
      jsonRequest("/imaging/api/evaluations"),
    ]);
    imagingOverview = data.imaging;
    imagingEvaluations = evaluations.evaluations || [];
    renderImageMetrics(data.imaging);
    const select = $("#evaluation-select");
    select.innerHTML = imagingEvaluations.map((entry) => `<option value="${escapeHtml(entry.id)}">${escapeHtml(entry.modality_label)} · ${escapeHtml(entry.experiment_identifier)}${entry.active_for_inference ? " · ACTIVE" : " · HISTORICAL"}</option>`).join("");
    const initial = `${$("#modality").value}_baseline`;
    if (imagingEvaluations.some((entry) => entry.id === initial)) select.value = initial;
    renderImageEvaluation();
    renderOmicsMetrics(data.omics);
    const scopeSelect = $("#omics-scope");
    [...scopeSelect.options].forEach((option) => {
      const model = data.omics.models.find((m) => m.scope === option.value);
      option.disabled = !model?.available;
      if (option.disabled) option.textContent += " · unavailable";
    });
    if (data.imaging.status !== "available") status($("#image-status"), "Imaging API could not load. Check the source project's local inference dependencies.", "error");
  } catch (error) {
    $("#image-metrics").innerHTML = `<div class="metric-intro"><strong>Model information unavailable</strong><small>${escapeHtml(error.message)}</small></div>`;
    status($("#omics-status"), `Dashboard data unavailable: ${error.message}`, "error");
  }
}

function clearImage() {
  state.imageFile = null;
  state.imageVerified = false;
  state.latestInspection = null;
  state.latestPrediction = null;
  $("#predict-image").disabled = true;
  $("#image-preview").classList.add("hidden");
  $("#image-empty").classList.remove("hidden");
  $("#image-result").classList.add("hidden");
  $("#verification-details").innerHTML = '<p class="detail-placeholder">Select an image to inspect format, dimensions, detector result, and validation status.</p>';
  $("#prediction-interpretation").innerHTML = '<p class="detail-placeholder">The model score comparison and distance to its decision boundary will appear after prediction.</p>';
  $("#gradcam-content").innerHTML = '<p class="detail-placeholder">Run an accepted image prediction to request a Grad-CAM heatmap.</p>';
  $("#model-information").innerHTML = '<p class="detail-placeholder">Model metadata will appear with the prediction result.</p>';
}

async function selectImage(file) {
  clearImage();
  status($("#image-status"));
  if (!file) return;
  const allowed = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"];
  const suffix = `.${file.name.split(".").pop().toLowerCase()}`;
  if (!allowed.includes(suffix)) return status($("#image-status"), "Unsupported image format.", "error");
  if (file.size > 20 * 1024 * 1024) return status($("#image-status"), "Image exceeds the 20 MB limit.", "error");
  state.imageFile = file;
  state.latestInspection = null;
  if (state.imageObjectUrl) URL.revokeObjectURL(state.imageObjectUrl);
  state.imageObjectUrl = URL.createObjectURL(file);
  const img = $("#preview-img");
  img.src = state.imageObjectUrl;
  $("#preview-name").textContent = file.name;
  $("#preview-dims").textContent = `${(file.size / 1024).toFixed(0)} KB`;
  $("#verify-state").textContent = "Checking image and modality…";
  $("#verify-state").classList.remove("good");
  $("#image-preview").classList.remove("hidden");
  renderVerification({ file, statusText: "PENDING", validation: "Checking image format and decoding" });
  status($("#image-status"), "Verifying modality using the saved independent detector…");
  const body = new FormData(); body.append("file", file);
  try {
    const inspected = await jsonRequest("/api/imaging/inspect", { method: "POST", body });
    state.latestInspection = inspected;
    $("#preview-dims").textContent = `${inspected.width} × ${inspected.height} px · ${(file.size / 1024).toFixed(0)} KB`;
    const selected = $("#modality").value;
    const matches = inspected.verified && inspected.detected_modality === selected;
    state.imageVerified = matches;
    const verify = $("#verify-state");
    verify.textContent = inspected.verified
      ? `Detected: ${inspected.detected_modality} · ${(inspected.confidence * 100).toFixed(1)}%${matches ? " · matches" : " · mismatch"}`
      : `${inspected.raw_modality} · uncertain / unsupported (${(inspected.confidence * 100).toFixed(1)}%)`;
    verify.classList.toggle("good", matches);
    $("#predict-image").disabled = !matches;
    renderVerification({ file, inspected, statusText: matches ? "PASSED" : "REJECTED", validation: "Passed format and image decoding checks" });
    status($("#image-status"), matches ? "Detector approved this modality. Prediction is available." : "Modality mismatch or low confidence. Prediction is blocked.", matches ? "success" : "error");
  } catch (error) {
    $("#verify-state").textContent = "Could not verify modality";
    renderVerification({ file, statusText: "REJECTED", validation: `Failed validation or detector check: ${error.message}` });
    status($("#image-status"), error.message, "error");
  }
}

$("#image-file").addEventListener("change", (event) => selectImage(event.target.files[0]));
$("#modality").addEventListener("change", () => {
  renderImageEvaluation();
  const baseline = `${$("#modality").value}_baseline`;
  if (imagingEvaluations.some((entry) => entry.id === baseline)) {
    $("#evaluation-select").value = baseline;
    renderImageEvaluation();
  }
  if (state.imageFile) selectImage(state.imageFile);
});
$("#evaluation-select").addEventListener("change", renderImageEvaluation);
const dropzone = $("#dropzone");
for (const eventName of ["dragenter", "dragover"]) dropzone.addEventListener(eventName, (event) => { event.preventDefault(); dropzone.classList.add("drag"); });
for (const eventName of ["dragleave", "drop"]) dropzone.addEventListener(eventName, (event) => { event.preventDefault(); dropzone.classList.remove("drag"); });
dropzone.addEventListener("drop", (event) => selectImage(event.dataTransfer.files[0]));

$("#predict-image").addEventListener("click", async () => {
  if (!state.imageFile || !state.imageVerified) return;
  const button = $("#predict-image"); button.disabled = true; button.firstChild.textContent = "Analysing… ";
  status($("#image-status"), "The inference API will verify modality again before classifier execution.");
  const body = new FormData(); body.append("file", state.imageFile);
  try {
    const result = await jsonRequest(`/imaging/api/predict?modality=${encodeURIComponent($("#modality").value)}`, { method: "POST", body });
    renderImagePrediction(result);
    status($("#image-status"), "Analysis complete. This result is research output only.", "success");
  } catch (error) {
    const gateRejected = error.payload?.code === "MODALITY_REJECTION" || /modality verification is unavailable/i.test(error.message);
    renderVerification({ file: state.imageFile, inspected: state.latestInspection, statusText: gateRejected ? "REJECTED" : "PASSED", validation: gateRejected ? `Prediction blocked: ${error.message}` : `Image checks passed; classifier failed: ${error.message}` });
    status($("#image-status"), error.message, "error"); button.disabled = false;
  }
  button.firstChild.textContent = "Verify and analyse ";
});

$("#omics-file").addEventListener("change", (event) => {
  state.omicsFile = event.target.files[0] || null;
  state.omicsResult = null;
  $("#omics-dashboard").classList.add("hidden");
  $("#omics-file-label").textContent = state.omicsFile?.name || "Select a feature CSV";
  $("#omics-file-hint").textContent = state.omicsFile ? `${(state.omicsFile.size / 1024).toFixed(0)} KB · Required feature columns will be checked.` : "Expected columns are checked from the chosen model metadata.";
  $("#predict-omics").disabled = !state.omicsFile;
  $("#omics-result").classList.add("hidden"); $("#omics-empty").classList.remove("hidden");
  status($("#omics-status"));
});

$("#omics-scope").addEventListener("change", () => {
  state.omicsResult = null;
  $("#omics-dashboard").classList.add("hidden");
  $("#omics-result").classList.add("hidden"); $("#omics-empty").classList.remove("hidden");
});

$("#predict-omics").addEventListener("click", async () => {
  if (!state.omicsFile) return;
  const button = $("#predict-omics"); button.disabled = true; button.firstChild.textContent = "Running pipeline… ";
  status($("#omics-status"), "Validating required columns and applying saved pipeline…");
  const body = new FormData(); body.append("file", state.omicsFile);
  try {
    const result = await jsonRequest(`/api/omics/predict?scope=${encodeURIComponent($("#omics-scope").value)}`, { method: "POST", body });
    renderOmicsDashboard(result);
    $("#omics-result").innerHTML = `<div class="omics-run-summary"><span class="step">${escapeHtml(result.required_feature_validation)} FEATURE CHECK</span><strong>${result.rows} samples analysed</strong><small>${escapeHtml(result.model)} · ${result.used_features} required features · historical test performance shown separately below</small></div>`;
    $("#omics-empty").classList.add("hidden"); $("#omics-result").classList.remove("hidden");
    status($("#omics-status"), "Analysis complete. Labels refer to Dataset C recorded vital status.", "success");
  } catch (error) { status($("#omics-status"), error.message, "error"); }
  button.disabled = false; button.firstChild.textContent = "Run outcome analysis ";
});

$("#omics-table-search").addEventListener("input", () => { omicsPage = 0; renderOmicsTable(); });
document.querySelector(".omics-detail-table thead").addEventListener("click", (event) => {
  const button = event.target.closest("[data-omics-sort]");
  if (!button) return;
  const key = button.dataset.omicsSort;
  omicsSort = { key, direction: omicsSort.key === key ? -omicsSort.direction : 1 };
  renderOmicsTable();
});
$("#omics-previous-page").addEventListener("click", () => { omicsPage = Math.max(0, omicsPage - 1); renderOmicsTable(); });
$("#omics-next-page").addEventListener("click", () => { omicsPage += 1; renderOmicsTable(); });
$("#download-omics-csv").addEventListener("click", downloadOmicsCsv);
$("#download-omics-pdf").addEventListener("click", downloadOmicsPdf);

const topbar = document.querySelector(".topbar");
const navToggle = $("#nav-toggle");
const primaryNavigation = $("#primary-navigation");
navToggle.addEventListener("click", () => {
  const isOpen = topbar.classList.toggle("nav-open");
  navToggle.setAttribute("aria-expanded", String(isOpen));
  navToggle.setAttribute("aria-label", isOpen ? "Close navigation" : "Open navigation");
});
primaryNavigation.addEventListener("click", (event) => {
  if (!event.target.closest("a")) return;
  topbar.classList.remove("nav-open");
  navToggle.setAttribute("aria-expanded", "false");
  navToggle.setAttribute("aria-label", "Open navigation");
});
const navigationSections = ["home", "imaging", "omics", "research"].map((id) => document.getElementById(id));
const navigationObserver = new IntersectionObserver((entries) => {
  const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
  if (!visible) return;
  primaryNavigation.querySelectorAll("a").forEach((link) => {
    const active = link.hash === `#${visible.target.id}`;
    link.classList.toggle("active", active);
    if (active) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
}, { rootMargin: "-25% 0px -60% 0px", threshold: [0, 0.1, 0.25] });
navigationSections.forEach((section) => navigationObserver.observe(section));

loadOverview();
