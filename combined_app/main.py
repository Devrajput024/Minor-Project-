from __future__ import annotations

import io
import json
import sys
import importlib
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
IMAGING_ROOT = WORKSPACE / "Breast Cncer Project"
OMICS_ROOT = WORKSPACE / "Breast part 2"
OMICS_MODELS = OMICS_ROOT / "results_dataset_C" / "final_optimization" / "models"
OMICS_METRICS = OMICS_ROOT / "results_dataset_C" / "final_optimization" / "tables" / "selected_scope_models_test_metrics.csv"
sys.path.insert(0, str(IMAGING_ROOT))

try:
    imaging_backend = importlib.import_module("app.main")
    imaging_app = imaging_backend.app
except Exception as exc:  # Do not start a partially safe imaging path.
    imaging_backend = None
    imaging_app = None
    IMAGING_IMPORT_ERROR = str(exc)
else:
    IMAGING_IMPORT_ERROR = None

app = FastAPI(
    title="AI-Driven Breast Cancer Analysis",
    description="Independent research modules for imaging and Dataset C multi-omics analysis.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="combined-static")
if imaging_app is not None:
    app.mount("/imaging", imaging_app, name="imaging-service")

MODEL_FILES = {
    "cnv": ("cnv__logistic_regression.joblib", "cnv__logistic_regression.json"),
    "rna": ("rna__random_forest.joblib", "rna__random_forest.json"),
    "mutation": ("mutation__logistic_regression.joblib", "mutation__logistic_regression.json"),
    "protein": ("protein__random_forest.joblib", "protein__random_forest.json"),
    "integrated": ("integrated__random_forest.joblib", "integrated__random_forest.json"),
}
MODEL_LABELS = {
    "cnv": "CNV · Logistic Regression",
    "rna": "RNA · Random Forest (CV selected winner)",
    "mutation": "Mutation · Logistic Regression",
    "protein": "Protein · Random Forest",
    "integrated": "Integrated four-omics · Random Forest",
}
MODEL_CACHE: dict[str, tuple[Any, dict[str, Any]]] = {}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VITAL_STATUS_TARGET = "Dataset C recorded vital status"
VITAL_STATUS_MAPPING = {"0": "Alive", "1": "Deceased"}
SAMPLE_ID_COLUMNS = (
    "id", "sample", "sampleid", "sampleidentifier", "patientid", "subjectid", "participantid",
    "caseid", "specimenid", "recordid",
)


def load_omics_model(scope: str) -> tuple[Any, dict[str, Any]]:
    if scope not in MODEL_FILES:
        raise HTTPException(400, "Unknown model scope.")
    if scope not in MODEL_CACHE:
        model_name, metadata_name = MODEL_FILES[scope]
        model_path, metadata_path = OMICS_MODELS / model_name, OMICS_MODELS / metadata_name
        if not model_path.is_file() or not metadata_path.is_file():
            raise HTTPException(503, f"Saved {scope} model or its feature metadata is missing.")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("scope") != scope or not metadata.get("features"):
                raise ValueError("Model metadata is incomplete or mismatched.")
            model = joblib.load(model_path)
        except Exception as exc:
            raise HTTPException(503, f"Could not load the saved {scope} model: {exc}") from exc
        MODEL_CACHE[scope] = (model, metadata)
    return MODEL_CACHE[scope]


def load_omics_test_metrics(scope: str) -> dict[str, Any]:
    if not OMICS_METRICS.is_file():
        raise HTTPException(503, "Saved held-out Dataset C metrics are missing.")
    try:
        metrics = pd.read_csv(OMICS_METRICS).set_index("scope")
        if scope not in metrics.index:
            raise ValueError(f"No held-out metrics are recorded for {scope}.")
        row = metrics.loc[scope]
        return {
            "accuracy": float(row["accuracy"]),
            "precision": float(row["precision_deceased"]),
            "recall": float(row["deceased_recall"]),
            "specificity": float(row["specificity"]),
            "macro_f1": float(row["macro_f1"]),
            "roc_auc": float(row["roc_auc"]),
            "pr_auc": float(row["pr_auc"]),
            "confusion_matrix": [
                [int(row["tn"]), int(row["fp"])],
                [int(row["fn"]), int(row["tp"])],
            ],
            "confusion_matrix_labels": ["Alive", "Deceased"],
            "source": "Saved held-out test evaluation; separate from uploaded-sample predictions.",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, f"Could not read saved held-out {scope} metrics: {exc}") from exc


@app.get("/")
def home() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "imaging_available": imaging_app is not None,
        "imaging_error": IMAGING_IMPORT_ERROR,
        "omics_models_present": sum((OMICS_MODELS / files[0]).is_file() for files in MODEL_FILES.values()),
    }


@app.get("/api/overview")
def overview() -> dict[str, Any]:
    imaging_info: dict[str, Any] = {"status": "unavailable", "modalities": []}
    if imaging_app is not None:
        try:
            imaging_info = imaging_backend.model_info()
            imaging_info["status"] = "available"
        except Exception as exc:
            imaging_info = {"status": "unavailable", "error": str(exc), "modalities": []}
    metrics = {}
    if OMICS_METRICS.is_file():
        try:
            metrics = pd.read_csv(OMICS_METRICS).set_index("scope").to_dict(orient="index")
        except Exception:
            metrics = {}
    return {
        "imaging": imaging_info,
        "omics": {
            "dataset": "Dataset C",
            "samples": 705,
            "predictors": 1936,
            "target": "Recorded vital status (0=alive, 1=deceased)",
            "train_samples": 564,
            "test_samples": 141,
            "models": [
                {"scope": scope, "label": MODEL_LABELS[scope], "features": len(json.loads((OMICS_MODELS / files[1]).read_text(encoding="utf-8")).get("features", [])) if (OMICS_MODELS / files[1]).is_file() else None, "available": (OMICS_MODELS / files[0]).is_file()}
                for scope, files in MODEL_FILES.items()
            ],
            "test_metrics": metrics,
            "global_cv_winner": "RNA Random Forest",
        },
    }


@app.post("/api/imaging/inspect")
async def inspect_image(file: UploadFile = File(...)) -> dict[str, Any]:
    """Preview facts and independent detector result; never run cancer inference here."""
    if imaging_app is None:
        raise HTTPException(503, f"Imaging service could not be loaded: {IMAGING_IMPORT_ERROR}")
    filename = file.filename or "upload"
    if Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(400, "Unsupported image format. Use JPG, PNG, BMP, TIFF, or WEBP.")
    payload = await file.read(20 * 1024 * 1024 + 1)
    if not payload or len(payload) > 20 * 1024 * 1024:
        raise HTTPException(400 if not payload else 413, "Image is empty or exceeds the 20 MB upload limit.")
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
        with Image.open(io.BytesIO(payload)) as image:
            image_format = image.format or Path(filename).suffix.lstrip(".").upper()
            image = image.convert("RGB")
            width, height = image.size
            if width < 64 or height < 64:
                raise HTTPException(400, "Image must be at least 64 × 64 pixels.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"Image is corrupt or unreadable: {exc}") from exc
    try:
        import torch
        from app.modality_detector import load_modality_detector

        detector = load_modality_detector(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        detected, confidence, raw_class = detector.classify_many([image])[0]
    except Exception as exc:
        raise HTTPException(503, f"Modality verification is unavailable; image prediction is blocked: {exc}") from exc
    return {
        "filename": filename,
        "width": width,
        "height": height,
        "format": image_format,
        "detected_modality": detected,
        "raw_modality": raw_class,
        "confidence": round(float(confidence), 6),
        "verified": detected in {"ultrasound", "histopathology"},
        "message": "Modality verified." if detected in {"ultrasound", "histopathology"} else "Modality is uncertain or unsupported; cancer inference is blocked.",
    }


@app.post("/api/omics/predict")
async def omics_predict(scope: str, file: UploadFile = File(...)) -> dict[str, Any]:
    if scope not in MODEL_FILES:
        raise HTTPException(400, f"scope must be one of: {', '.join(MODEL_FILES)}")
    if Path(file.filename or "").suffix.lower() != ".csv":
        raise HTTPException(400, "Upload a CSV file with the model's named Dataset C features.")
    payload = await file.read(20 * 1024 * 1024 + 1)
    if len(payload) > 20 * 1024 * 1024:
        raise HTTPException(413, "CSV exceeds the 20 MB upload limit.")
    if not payload:
        raise HTTPException(400, "Uploaded CSV is empty.")
    try:
        frame = pd.read_csv(io.BytesIO(payload))
    except Exception as exc:
        raise HTTPException(400, f"Could not read CSV: {exc}") from exc
    if frame.empty or len(frame) > 1000:
        raise HTTPException(400, "CSV must contain 1–1000 sample rows.")
    model, metadata = load_omics_model(scope)
    features = metadata["features"]
    missing = [name for name in features if name not in frame.columns]
    if missing:
        preview = ", ".join(missing[:12]) + (" …" if len(missing) > 12 else "")
        raise HTTPException(422, f"CSV is missing {len(missing)} required {scope} feature columns: {preview}")
    data = frame.loc[:, features].copy()
    normalized_columns = {
        "".join(character for character in str(column).casefold() if character.isalnum()): column
        for column in frame.columns
    }
    identifier_column = next(
        (normalized_columns[name] for name in SAMPLE_ID_COLUMNS if name in normalized_columns),
        None,
    )
    # The trained pipelines expect numeric assay data. Coerce representation only;
    # keep absent/invalid cells as NaN so the saved pipeline applies its own imputer.
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    invalid_rows = data.isna().all(axis=1)
    if invalid_rows.any():
        raise HTTPException(422, f"Rows with no usable numeric {scope} values: {(invalid_rows[invalid_rows].index + 2).tolist()[:20]}")
    try:
        predicted = model.predict(data)
        probability = model.predict_proba(data) if hasattr(model, "predict_proba") else None
    except Exception as exc:
        raise HTTPException(422, f"The uploaded values could not be processed by the saved {scope} pipeline: {exc}") from exc
    classes = [str(value) for value in getattr(model, "classes_", [])]
    results = []
    for row_number, value in enumerate(predicted):
        identifier = None
        if identifier_column is not None:
            raw_identifier = frame.iloc[row_number][identifier_column]
            if not pd.isna(raw_identifier):
                identifier = str(raw_identifier)
        item: dict[str, Any] = {
            "row": row_number + 1,
            "sample_identifier": identifier,
            "predicted_vital_status": "deceased" if int(value) == 1 else "alive",
            "predicted_class": int(value),
        }
        if probability is not None:
            item["class_probabilities"] = {
                ("deceased" if str(label) == "1" else "alive" if str(label) == "0" else str(label)): round(float(score), 6)
                for label, score in zip(classes, probability[row_number])
            }
        results.append(item)
    model_filename, _ = MODEL_FILES[scope]
    return {
        "scope": scope,
        "model": MODEL_LABELS[scope],
        "filename": Path(file.filename or "uploaded.csv").name,
        "rows": len(results),
        "used_features": len(features),
        "input_feature_count": len(features),
        "missing_required_features": [],
        "required_feature_validation": "Passed",
        "sample_identifier_column": str(identifier_column) if identifier_column is not None else None,
        "model_details": {
            "feature_group": "Integrated four-omics" if scope == "integrated" else scope.upper(),
            "algorithm": metadata["model"].replace("_", " ").title(),
            "checkpoint_filename": model_filename,
            "input_features": len(features),
            "preprocessing": metadata.get("preprocessing", {}),
            "target_variable": VITAL_STATUS_TARGET,
            "class_mapping": VITAL_STATUS_MAPPING,
            "samples_analysed": len(results),
        },
        "test_performance": load_omics_test_metrics(scope),
        "target": "Dataset C recorded vital status (0=Alive, 1=Deceased)",
        "results": results,
        "disclaimer": "Exploratory research output only. These predictions classify Dataset C recorded vital-status labels (0=Alive, 1=Deceased); they do not diagnose cancer or predict survival time. Model scores are not calibrated clinical risk estimates.",
    }


@app.post("/api/omics/report.pdf")
async def omics_report(request: Request) -> StreamingResponse:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(400, f"Could not read report request: {exc}") from exc
    scope = payload.get("scope") if isinstance(payload, dict) else None
    rows = payload.get("results") if isinstance(payload, dict) else None
    if scope not in MODEL_FILES or not isinstance(rows, list) or not 1 <= len(rows) <= 1000:
        raise HTTPException(400, "Report requires a supported model and 1–1000 prediction rows.")
    _, metadata = load_omics_model(scope)
    test_metrics = load_omics_test_metrics(scope)
    normalized_rows = []
    for row in rows:
        if not isinstance(row, dict) or row.get("predicted_class") not in (0, 1):
            raise HTTPException(400, "Report contains an invalid prediction row.")
        probabilities = row.get("class_probabilities") or {}
        try:
            alive = float(probabilities["alive"])
            deceased = float(probabilities["deceased"])
            if not (0 <= alive <= 1 and 0 <= deceased <= 1):
                raise ValueError("probabilities must be between zero and one")
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(400, f"Report contains invalid class probabilities: {exc}") from exc
        normalized_rows.append({
            "row": int(row.get("row", len(normalized_rows) + 1)),
            "sample_identifier": row.get("sample_identifier"),
            "predicted_class": int(row["predicted_class"]),
            "alive": alive,
            "deceased": deceased,
        })

    alive_count = sum(row["predicted_class"] == 0 for row in normalized_rows)
    deceased_count = len(normalized_rows) - alive_count
    output = io.BytesIO()
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt

        with PdfPages(output) as pdf:
            fig, ax = plt.subplots(figsize=(11.7, 8.3))
            ax.axis("off")
            model_name = MODEL_LABELS[scope]
            filename = Path(str(payload.get("filename") or "uploaded.csv")).name
            summary_lines = [
                "AI-Driven Breast Cancer Analysis",
                "Multi-omics prediction report",
                "",
                f"Uploaded CSV: {filename}",
                f"Samples analysed: {len(normalized_rows)}",
                f"Selected model: {model_name}",
                f"Feature group: {'Integrated four-omics' if scope == 'integrated' else scope.upper()}",
                f"Input features: {len(metadata.get('features', []))}",
                f"Predicted Alive: {alive_count} ({alive_count / len(normalized_rows):.1%})",
                f"Predicted Deceased: {deceased_count} ({deceased_count / len(normalized_rows):.1%})",
                "",
                "Saved held-out test performance (historical evaluation; not this upload):",
                f"Accuracy {test_metrics['accuracy']:.3f} | Precision {test_metrics['precision']:.3f} | Recall {test_metrics['recall']:.3f}",
                f"Specificity {test_metrics['specificity']:.3f} | Macro-F1 {test_metrics['macro_f1']:.3f}",
                f"ROC-AUC {test_metrics['roc_auc']:.3f} | PR-AUC {test_metrics['pr_auc']:.3f}",
                f"Confusion matrix [Alive, Deceased]: {test_metrics['confusion_matrix']}",
                "",
                "Target: Dataset C recorded vital status (0=Alive, 1=Deceased).",
                "Predictions classify recorded vital-status labels; they do not diagnose cancer or predict survival time.",
                "Model probabilities are not calibrated clinical risk estimates. Research use only.",
            ]
            ax.text(0.06, 0.94, "\n".join(summary_lines), va="top", fontsize=11, linespacing=1.5, wrap=True)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            for start in range(0, len(normalized_rows), 28):
                batch = normalized_rows[start:start + 28]
                fig, ax = plt.subplots(figsize=(11.7, 8.3))
                ax.axis("off")
                identifier_label = payload.get("sample_identifier_column") or "Row number"
                table_rows = []
                for row in batch:
                    identifier = row["sample_identifier"]
                    table_rows.append([
                        str(identifier) if identifier is not None else f"Row {row['row']}",
                        VITAL_STATUS_MAPPING[str(row["predicted_class"])],
                        f"{row['alive']:.4f}",
                        f"{row['deceased']:.4f}",
                        f"{abs(row['alive'] - row['deceased']):.4f}",
                    ])
                table = ax.table(
                    cellText=table_rows,
                    colLabels=[identifier_label, "Predicted status", "Alive probability", "Deceased probability", "Probability difference"],
                    loc="center",
                    cellLoc="left",
                )
                table.auto_set_font_size(False)
                table.set_fontsize(7)
                table.scale(1, 1.45)
                ax.set_title(f"Individual predictions ({start + 1}-{start + len(batch)} of {len(normalized_rows)})", loc="left", pad=18)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Could not generate the detailed analysis PDF: {exc}") from exc
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{scope}_detailed_analysis.pdf"'},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=False)
