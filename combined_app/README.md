# AI-Driven Breast Cancer Analysis

Unified local dashboard for the workspace's medical imaging and Dataset C multi-omics research. The modules remain independent: imaging classifies image dataset labels; Dataset C models classify recorded vital status. This is an educational research application, not a clinical tool.

## Start

The imaging project already has a local environment at `.venv/`. From the workspace root, use it to start the dashboard:

```powershell
& '.\Breast Cncer Project\.venv\Scripts\python.exe' -m uvicorn combined_app.main:app --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080`. This workspace's `.venv/` includes FastAPI, multipart support, PyTorch, and scikit-learn 1.9.1. The application reads the source projects from the sibling directories `Breast Cncer Project/` and `Breast part 2/`; it does not copy or retrain their artifacts. For a new environment, install dependencies from `requirements.txt` (PyTorch may need a device-specific build). If either source directory is renamed or moved, update `IMAGING_ROOT` and `OMICS_ROOT` in `main.py`.

## Modules

- **Medical imaging** reuses the imaging project's FastAPI inference app and its detector/checkpoint guard. The dashboard supports ultrasound and histopathology only. The upload inspection endpoint returns image dimensions and detector confidence; predictions are enabled only when the selected modality matches a confident supported detector result. Chest X-ray remains excluded by the source app.
- **Multi-omics** accepts CSV rows with exact named feature columns from the selected saved Dataset C model metadata. Missing required columns are rejected. Partial missing assay values remain missing and are handled by the saved pipeline's own imputation. Uploads support up to 1,000 rows and 20 MB. Results include per-sample labels and model scores, an upload-specific distribution, input validation details, and saved model metadata. When the input has an explicit sample identifier column it is preserved; otherwise rows are labelled by row number.
- Uploaded-sample predictions are kept separate from historical held-out test performance. The dashboard does not calculate accuracy on prediction-only uploads. Prediction CSV and detailed PDF downloads contain the actual model outputs; scores are not calibrated clinical risk estimates.

The omics target is `0=alive`, `1=deceased` in the Dataset C cohort. It is not a breast-cancer diagnosis or survival-time prediction. See [PROJECT_AUDIT.md](PROJECT_AUDIT.md) for discovered data, checkpoint, metric, dependency, and limitation details.

## Routes

- `GET /api/health` — service availability
- `GET /api/overview` — model descriptions and saved test metrics
- `POST /api/imaging/inspect` — independently detect image modality and dimensions
- `POST /imaging/api/predict?modality=ultrasound|histopathology` — existing guarded inference API
- `GET /imaging/api/evaluations` — checkpoint-matched active and separately labelled historical imaging runs
- `GET /imaging/api/reports/evaluation/{evaluation_id}.pdf` — downloadable evaluation report
- `POST /imaging/api/reports/prediction.pdf` — individual result PDF (multipart `file` and `result_json`)
- `POST /api/omics/predict?scope=cnv|rna|mutation|protein|integrated` — saved Dataset C pipeline prediction; multipart field is `file`
- `POST /api/omics/report.pdf` — downloadable prediction report from a completed multi-omics result payload

## Research limitations

The two analyses use separate cohorts, models, and targets. Their outputs are not combined. Imaging checkpoints and omics models have limited held-out evaluation and no clinical validation. Dataset C has only 19 deceased samples in the 141-row test partition and lacks patient IDs and follow-up duration. Results should be interpreted as exploratory model outputs only.
