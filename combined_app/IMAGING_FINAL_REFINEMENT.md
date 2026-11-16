# Medical Imaging Final Refinement

## Evaluation provenance

The live imaging API uses only the preserved ultrasound and histopathology checkpoints in `Breast Cncer Project/msi_results_full/`. Each active evaluation below is loaded from the metrics JSON beside that exact checkpoint. Confusion matrix order is actual class rows and predicted class columns, with class 0 first.

| Evaluation | Status | Test n | Accuracy | Precision (malignant) | Recall / sensitivity | Specificity | Macro F1 | ROC-AUC | Confusion matrix |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Ultrasound preserved baseline | Active | 166 | 0.8614 | 0.8370 | 0.9059 | 0.8148 | 0.8608 | 0.9089 | `[[66,15],[8,77]]` |
| Histopathology preserved baseline | Active | 250 | 0.7600 | 0.8738 | 0.6569 | 0.8850 | 0.7596 | 0.8594 | `[[100,13],[47,90]]` |
| Chest X-ray preserved baseline | Historical; inference disabled | 200 | 0.4900 | 0.4762 | 0.6250 | 0.3654 | 0.4838 | 0.5468 | `[[38,66],[36,60]]` |
| Ultrasound improved experiment | Historical; separate checkpoint | 166 | 0.8373 | 0.9677 | 0.7059 | 0.9753 | 0.8352 | 0.9497 | `[[79,2],[25,60]]` |
| Histopathology experiment 001 | Historical; separate split/checkpoint | 214 | 0.6589 | 0.9451 | 0.5584 | 0.9167 | 0.6516 | 0.8190 | `[[55,5],[68,86]]` |
| Chest X-ray Exp4 winner | Historical; inference disabled | 200 | 0.5700 | 0.5532 | 0.5417 | 0.5962 | 0.5689 | 0.6034 | `[[62,42],[44,52]]` |
| Ultrasound saved evaluation without checkpoint provenance | Historical; unmatched | 102 | 0.8431 | 0.5676 | 1.0000 | 0.8025 | 0.8073 | 1.0000 | `[[65,16],[0,21]]` |

The unmatched 102-image ultrasound result is kept separate because its checkpoint and split provenance are not recorded in the evaluation artifact. It is not shown as active model performance.

The chest X-ray baseline has a source discrepancy: its checkpoint-adjacent JSON and confusion matrix give specificity 0.3654, while `research_paper_evidence/03_VERIFIED_RESULTS.csv` records 0.4684 for the same named baseline. The displayed historical row uses only the checkpoint-adjacent JSON and marks the conflict; it remains unresolved.

## Curves

Saved ROC images matched to a named historical run are available for the improved ultrasound experiment, histopathology experiment 001, chest X-ray Exp4, and the unmatched 102-image ultrasound evaluation. The active preserved ultrasound and histopathology baselines have no ROC plot or curve points matched to those exact checkpoints and test splits. No PR curve artifacts were found for these runs. The UI and evaluation PDFs state this rather than reusing another experiment's curve.

## Model information

Both active checkpoints were loaded from `msi_results_full/`. They are torchvision ResNet-18 models with 224×224 RGB inputs and class mapping `{benign: 0, malignant: 1}`. The preprocessing follows the existing backend: RGB conversion, resize to 224×224, tensor conversion, and ImageNet channel normalization. Inference device and actual PyTorch/torchvision versions are reported at runtime. The checkpoint contains no explicit version or experiment ID; the interface reports that field as unavailable.

The live classifier uses two-class softmax argmax, equivalent to a 0.50 malignant-score boundary; ties resolve by class index order. Scores are not calibrated cancer probabilities. The independent modality detector remains a fail-closed gate before classifier execution.

## Downloadable reports and routes

- `POST /imaging/api/reports/prediction.pdf`: individual uploaded-image prediction report with verification, scores, interpretation, model details, UTC timestamp, original image, and Grad-CAM images when produced.
- `GET /imaging/api/reports/evaluation/{evaluation_id}.pdf`: checkpoint/run-specific performance report with dataset/split information, metrics, confusion matrix, available matched curves, and limitations.
- `GET /imaging/api/evaluations`: catalog used by the dashboard. Active and historical runs are labeled separately.

## Validation

Run from `Breast Cncer Project/`:

```powershell
& '.venv/Scripts/python.exe' -m pytest -p no:cacheprovider -q tests/test_backend_modality_guard.py tests/test_final_refinement.py
```

Final result: **20 passed**. The covered cases include real held-out ultrasound/histopathology predictions, scores and model metadata, Grad-CAM when available, unsupported and mismatched uploads, missing detector and classifier checkpoint failure, evaluation provenance separation, and both PDF downloads. One upstream Starlette/httpx deprecation warning remains.
