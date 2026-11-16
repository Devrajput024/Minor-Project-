# Project audit

Audit date: 2026-11-16. The workspace contains the source projects as `Breast Cncer Project/` (imaging) and `Breast part 2/` (multi-omics), rather than the prompt's proposed `Medical_Imaging/` and `Multi_Omics/` names. No `combined_app/` existed at audit start. This audit records the actual local workspace; saved artifacts and source reports were inspected before application implementation.

## Medical imaging

### Important files and implementation

- `Breast Cncer Project/app/main.py`: FastAPI inference API, file validation, detector guard, ResNet-18 loading, preprocessing, `/api/model-info`, `/api/predict`, and health route.
- `Breast Cncer Project/app/modality_detector.py`: separate ResNet-18 modality classifier; classes are ultrasound, histopathology, chest_xray, other; checkpoint threshold is read from artifact.
- `Breast Cncer Project/app/templates/index.html`, `app/static/styles.css`, `app/static/app.js`: existing imaging UI.
- `MSI_GPU_Training_Package/train_breast_cancer.py`, `predict.py`, `train_modality_detector.py`, `evaluate_models.py`, `evaluate_external_modality_detector.py`, `feature_audit.py`: training, inference, detector, evaluation, and explainability implementations. `feature_audit.py` has an existing Grad-CAM implementation targeting `model.layer4[-1].conv2`; it writes research audit images and was not connected to live predictions before this dashboard update.
- `tests/test_backend_modality_guard.py`: tests ensure cross-modality, X-ray, and unrelated uploads are rejected before cancer inference.
- `research_paper_evidence/00_START_HERE.md`, `01_PROJECT_AND_METHODOLOGY.md`, `02_DATASET_AND_SPLIT_AUDIT.md`, `03_VERIFIED_RESULTS.csv`, `04_RESULTS_ANALYSIS.md`, `05_INTERPRETABILITY_AND_AUDITS.md`, `06_APPLICATION_AND_TESTS.md`: project evidence and result provenance.
- `MSI_GPU_Training_Package/requirements.txt`; the existing app README gives a uvicorn launch command. Dataset manifests/splits are under `dataset_manifests/` and `msi_results_full/`.

### Data and model artifacts

`MultiModel Breast Cancer MSI Dataset/` contains 3,052 images: ultrasound 806 (406 benign, 400 malignant); histopathology 1,246 (623 benign, 623 malignant); chest X-ray 1,000 (500 Normal, 500 Malignant). The training dataset class conventions are benign/malignant for ultrasound and histopathology, and Normal/Malignant for X-ray. The validation/test split details are recorded in manifests and audit reports. `datasets/` is empty; the images are in the named dataset directory.

Saved cancer-model weights are present in `msi_results_full/` for all three modalities, in `outputs/`, and in experiment/checkpoint folders. The current audited app intentionally loads only the preserved baseline ultrasound and histopathology checkpoints (`ultrasound_best_resnet18.pt`, `histopathology_best_resnet18.pt`); chest X-ray is deliberately unavailable to live inference. Additional artifacts include `improved_experiments/ultrasound/best_checkpoint.pt`, histopathology experiment checkpoints, and four chest-X-ray experimental `best_model.pt` files. They are not interchangeable with the baseline without updating the model and evaluation configuration. The independent detector checkpoint `modality_detector/modality_resnet18.pt` is present.

The cancer models are torchvision ResNet-18 with a two-class output. Existing backend checks checkpoint modality and exact class mapping `{benign: 0, malignant: 1}`, then uses checkpoint `image_size` (default 224), RGB conversion, resize, tensor conversion, and checkpoint mean/std normalization. Detector input is RGB resized to 112×112 and ImageNet normalized. Detector confidence threshold is loaded from its checkpoint (the saved report says 0.7413211465). Unsupported extensions are rejected; accepted suffixes are jpg/jpeg/png/bmp/tif/tiff/webp; corrupt and empty images are rejected.

### Evaluations and limitations

Verified saved test results in `research_paper_evidence/03_VERIFIED_RESULTS.csv`:

| Modality/model | Split n | Accuracy | Macro F1 | Malignant recall | Specificity | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Ultrasound baseline ResNet-18 | test 166 | 0.8614 | 0.8608 | 0.9059 | 0.8148 | 0.9089 |
| Histopathology baseline ResNet-18 | test 250 | 0.7600 | 0.7596 | 0.6569 | 0.8850 | 0.8594 |
| Chest X-ray baseline ResNet-18 | test 200 | 0.4900 | 0.4838 | 0.6250 | 0.4684 | 0.5468 |
| Chest X-ray later experimental winner | test 200 | 0.5700 | 0.5689 | 0.5417 | 0.5962 | 0.6034 |

The UI/backend expose only ultrasound and histopathology because the X-ray results are weak and the existing app explicitly excludes X-ray from reliable inference. Detector internal test accuracy is 0.9984 on 628 images, with 12/12 small held-out chart images rejected and zero wrong-modality confident acceptances at the selected threshold. External detector evaluations report lower top-1 accuracy (0.9127 on 779 unique BUSI ultrasound images accepted as expected 490; 0.9200 on 1,000 histology patches accepted as expected 697) and some confident wrong-class predictions; rejection is a safety gate, not proof of universal modality recognition. Histology external data are patches, not whole-slide images. Dataset/split audit and feature-mask reports exist under `audit_review/`, `reproducible_evaluation_plan/`, and `msi_feature_audit/`.

### Dependencies and readiness

Training requirements specify torch >=2.0, torchvision >=0.15, scikit-learn >=1.2, pandas >=2.0, numpy >=1.24, matplotlib >=3.7, Pillow >=9.5. The source `.venv` contains FastAPI 0.161.1, uvicorn 0.54.0, scikit-learn 1.9.1, pandas 3.1.0, NumPy 2.5.4, joblib 1.6.0, torch 2.14.1+cpu, torchvision 0.29.1+cpu, and Pillow 13.0.0. `.venv311` has scikit-learn 1.9.1 and was used to check model artifact loading; it does not have FastAPI. Runtime load check: in the source `.venv`, the existing backend loaded the preserved ultrasound and histopathology ResNet-18 checkpoints and confirmed both class maps are `{benign: 0, malignant: 1}`. The modality detector also loaded and reported threshold 0.7413211465. In `.venv311`, all five final `.joblib` artifacts loaded as sklearn `Pipeline` objects and reported classes `[0, 1]`; their JSON metadata lists 860 CNV, 604 RNA, 249 mutation, 223 protein, and 1,936 integrated input features, respectively. Chest X-ray is not enabled. Existing app is reusable as an independent API; integration should preserve its gating and its research-only disclaimer.

## Multi-omics

### Important files and data

- `Breast part 2/dataset_c_extracted/data.csv`: primary numeric Dataset C; 705 rows × 1,937 columns (1,936 predictors plus target), 0 missing cells, 0 duplicate rows.
- `dataset_c_extracted/brca_data_w_subtypes.csv`: 705 × 1,941, 1,937 numeric and 4 nonnumeric columns, 389 missing cells.
- Dataset docs and audits: `results_dataset_C/dataset_validation.md`, `dataset_c_audit/`, `multiomics_dataset_audit/`, `new_dataset_audit/`, `dataset_c_recovery/`, and `DATASET_C_COMPLETE_EXPERIMENT_AUDIT.md`.
- Training/reproduction: `dataset_C_reproducibility.py`, `results_dataset_C/final_optimization/run_optimization.py`, plus scripts under `results_dataset_C/final_research/reproducibility/`.
- Selected final model artifacts and metadata: `results_dataset_C/final_optimization/models/`.
- Evaluation: `results_dataset_C/final_optimization/tables/selected_scope_models_test_metrics.csv`, `nested_cv_summary.csv`, final research tables/figures, predictions, confusion matrices, and reports.

Dataset C target is recorded vital status: `0=alive`, `1=deceased`; this is exploratory classification, not survival-time prediction and not breast-cancer diagnosis. There are 611 alive and 94 deceased samples. Cohort is split into 564 training and 141 held-out test rows (19 deceased in test). Patient IDs/follow-up durations are unavailable, so patient-level independence and survival outcomes cannot be confirmed. Feature groups are prefixes `rs_` (RNA), `cn_` (CNV), `mu_` (mutation), and `pp_` (protein); scope definitions and selected columns are in per-model JSON and audit/feature-list files. Main CSV has 705 samples and 1,936 input features; use model metadata to enforce exact input columns.

### Models, preprocessing, metrics

Five final pipelines are present as `.joblib` plus matching `.json`: CNV Logistic Regression, RNA Random Forest, Mutation Logistic Regression, Protein Random Forest, and integrated all-four-omics Random Forest. Model metadata includes exact ordered feature names, class mapping, seed 20261110, estimator parameters, train/test indices, and preprocessing. Pipelines perform scope-specific imputation (CNV most-frequent; others median), CNV one-hot encoding with unknown categories ignored, RNA/protein scaling, mutation variance filtering, and a `ColumnTransformer` inside the model pipeline. Training used nested stratified 3×3 CV on training data, then selected scope models were refit on 564 training rows and evaluated once on the 141-row holdout. Do not refit or alter the study method in the app.

Held-out metrics from `selected_scope_models_test_metrics.csv`:

| Scope/model | Accuracy | Balanced accuracy | Deceased recall | Specificity | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| CNV Logistic Regression | 0.7234 | 0.4847 | 0.1579 | 0.8115 | 0.5345 | 0.1503 |
| RNA Random Forest (CV global winner) | 0.8794 | 0.6415 | 0.3158 | 0.9672 | 0.5794 | 0.3484 |
| Mutation Logistic Regression | 0.6879 | 0.4642 | 0.1579 | 0.7705 | 0.3904 | 0.1222 |
| Protein Random Forest | 0.8582 | 0.5181 | 0.0526 | 0.9836 | 0.4707 | 0.1684 |
| Integrated Random Forest | 0.8865 | 0.6456 | 0.3158 | 0.9754 | 0.6053 | 0.3561 |

The report documents a dummy majority baseline with 0.865 accuracy but 0 deceased recall and 0.5 ROC-AUC. The test set has only 19 deceased cases. The integrated model's small descriptive advantage on this holdout is not evidence of a validated integration benefit; selected overall winner by training-only composite nested CV was RNA Random Forest. No independent external validation. Feature importance is model-dependent association, not causality or validated biomarkers. Missing optional packages: imbalanced-learn, SHAP, XGBoost, LightGBM; they were not installed or used. A separate Dataset C reproducibility package has only three `.joblib` checkpoints; the newer final-optimization directory contains the five final models used for this audit.

Dependency versions recorded in final optimization metadata: Python 3.11.9, scikit-learn 1.9.1, NumPy 2.4.6, pandas 3.1.1, joblib 1.6.0, matplotlib 3.11.2, threadpoolctl 3.8.0. `Breast part 2/requirements.txt` pins those packages.

## Blockers and integration readiness

- Both source directories contain functional research assets, but no prompt-named `Medical_Imaging/` or `Multi_Omics/`; application references must point to the actual local paths and remain configurable.
- Imaging has model weights and a detector; only ultrasound/histopathology baseline classifiers are enabled by the existing backend. X-ray inference should remain unavailable.
- Multi-omics final models have actual checkpoints and schema JSON. Inference can be enabled for CSV uploads only when required columns, numeric conversion, and model metadata validate; reject schema mismatch or missing features instead of filling unobserved molecular values with fabricated defaults.
- These modules address different targets and datasets. They must stay independent and must not be presented as a clinically validated joint model.
- Both branches are educational/research work only. Their reports explicitly state no clinical validation and document major data/split/cohort limitations.
