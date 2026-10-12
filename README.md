**English** · [Português](README.pt.md)

# Verifying a greenhouse registry with machine learning
### Automated validation of GIS-registered greenhouses against current imagery · Esposende–Vila do Conde Vulnerable Zone, Portugal

> Classical machine learning (Random Forest / Gradient Boosting) on
> spectral, textural and geometric features derived from RGB orthoimagery,
> to verify, polygon by polygon, whether greenhouses registered in a GIS
> platform since 2013 are still present on the ground — with a class
> probability for every decision, used to target field checks.

---
![Classified registry polygons](docs/classified_polygons.png)
*Registry polygons classified by the model over DGT ortoSat2023 imagery — greenhouse present (1); no longer present (0).*

![Feature importance](docs/feature_importance.png)
*Impurity-based feature importance of the final Random Forest model (10 selected features).*

![Confusion matrix](docs/confusion_matrix.png)
*Held-out test set: 39 of 41 polygons correctly classified.*

## Why verify

Since 2013, the teams monitoring the Esposende–Vila do Conde Vulnerable
Zone (*Zona Vulnerável*, ZV1) have been mapping greenhouse polygons in a
dedicated GIS platform (SIA-ZV), based on interviews with farmers. However,
the registry has never been systematically updated, either in the field or
at the desk, because doing so is time-consuming and costly. A decade later,
the registry and the territory had visibly diverged: structures had been
demolished, replaced, extended or newly built.

An outdated registry undermines everything built on top of it: pressure
mapping, compliance checks, and the cross-referencing of nitrate monitoring
data. Yet re-surveying hundreds of polygons manually was precisely the cost
the institution could not afford. The task called for an automated method
to check, efficiently, whether each registered greenhouse still exists.

## The approach

For each registered polygon, the pipeline — `verify_greenhouses.py` —
extracts a statistical signature from current high-resolution RGB
orthoimagery and trains a binary classifier to decide: *is there still a
greenhouse here?*

### 1. Feature engineering

Only pixels inside the polygon are used: pixels outside it, nodata and
black fill are masked. Polygons outside the imagery, or with fewer than
20 valid pixels, are flagged as having no data. Working with RGB only (no
near-infrared band), the script computes **34 candidate features per
polygon**:

- **Per-band statistics** (12) — mean, standard deviation and 25th/75th
  percentiles of R, G and B;
- **Texture** (3) — mean absolute difference between horizontally and
  vertically adjacent pixels, per band (greenhouse roofs are locally smooth
  but regularly structured);
- **Entropy** (3) — Shannon entropy of a 32-bin histogram over the full
  radiometric range, per band;
- **RGB vegetation indices** (4) — Excess Green (ExG), Excess Green minus
  Excess Red (ExGR), VARI (per-pixel median, near-zero denominators masked)
  and the normalised G–R difference: common NIR-free substitutes for NDVI;
- **Colour descriptors** (8) — normalised G–B and R–B differences,
  brightness, chroma (max − min of the three bands), dominant channel, and
  the R/G, R/B and G/B ratios of band means;
- **Geometry** (4) — area, perimeter, compactness (4πA/P²) and
  perimeter/area ratio.

Geometric features require a projected CRS in metres (e.g. PT-TM06/ETRS89,
EPSG:3763); the script stops if the CRS is geographic.

### 2. Leakage-free training pipeline

The labelled polygons are first split, stratified by class, into a training
set (75 %) and a held-out test set (25 %). Every subsequent step is fitted
on training data only, inside a single `imbalanced-learn` pipeline, so that
it is re-fitted within each cross-validation fold:

1. **KNN imputation** (5 neighbours) of missing feature values;
2. **Robust scaling** (median/IQR), insensitive to the outliers that
   spectral statistics inevitably produce;
3. **Feature selection** — features whose Random Forest importance exceeds
   the mean importance are kept;
4. **SMOTE oversampling** of the minority class, enabled only when the
   minority/majority ratio in the training set is below 0.7 (a heuristic
   threshold). Synthetic samples are generated from training folds only.
   In the run reported below the labelled classes were balanced (ratio
   0.98), so SMOTE was not triggered;
5. **Classifier.**

### 3. Model comparison and tuning

Random Forest and Gradient Boosting are tuned by grid search over
stratified 5-fold cross-validation on the training set, scored on
**balanced accuracy**. The model with the best cross-validated balanced
accuracy is selected and then evaluated **once** on the held-out test set,
reporting accuracy with a 95 % confidence interval, balanced accuracy,
per-class precision and recall, and AUC-ROC.

### 4. Final model and output

After evaluation, the selected pipeline is refitted on all labelled
polygons with the same hyperparameters and applied to every polygon in the
registry. Each polygon receives:

- `Conf_pred` — predicted class: greenhouse present (1) or absent (0);
- `Prob_pos` — predicted probability of class 1. Probabilities are **not
  calibrated**: they rank polygons by model confidence but should not be
  read as exact frequencies;
- `Triagem` — triage: `presente` (`Prob_pos` ≥ 0.7), `ausente` (≤ 0.3),
  `verificar` (in between: goes on the short list for field or desk
  inspection) or `sem_dados` (no valid imagery);
- `Conjunto` — whether the polygon was used for training, testing, or was
  unlabelled.

The result is written as a FileGDB layer and a GeoJSON file (WGS84), ready
for the SIA-ZV updating workflow. The triage thresholds are configurable.

## Results

The ground truth consists of **162 registry polygons** (80 greenhouse
present, 82 absent), labelled by visual photo-interpretation of the
ortoSat2023 imagery on polygons picked while browsing the map — a
convenience sample, not a formal random sample. Random Forest was selected
over Gradient Boosting. Feature selection kept **10 of the 34** candidate
features, led by `gb_index`, `g_b_ratio`, `rb_index` and `p75_B`; no
geometric feature was retained, so the decision rests on colour alone.
The dominance of blue-band relationships is consistent with the bright,
low-chroma signature of plastic covers relative to bare soil and
vegetation, which absorb comparatively more in the blue.

| Metric | Value |
|---|---|
| Held-out test set | 41 polygons (21 absent, 20 present) |
| Test accuracy (95 % CI, Clopper–Pearson) | **95.1 %** (83.5–99.4 %) |
| Test balanced accuracy | 95.2 % |
| Class 0 (greenhouse absent) — recall / precision | 0.90 / 1.00 |
| Test AUC-ROC | 1.000 |
| Cross-validated balanced accuracy (5-fold, training set) | 100 % in every fold |
| Registry polygons assessed | 1,792 (all with valid imagery) |

Both test errors are greenhouses that had disappeared but were predicted
as present; every polygon predicted as absent was indeed absent. An AUC of
1.0 means the model ranked all test polygons correctly — the two errors lie
just on the wrong side of the 0.5 decision threshold. Perfect
cross-validation scores reflect a labelled sample made mostly of clear-cut
cases and should not be read as an error-free classifier; the test interval
is the more realistic measure.

Applied to the full registry, the model estimates that **1,300 of the
1,792 registered greenhouses (72.5 %) are still present**, covering
142.6 of 178.0 ha (80 %) — removed structures are, on average, smaller.
Only **98 polygons (5.5 %)** fall in the uncertain band and are flagged for
field or desk verification. Small polygons concentrate both the losses and
the uncertainty:

| Polygon area | Polygons | Predicted present | Flagged for verification |
|---|---|---|---|
| < 500 m² | 556 | 60 % | 10 % |
| 500–2,000 m² | 1,045 | 76 % | 4 % |
| > 2,000 m² | 191 | 90 % | 2 % |

### Limitations

- **The reference is photo-interpretation, not field data.** Labels were
  produced by interpreting the same ortoSat2023 imagery the model uses, so
  the metrics measure agreement with an expert interpreter on that image,
  not with conditions on the ground.
- **Non-random, small labelled sample.** The labelled set is about half
  absences, whereas the model predicts roughly a quarter of the unlabelled
  polygons as absent: the sample over-represents disappeared and clear-cut
  cases. The test metrics therefore describe how well the classes are
  separated, not the accuracy over the whole registry, and they move
  noticeably as labels are added. A random sample of confident predictions,
  checked independently, would provide an unbiased registry-wide estimate.
- **Errors lean towards keeping obsolete entries.** Missed absences (class 0
  recall 0.90) leave disappeared greenhouses in the registry; lowering the
  decision threshold or widening the `verificar` band trades workload for
  fewer such misses.
- **Presence, not use.** The model detects a greenhouse structure, not
  whether it is in production.
- **Single imagery date.** Structures temporarily without plastic, or under
  construction, may be misclassified, and changes after the imagery date
  are not captured until the next campaign.
- **Indicative importances.** Impurity-based importances are biased among
  correlated features (e.g. band ratios and normalised differences), so the
  ranking should be read qualitatively.

## Why it matters

- **Registry maintenance at a fraction of the cost** — instead of
  re-surveying 1,792 polygons, field effort concentrates on the ~100
  flagged as uncertain;
- **A repeatable audit** — the method can be re-run on every new imagery
  campaign (e.g. each DGT orthoimagery release), turning registry
  validation from a one-off project into a routine;
- **Complementary to detection** — together with the
  [U-Net greenhouse detection](https://github.com/LFilipePacheco/greenhouse-detection-unet)
  project, it closes the loop: detection finds structures missing from the
  registry; verification flags registered structures missing from the
  ground.

## Repository contents

| File | Purpose |
|---|---|
| `verify_greenhouses.py` | Full pipeline: feature extraction from imagery, leakage-free preprocessing and feature selection, model comparison and tuning, held-out evaluation, per-polygon classification with probabilities and triage |
| `requirements.txt` | Python dependencies |
| `docs/` | Figures used in this README |

## Usage

Edit the configuration block at the top of `verify_greenhouses.py`:

- `poligonos_path` / `poligonos_layer` — the registry polygon layer;
- `campo_alvo` — the ground-truth field (`1` present, `0` absent, empty if
  unlabelled; default `Conf_novo`);
- `raster_path` — the RGB orthoimagery (bands 1–3 = R, G, B);
- `output_gdb`, `output_layer`, `output_dir` — output locations.

Then run:

```bash
pip install -r requirements.txt
python verify_greenhouses.py
```

The results folder contains the output layer (GeoJSON), a text report with
all metrics and selected hyperparameters, and the correlation,
feature-importance and confusion-matrix figures. Writing FileGDB output
requires GDAL ≥ 3.6.

## Stack

Python · NumPy · pandas · scikit-learn · imbalanced-learn · SciPy ·
GeoPandas · Shapely · rasterio · Matplotlib · seaborn · DGT ortoSat2023
orthoimagery (open WMS, Direção-Geral do Território)

## About the data

The greenhouse registry (SIA-ZV), the ground-truth labels and the
verification results are institutional property of CCDR-Norte, I.P. and are
not published here. The imagery is the open ortoSat2023 service of the
Direção-Geral do Território (DGT). The code is shared as a working
reference implementation.

---

**Luís Filipe Pacheco** — Senior Engineer & Data Scientist,
CCDR-Norte, I.P. · [GitHub profile](https://github.com/LFilipePacheco) ·
[LinkedIn](https://www.linkedin.com/in/lu%C3%ADs-filipe-pacheco-471495b/) ·
[ORCID](https://orcid.org/0009-0001-7676-6542)
