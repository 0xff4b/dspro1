# Predicting Apartment Rental Prices in Switzerland

> HSLU DSPRO1 — Team 8 — Machine-Learning-Modell zur Vorhersage von Kaltmieten Schweizer Wohnungen aus Wohnungs-, Geo- und Lagedaten.

---

## Was macht das Projekt?

Wir trainieren mehrere Regressions-Modelle (Ridge, RandomForest, GradientBoosting, XGBoost, LightGBM, Stacking) auf einem selbst zusammengetragenen Datensatz Schweizer Mietwohnungen, vergleichen sie systematisch und bauen daraus eine produktionsnahe Pipeline (`RentPredictor`), die aus Wohnungs-Features eine Mietpreis-Vorhersage liefert — inklusive Modell-Karte, Stabilitäts-Check und Streamlit-Demo.

**Endstand:** LightGBM mit KNN-Distance-Features erreicht **RMSE Eval = 393 CHF** und **R² = 0.751** auf dem 80/20-Split. Im Vergleich zur Dummy-Baseline (RMSE 847 CHF) ist das eine Reduktion von ≈ 54 %. Als Robustheits-Fallback gibt es eine Wide-Pipeline (4 Features, ~9'500 Zeilen) mit halbiertem Train/Eval-Gap.

Demo: `make app` (oder `streamlit run src/app.py`) öffnet ein interaktives Frontend, in dem man Wohnungs-Parameter eingibt und sofort eine Preis-Schätzung bekommt.

## Quick Start

```bash
# 1. Repository klonen
git clone <repo-url>
cd dspro1

# 2. (Empfohlen) Virtuelles Environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Abhängigkeiten installieren
pip install -r requirements.txt

# 4. Notebook öffnen (volle Pipeline + Analyse, 21 Kapitel)
jupyter lab src/notebooks/model_v3_clean.ipynb

# 5. Demo-App starten (interaktive Vorhersage im Browser)
make app                       # oder: streamlit run src/app.py

# 6. Final-Report-PDF bauen (pdflatex + bibtex + 2x pdflatex)
make report
```

`make help` listet alle verfügbaren Targets.

## Projektstruktur

```
dspro1/
├── README.md                       <- du bist hier
├── Makefile                        <- Häufige Workflows: make report / app / notebook-fix
├── requirements.txt                <- Python-Abhängigkeiten
├── .gitignore
├── docs/                           <- Berichte, Präsentationen, AI Canvas, Data Sheet
│   ├── ai-canvas/
│   ├── data-sheet/
│   ├── final-report/               <- LaTeX-Source + fig/ + kompiliertes PDF
│   │   ├── *.tex / *.bib
│   │   └── fig/                    <- PNGs vom Notebook-Export (Kap. 21)
│   ├── presentation-final/
│   ├── presentation-mid-term/
│   ├── project-proposal/
│   └── schemes/                    <- drawio Architektur-Diagramme
├── src/
│   ├── app.py                      <- Streamlit Demo-App
│   ├── notebooks/
│   │   ├── model_v3_clean.ipynb        <- Hauptnotebook (21 Kapitel)
│   │   ├── model_v3_clean.backup.ipynb <- Pre-Cleanup-Snapshot (Referenz)
│   │   ├── humanize_notebook.py        <- Cleanup-Script (rewrite, drop, restyle)
│   │   ├── fix_summary_blocks.py       <- Post-Cleanup-Patches für bekannte NameErrors
│   │   └── models/                     <- gespeicherte .joblib-Artefakte
│   │       ├── best_model_v3.joblib
│   │       ├── rent_predictor_v3.joblib
│   │       ├── lgbm_minimal_4feat.joblib
│   │       ├── lgbm_wide_4feat.joblib
│   │       └── requirements_used.txt
│   ├── external-sources/           <- Daten-Sync-Notebooks + aufbereitete CSVs
│   │   ├── gwr_egid_db_sync.ipynb
│   │   ├── swisstopo_enrich_db_sync.ipynb
│   │   ├── final_records.ipynb
│   │   └── output_csv/
│   │       ├── model.csv               <- Standard-Trainings-Set (~4'500 Wohnungen)
│   │       └── model_wide.csv          <- Wide-Pipeline-Set (~9'500 Wohnungen)
│   ├── scrapegoat/                 <- Rust-basierter Scraper (Hauptpipeline)
│   └── rentables-scraper/          <- Rust-Scraper (erste Iteration)
```

## Was steckt im Hauptnotebook (`model_v3_clean.ipynb`)?

21 Kapitel, linear aufgebaut nach dem Vorgehen, das wir auch real eingehalten haben — erst Daten anschauen, dann eine dumme Baseline, dann immer komplexere Modelle, dann Diagnose, dann End-to-End-Pipeline.

| Kapitel | Inhalt |
|---|---|
| 1–7   | Setup, Datenladen, Spalten-Rename, Datenqualität, Outlier-Filter, Feature Engineering, zentrale Feature-Sets (`FEATURES_SMALL/ENGINEERED/ALL`) |
| 8     | Sauberer Train/Eval-Split (80/20, fixer `random_state=42`) |
| 9–13  | Modell-Pipelines, fairer Modellvergleich, Train-vs-Eval-Plots, Overfitting-Diagnose, 5-Fold-CV |
| 14    | Actual-vs-Predicted, Residuen-Analyse (Histogramm, Residuals-vs-Predicted, Q-Q) |
| 15–17 | Fehleranalyse nach Preis-Quartilen, Feature Importance, Modell-Auswahl + Sanity-Predict |
| 18    | Geo-Analyse: EDA, KMeans- und DBSCAN-Clustering, Group-Split-Robustheits-Check |
| 19    | Iterative Verbesserungen: Tuning (RandomizedSearchCV + Halving), Stacking, Bootstrap-CIs, KNN-Distance-Features (entscheidender Hebel: 399 → 393 CHF), Conformal Prediction (MAPIE), Drift-Check, Bias-Analyse, Modell-Karte, regularisiertes LGBM auf reduzierten Feature-Sets, Wide-Pipeline |
| 20    | End-to-End `RentPredictor`-Klasse, 60/20/20-Split, Hold-Out-Test (bis zum Schluss unangetastet), Data-Sheet |
| 21    | Export aller Figures für den Final Report (`docs/final-report/fig/`) |

Die Backup-Datei `model_v3_clean.backup.ipynb` enthält den ungestrafften Pre-Cleanup-Stand mit allen Experimenten (Log-Target, Imputation, RFECV usw.), die im finalen Notebook nicht mehr drin sind.

## Datenquellen

- **GWR** (Gebäude- und Wohnungsregister) — siehe `src/external-sources/gwr_egid_db_sync.ipynb`
- **swisstopo** (Geo-Koordinaten LV95, Höhe) — siehe `src/external-sources/swisstopo_enrich_db_sync.ipynb`
- **Eigener Scraper** für rentumo.ch: `src/scrapegoat/` (Rust, Hauptpipeline) und `src/rentables-scraper/` (erste Iteration)

Aufbereitetes Trainings-Set: `src/external-sources/output_csv/model.csv` (~4'500 Wohnungen × 12 Spalten). Der Wide-Pipeline-Datensatz `model_wide.csv` enthält ~9'500 Zeilen mit nur 4 Kern-Features als Fallback, wenn die GWR-/swisstopo-Enrichment-Daten lückenhaft sind.

## Reproduzierbarkeit

- `RANDOM_STATE = 42` durchgängig in Notebook und App
- `REFERENCE_YEAR = 2026` für `building_age`-Berechnung
- `requirements.txt` mit Versions-Pins
- `models/rent_predictor_v3.joblib` enthält die finale Pipeline + Metadata (`training_date`, `python_version`, `test_metrics`)
- `models/lgbm_wide_4feat.joblib` für die Wide-Pipeline-Fallback-Variante
- Modell-Karte und Datasheet im Notebook (Kap. 19 / 20) sowie in `docs/data-sheet/`

## Wartungs-Scripts

Das Notebook wurde mit `humanize_notebook.py` aus einem deutlich grösseren Explorations-Notebook auf das aktuelle Format eingedampft. Wenn nach einem Cleanup ein `NameError` aus einer entfernten Kapitelzelle hochkommt (z.B. `summary_blocks`, `FEATURES_MINIMAL`, …), wendet `fix_summary_blocks.py` defensive Patches an. Beide Scripts sind idempotent.

```bash
make notebook-clean   # python humanize_notebook.py — vollständiger Cleanup-Lauf
make notebook-fix     # python fix_summary_blocks.py — Post-Cleanup-Regression-Fixes
```

## Team und Lizenz

- **Team 8 — DSPRO1 HSLU** (Hochschule Luzern)
- Maintainer: Elias Martinelli
- Co-Autor: Timo Schlumpf
- Status: Final (Abgabe-Stand)
- Lizenz: tbd (akademisches Projekt)
