# Predicting Apartment Rental Prices in Switzerland

> HSLU DSPRO1 — Team 8 — Machine-Learning-Modell zur Vorhersage von Kaltmieten Schweizer Wohnungen aus Wohnungs-, Geo- und Lagedaten.

---

## Was macht das Projekt?

Wir trainieren mehrere Regressions-Modelle (Ridge, RandomForest, GradientBoosting, XGBoost, LightGBM, Stacking) auf einem selbst zusammengetragenen Datensatz Schweizer Mietwohnungen, vergleichen sie systematisch und bauen daraus eine produktionsnahe Pipeline (`RentPredictor`), die aus Wohnungs-Features eine Mietpreis-Vorhersage liefert — inklusive Modell-Karte, Stabilitäts-Check und Streamlit-Demo.

**Endstand:** LightGBM mit KNN-Distance-Features erreicht **RMSE Eval = 393 CHF** und **R² = 0.751** auf dem 80/20-Split. Im Vergleich zur Dummy-Baseline (RMSE 847 CHF) ist das eine Reduktion von ≈ 54 %. Als Robustheits-Fallback gibt es eine Wide-Pipeline (4 Features, ~9'500 Zeilen) mit halbiertem Train/Eval-Gap.

Demo: `make app` (oder `streamlit run src/app.py`) öffnet ein interaktives Frontend, in dem man Wohnungs-Parameter eingibt und sofort eine Preis-Schätzung bekommt.

## Quick Start: klonen und starten

Voraussetzungen: **Git und Python 3.12 (64 Bit)**. Die erste Installation braucht Internet und mehrere Minuten. Das Setup richtet App, JupyterLab, ML-Pakete und den Notebook-Kernel automatisch in der lokalen .venv ein und prüft die mitgelieferten Modelle.

**Windows (PowerShell):**

~~~powershell
git clone https://github.com/0xff4b/dspro1.git
py -3.12 dspro1/setup.py --start notebook
~~~

**Linux / WSL / macOS:**

~~~bash
git clone https://github.com/0xff4b/dspro1.git
python3.12 dspro1/setup.py --start notebook
~~~

Auf macOS wird zusätzlich die OpenMP-Laufzeit benötigt: **brew install libomp**. Auf minimalen Linux-Systemen muss **libgomp1** installiert sein. Im Notebook den Kernel **DSPRO (.venv, Python 3.12)** auswählen.

Für die **Streamlit-Demo** im zweiten Befehl **--start notebook** durch **--start app** ersetzen. Mit **--profile app --start app** werden nur die App-Abhängigkeiten installiert. Ohne **--start** wird nur die Umgebung vorbereitet. Eine manuelle Aktivierung ist nicht nötig.

**App mit Docker**, ohne lokale Python-Installation (Git und laufendes Docker mit Compose vorausgesetzt):

~~~bash
git clone https://github.com/0xff4b/dspro1.git
docker compose -f dspro1/compose.yaml up --build
~~~

Die App läuft unter http://localhost:8501. Stoppen mit Ctrl+C, Container entfernen mit **docker compose -f dspro1/compose.yaml down**. Diese Befehle starten keine Azure-Ressourcen.

Eine kompatible bestehende .venv wird weiterverwendet. Eine defekte oder inkompatible Umgebung wird vor dem Neubau umbenannt und gesichert. Aktive Umgebungen ausserhalb des Projekts und eine vorhandene .python-version werden nicht geändert.

**Probleme nach einem Pull?** Im Projektordner zuerst **python3.12 setup.py --doctor** ausführen (Windows: **py -3.12 setup.py --doctor**). Fehlende oder beschädigte Projektdateien lassen sich mit **--repair** aus dem lokalen Git-Stand wiederherstellen; vorhandene Inhalte werden vorher gesichert. Bewusst neu trainierte Modelle lassen sich mit **--allow-local-data** prüfen und verwenden.

Details zu Update, Reparatur, Backups, Abhängigkeiten und optionalen Datenbank-/Scraper-Schritten: [Setup-Anleitung](docs/dspro1/SETUP.md). **make help** zeigt die Kurzbefehle unter Linux/macOS.

## Projektstruktur

```
dspro1/
├── README.md                       <- du bist hier
├── Makefile                        <- Häufige Workflows: make setup / report / app / notebook / doctor
├── requirements.txt                <- Eingabe für die vollständige Paketliste
├── .gitignore
├── docs/
│   ├── dspro1/                     <- sämtliche bisherige Dokumentation
│   │   ├── __templates/
│   │   ├── ai-canvas/
│   │   ├── data-sheet/
│   │   ├── final-report/           <- LaTeX, PDF und PNG-Abbildungen in fig/
│   │   ├── fig/poster-ai-event/    <- zusätzliche SVG-Exporte fürs Poster
│   │   ├── presentation-final/
│   │   ├── presentation-mid-term/
│   │   ├── project-proposal/
│   │   └── schemes/                <- drawio Architektur-Diagramme
│   └── dspro2/                     <- vorbereitet für neue Arbeiten
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
| 21    | Export aller Figures für den Final Report (`docs/dspro1/final-report/fig/`) |

Die Backup-Datei `model_v3_clean.backup.ipynb` enthält den ungestrafften Pre-Cleanup-Stand mit allen Experimenten (Log-Target, Imputation, RFECV usw.), die im finalen Notebook nicht mehr drin sind.

## Datenquellen

- **GWR** (Gebäude- und Wohnungsregister) — siehe `src/external-sources/gwr_egid_db_sync.ipynb`
- **swisstopo** (Geo-Koordinaten LV95, Höhe) — siehe `src/external-sources/swisstopo_enrich_db_sync.ipynb`
- **Eigener Scraper** für rentumo.ch: `src/scrapegoat/` (Rust, Hauptpipeline) und `src/rentables-scraper/` (erste Iteration)

Aufbereitetes Trainings-Set: `src/external-sources/output_csv/model.csv` (~4'500 Wohnungen × 12 Spalten). Der Wide-Pipeline-Datensatz `model_wide.csv` enthält ~9'500 Zeilen mit nur 4 Kern-Features als Fallback, wenn die GWR-/swisstopo-Enrichment-Daten lückenhaft sind.

## Reproduzierbarkeit

- `RANDOM_STATE = 42` durchgängig in Notebook und App
- `REFERENCE_YEAR = 2026` für `building_age`-Berechnung
- requirements-full.lock und requirements-app.lock mit exakten Versionen und SHA-256-Prüfsummen
- `models/rent_predictor_v3.joblib` enthält die finale Pipeline + Metadata (`training_date`, `python_version`, `test_metrics`)
- `models/lgbm_wide_4feat.joblib` für die Wide-Pipeline-Fallback-Variante
- Modell-Karte und Datasheet im Notebook (Kap. 19 / 20) sowie in `docs/dspro1/data-sheet/`

## Wartung

Das Setup verändert keine Notebook-Zellen und startet kein Training. Historische Skripte zur Notebook-Bereinigung liegen weiterhin unter src/notebooks/; sie gehören nicht zum normalen Setup.

Paketänderungen werden in requirements.txt bzw. requirements-app.txt gepflegt und anschliessend in beide Lockdateien übernommen. GitHub Actions prüft den frischen Checkout, die vollständige Installation, Modellvorhersagen und die Wiederverwendung der Umgebung unter Windows, Linux und macOS. Die [Setup-Anleitung](docs/dspro1/SETUP.md) beschreibt die Pflege und Fehlerbehebung.

## Team und Lizenz

- **Team 8 — DSPRO1 HSLU** (Hochschule Luzern)
- Maintainer: Elias Martinelli
- Co-Autor: Timo Schlumpf
- Status: Final (Abgabe-Stand)
- Lizenz: tbd (akademisches Projekt)

## Azure-Prototyp mit Docker

Bereitstellung, automatischer GitHub-Upload sowie Start/Stopp/Löschen: [Azure-Anleitung](docs/dspro1/AZURE.md).

Poster-Abbildungen werden zusätzlich als SVG nach docs/dspro1/fig/poster-ai-event/ exportiert. Details: [Abbildungsexport](docs/dspro1/fig/poster-ai-event/README.md).
