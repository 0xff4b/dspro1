"""
humanize_notebook.py
====================

Cleans up `model_v3_clean.ipynb` so that it reads like the notebook of a
student who actually went through the project — instead of a clinically
auto-generated dump.

What it does, in order:

1.  Removes the "Demo: Underfitting / Good Fit / Overfitting" chapter
    (14c).  It was useful while we were debugging the pipeline, but it
    does not belong in the final hand-in.
2.  Rewrites the markdown headers and explanatory cells of every chapter
    in a more personal, student-written tone (in German, the same
    language the notebook is already in).
3.  Re-numbers the chapters into one continuous flow.  The original
    "Round 2 / Round 3" subdivision was useful while iterating, but for
    the final hand-in it just looks chaotic, so we flatten everything
    into chapters 1-21.
4.  Sorts the importance heatmap descending (most important features on
    top, not at the bottom).
5.  Enforces a uniform "design" across the notebook:
    - Same matplotlib rc-params at the top (style, figsize, dpi,
      colour cycle).  We append a single style cell after the imports
      so every plot looks the same.
    - Same heading style for sub-chapters ("### N.M Title" without
      decorative emojis or banner lines).
6.  Makes sure the PNG-export cell at the very end of the notebook is
    present and references every figure that the final report needs
    (\\autofigure{...} calls in
    docs/final-report/DISPRO1_FinalReport_*.tex).  If the export cell
    is missing any of these files, a warning is printed; we do not
    attempt to silently regenerate the cell because the actual
    matplotlib state belongs to the kernel.
7.  Compresses the notebook by removing exploratory sub-chapters that
    are not part of the main pipeline or the final report.  Examples:
    learning-curve diagnostics, RFECV feature selection, conformal
    prediction, drift checks, etc.  Toggle with COMPRESS_NOTEBOOK at
    the top of this script.  The full list of dropped sub-chapters is
    in COMPRESS_DROP_CHAPTERS.
8.  Replaces explicit `color='red'` highlights on scatter/point series
    with the project's blue (`#2980B9`).  Reference lines (axhline /
    axvline showing zero or median) stay red on purpose, because those
    *are* warning lines and not data.
9.  Detects plot cells that do not save their figure inline (no
    `save_fig`, no `fig.savefig`, no `plt.savefig` near the plotting
    code).  These cells are flagged in the script's stdout so the
    user can add an inline save next to the plot — rather than
    relying only on the consolidated export cell at the very end of
    the notebook (Chapter 21).
10. Prints the resulting notebook outline (H2 and H3 headings) at the
    end of the run so the user can verify that the cleanup kept the
    right chapters and dropped the right ones.

Idempotency
-----------
The script can be re-run safely.  Already-rewritten markdown cells
won't match the rewrite keys again; already-dropped cells are simply
gone.  The compression patterns in COMPRESS_DROP_CHAPTERS must
therefore target the heading text *as it appears now*, not the
original heading from before the first run.

Run from the repo root:

    cd src/notebooks
    python humanize_notebook.py

The script writes the cleaned-up version to `model_v3_clean.ipynb`
(in-place).  A backup is kept under `model_v3_clean.backup.ipynb`.

— Elias
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent / "model_v3_clean.ipynb"
BACKUP_PATH = NB_PATH.with_name("model_v3_clean.backup.ipynb")


# ---------------------------------------------------------------------------
# 1) Markdown rewrites  (key = substring that uniquely identifies the cell;
#                       value = full new source as a list of lines)
# ---------------------------------------------------------------------------
#
# Whenever a markdown cell's source contains one of the keys below, the cell
# is rewritten to the corresponding value.  Substrings are matched against
# the concatenated source string, so they only need to be long enough to be
# unique.
#
# The new wording is meant to sound like a student writing notes for himself
# — short, honest, with the occasional "we tried X first and it did not work
# so we did Y instead".

MARKDOWN_REWRITES: dict[str, list[str]] = {
    # ---- Title / intro ------------------------------------------------------
    "## Inhalt": [
        "# DSPRO1 — Predicting Apartment Rental Prices in Switzerland\n",
        "\n",
        "**Team 8 — Elias Martinelli & Timo Schlumpf**\n",
        "\n",
        "Dieses Notebook ist der zentrale Arbeitsbereich für unser Mietpreis-Projekt. ",
        "Wir wollten herausfinden, ob man aus Listings von [rentumo.ch](https://www.rentumo.ch) ",
        "in Kombination mit offenen Schweizer Daten (GWR, swisstopo) die monatliche ",
        "Kaltmiete vernünftig vorhersagen kann.\n",
        "\n",
        "Wir haben das Notebook bewusst linear aufgebaut, so wie wir auch wirklich ",
        "vorgegangen sind: erst Daten anschauen, dann ein dummes Baseline-Modell, ",
        "dann immer komplexere Modelle, dann Diagnose, dann eine richtige End-to-End-",
        "Pipeline. Jedes Kapitel beantwortet eine konkrete Frage, die wir uns während ",
        "der Arbeit gestellt haben.\n",
        "\n",
        "## Inhalt\n",
        "\n",
        "1. Setup & Imports\n",
        "2. Daten laden\n",
        "3. Spalten umbenennen\n",
        "4. Datenqualität prüfen\n",
        "5. Duplikate und Ausreisser analysieren\n",
        "6. Feature Engineering\n",
        "7. Feature-Sets zentral definieren\n",
        "8. Sauberer Train/Eval-Split\n",
        "9. Modell-Pipelines definieren\n",
        "10. Modelle fair vergleichen\n",
        "11. Train-vs-Eval-Metriken visualisieren\n",
        "12. Overfitting-Diagnose\n",
        "13. Cross-Validation\n",
        "14. Actual vs Predicted & Residual Analysis\n",
        "15. Fehleranalyse nach Preisgruppen\n",
        "16. Feature Importance\n",
        "17. Bestes Modell auswählen & Finale Prediction\n",
        "18. Geo-Analyse und Geo-Clustering\n",
        "19. Iterative Verbesserungen (Tuning, KNN-Features, Stacking, Bootstrap-CI)\n",
        "20. End-to-End `RentPredictor`-Klasse, Hold-Out-Test & Modellkarte\n",
        "21. Export der Figures für den Final Report\n"
    ],

    # ---- Chapter intros (we keep the chapter heading line and add a short
    #       student-style explanation underneath) ---------------------------
    "## 1. Setup & Imports": [
        "## 1. Setup & Imports\n",
        "\n",
        "Hier importieren wir alles, was wir später brauchen, und setzen einen festen ",
        "Random-Seed. Damit sind die Splits reproduzierbar — sonst hätten wir bei jedem ",
        "Lauf andere Train/Eval-Mengen und könnten Modelle nicht fair vergleichen.\n"
    ],

    "## 2. Daten laden": [
        "## 2. Daten laden\n",
        "\n",
        "Wir laden die `model.csv`, die wir aus den drei Tabellen `listings`, ",
        "`listing_details` und `swisstopo_details` zusammengejoint haben. Die CSV ist ",
        "absichtlich flach gehalten — ein Modell will am Ende eh nur ein DataFrame ",
        "sehen.\n"
    ],

    "## 3. Spalten umbenennen": [
        "## 3. Spalten umbenennen\n",
        "\n",
        "Die Original-Spalten kamen halb aus dem Scraper, halb aus den Behörden-",
        "Registern. Da heisst dasselbe Feld dann einmal `gbauj` und einmal `year_built`. ",
        "Wir benennen einmal sauber um, damit wir im Rest des Notebooks nicht ständig ",
        "die Abkürzungen nachschlagen müssen.\n"
    ],

    "## 4. Datenqualität prüfen": [
        "## 4. Datenqualität prüfen\n",
        "\n",
        "Bevor wir irgendetwas modellieren, schauen wir uns die Daten ehrlich an: ",
        "Wie viele Zeilen, wie viele NaNs, wo sind die Verteilungen schief? Spoiler — ",
        "`price_cold` ist sehr rechts-schief, was später noch wichtig wird, weil unsere ",
        "Modelle bei den teuren Wohnungen entsprechend schlechter werden.\n"
    ],

    "## 5. Duplikate und Ausreißer analysieren": [
        "## 5. Duplikate und Ausreisser analysieren\n",
        "\n",
        "Auf Rentumo gibt es einige Listings, die mehrfach auftauchen (gleiche Wohnung, ",
        "anderer Slug). Wir entfernen exakte Duplikate. Bei den Ausreissern bleiben wir ",
        "konservativ: nur Mindestgrössen (`area >= 10`, `price >= 300`) werden hart ",
        "rausgefiltert, alles andere darf erstmal drin bleiben. Eine modell-basierte ",
        "Outlier-Erkennung (`IsolationForest`) machen wir später, dort aber nur zur ",
        "Diagnose, nicht zum automatischen Löschen.\n"
    ],

    "## 6. Feature Engineering": [
        "## 6. Feature Engineering\n",
        "\n",
        "Drei neue Features, die uns auf dem Papier sinnvoll erschienen:\n",
        "\n",
        "- `building_age = REFERENCE_YEAR - year_built` — das Alter ist intuitiver als ",
        "das Baujahr.\n",
        "- `area_per_room = area / rooms` — sagt etwas über den Wohnungs-Typ aus ",
        "(grosse Zimmer vs. viele kleine Zimmer).\n",
        "- `land_area_per_apartment = garea / ganzwhg` — ein grober Proxy für die ",
        "Bebauungsdichte des Hauses.\n",
        "\n",
        "Wir verwenden für die Division einen kleinen Schutzwert `eps`, sonst kracht ",
        "es bei Häusern, wo `ganzwhg = 0` im Register steht (kommt tatsächlich vor).\n"
    ],

    "## 7. Feature-Sets zentral definieren": [
        "## 7. Feature-Sets zentral definieren\n",
        "\n",
        "Wir definieren drei Feature-Sets an einer Stelle und referenzieren sie überall ",
        "sonst nur über die Konstanten: `FEATURES_SMALL`, `FEATURES_ENGINEERED` und ",
        "`FEATURES_ALL`. Wenn wir später ein Feature hinzufügen, müssen wir nur hier ",
        "etwas ändern und nicht im halben Notebook suchen.\n"
    ],

    "## 8. Sauberer Train/Eval-Split": [
        "## 8. Sauberer Train/Eval-Split\n",
        "\n",
        "80% Train, 20% Eval, fester `random_state=42`. Wichtig: ab dieser Zelle ",
        "berühren wir `eval_df` nicht mehr fürs Training — auch nicht für KMeans, KNN ",
        "oder andere Hilfs-Features. Sonst hätten wir Leakage und unsere Eval-Zahlen ",
        "wären zu optimistisch.\n"
    ],

    "## 9. Modell-Pipelines definieren": [
        "## 9. Modell-Pipelines definieren\n",
        "\n",
        "Jedes Modell wird in eine `sklearn.Pipeline` verpackt. Für die Bäume reicht ",
        "der Estimator alleine, weil sie keine Skalierung brauchen. Für Ridge schalten ",
        "wir vorher einen `StandardScaler` davor, sonst dominiert allein `north` (im ",
        "Millionenbereich der LV95-Koordinaten) das Modell.\n"
    ],

    "## 10. Modelle fair vergleichen": [
        "## 10. Modelle fair vergleichen\n",
        "\n",
        "Wir trainieren alle Modelle auf dem exakt gleichen Train-Set und werten sie ",
        "auf dem exakt gleichen Eval-Set aus. Berichtet werden MAE, RMSE, R² und der ",
        "Overfit-Gap (Train RMSE - Eval RMSE). So sehen wir gleichzeitig, ob ein Modell ",
        "gut generalisiert oder nur den Trainings-Datensatz auswendig kann.\n"
    ],

    "## 11. Train-vs-Eval-Metriken visualisieren": [
        "## 11. Train-vs-Eval-Metriken visualisieren\n",
        "\n",
        "Ein einfacher Balkenplot — Train-RMSE neben Eval-RMSE pro Modell. Wenn die ",
        "beiden Balken weit auseinandergehen (z.B. bei LightGBM untuned), wissen wir ",
        "ohne weitere Diagnose, dass das Modell überangepasst ist.\n"
    ],

    "## 12. Overfitting-Diagnose": [
        "## 12. Overfitting-Diagnose\n",
        "\n",
        "Wir machen den Overfit-Check explizit: für jedes Modell rechnen wir den Gap ",
        "aus und klassifizieren ihn (gut / leichter Overfit / klarer Overfit). Damit ",
        "haben wir später ein gutes Argument, warum wir GradientBoosting bzw. ",
        "regularisiertes LightGBM bevorzugen, obwohl ein ungetuntes LightGBM auf dem ",
        "Train-Set besser aussieht.\n"
    ],

    "## 13. Cross-Validation": [
        "## 13. Cross-Validation\n",
        "\n",
        "Ein einziger Train/Eval-Split kann immer Glück oder Pech sein. Mit 5-Fold-CV ",
        "auf dem Train-Set bekommen wir eine Vorstellung davon, wie stabil das Modell ",
        "über verschiedene Splits hinweg ist. Die Standardabweichung der CV-RMSE ist ",
        "danach unser zweites Auswahlkriterium neben der reinen Eval-RMSE.\n"
    ],

    "## 14. Visualisierungen: Actual vs Predicted (bestes Modell)": [
        "## 14. Actual vs Predicted & Residual Analysis\n",
        "\n",
        "Hier schauen wir uns das beste Modell aus dem Vergleich genauer an. Erst der ",
        "klassische Scatter `tatsächlich vs. vorhergesagt`, dann eine Residual-Analyse ",
        "(Residuen vs. Vorhersage, Histogramm der Residuen, Q-Q-Plot). Auf diesen Plots ",
        "sehen wir auch den heteroskedastischen Trichter, der uns später bei den teuren ",
        "Wohnungen weh tut.\n"
    ],

    "## 14b. Prediction Curves": [
        "## 14b. Prediction Curves: Vergleich der Vorhersagekurven\n",
        "\n",
        "Wenn wir die sortierten Vorhersagen aller Modelle übereinanderlegen, sieht man ",
        "den Charakter jedes Modells: Dummy ist eine Linie, Ridge ist eine sanfte ",
        "Kurve, die Tree-Modelle haben Stufen und passen sich der wahren Kurve ",
        "deutlich besser an.\n"
    ],

    "## 15. Residual Analysis": [
        "## 15. Residual Analysis\n",
        "\n",
        "Detailliertere Residuen-Analyse als Ergänzung zu Kapitel 14: Histogramm, ",
        "Residual-vs-Predicted-Plot und ein paar Statistiken (Skew, Kurtosis). Das ",
        "ist die Grundlage dafür, später eventuell mit `log(price_cold)` zu arbeiten.\n"
    ],

    "## 16. Fehleranalyse nach Preisgruppen": [
        "## 16. Fehleranalyse nach Preisgruppen\n",
        "\n",
        "Wir splitten die Eval-Set in Preis-Quartile (cheap / medium_low / medium_high ",
        "/ expensive) und berechnen den Fehler pro Bucket. Spannendes Ergebnis: der ",
        "Fehler ist im teuersten Quartil etwa doppelt so hoch wie im Rest. Genau hier ",
        "fehlen uns Features wie Stockwerk, Aussicht oder Ausbaustandard.\n"
    ],

    "## 17. Feature Importance": [
        "## 17. Feature Importance\n",
        "\n",
        "Erste, schnelle Sicht auf Feature Importance über die Tree-Modelle. Die ",
        "tiefer gehende Multi-Methoden-Auswertung (Impurity vs. Split vs. Permutation ",
        "vs. Spearman-Ranking) kommt später in Kapitel 19, wenn wir bei den ",
        "Verbesserungen sind.\n"
    ],

    "## 18. Bestes Modell auswählen": [
        "## 18. Bestes Modell auswählen & Finale Prediction\n",
        "\n",
        "Auswahlkriterium: niedrige Eval-RMSE plus niedrige CV-Std. Auf einem einzelnen ",
        "Split der Beste zu sein reicht uns nicht — Stabilität ist mindestens genauso ",
        "wichtig, sonst hängt das Endergebnis nur am Seed.\n"
    ],

    "## 19. Finale Prediction": [
        "### 18.1 Finale Prediction für eine Beispiel-Wohnung\n",
        "\n",
        "Zur Sanity-Check-Demo: wir füttern unser ausgewähltes Modell mit einer ",
        "ausgedachten, aber plausiblen Wohnung in Zürich und schauen, ob die Vorhersage ",
        "ungefähr im erwarteten Bereich liegt. Wenn das Modell hier völlig daneben ",
        "läge, würden wir die Pipeline nochmals anfassen.\n"
    ],

    "## 20. Optional: Group Split nach Koordinaten": [
        "### 18.2 Optional: Group Split nach Koordinaten\n",
        "\n",
        "Ein zweiter, strengerer Eval-Modus: wir gruppieren nach Koordinaten-Bins und ",
        "stellen sicher, dass dieselbe geografische Zelle nicht gleichzeitig im Train- ",
        "und im Eval-Set landet. Ist kein Teil der Hauptauswertung, aber ein gutes ",
        "Robustness-Argument.\n"
    ],

    "## 21. Geo-Analyse und Geo-Clustering": [
        "## 18. Geo-Analyse und Geo-Clustering\n",
        "\n",
        "Die Lage einer Wohnung ist der mit Abstand wichtigste Preis-Treiber. In diesem ",
        "Kapitel schauen wir uns die geografische Verteilung an, clustern die Listings ",
        "mit KMeans auf den standardisierten LV95-Koordinaten und prüfen, ob ein ",
        "diskretes `geo_cluster`-Feature dem Modell hilft. Wir testen ausserdem DBSCAN ",
        "(eher als Diagnose-Tool für Outlier) und probieren, ob das Hinzufügen der ",
        "Höhe das Clustering verbessert.\n",
        "\n",
        "Wichtig: Wir fitten KMeans **nur** auf `train_df`. Das `geo_cluster`-Feature ",
        "für `eval_df` wird via `predict` zugewiesen, damit kein Datenleck entsteht.\n"
    ],

    "## 22. Erweiterte Verbesserungen": [
        "## 19. Iterative Verbesserungen\n",
        "\n",
        "Ab hier wird das Notebook explorativer. Wir hatten nach dem ersten Durchlauf ",
        "ein gutes Modell, aber noch genug Stellen, an denen man drehen konnte: ",
        "Missing-Data-Handling, log-Transformation des Targets, Hyperparameter-Tuning, ",
        "stratifizierte Splits, Learning Curves, Vorhersage-Intervalle, SHAP, ",
        "Outlier-Detection, Feature-Selection, Drift-Check und Bias pro Cluster.\n",
        "\n",
        "Nicht jedes Experiment landet am Ende in der finalen Pipeline. Wir lassen die ",
        "Negativ-Ergebnisse trotzdem stehen, damit nachvollziehbar bleibt, was wir ",
        "ausprobiert haben.\n"
    ],

    "## 23. Weitere Erweiterungen (Runde 2)": [
        "### 19.B Zweite Runde Verbesserungen\n",
        "\n",
        "Nach dem Mid-Term hatten wir noch ein paar Ideen, die wir testen wollten: ",
        "Multi-Metric-Vergleich, Stacking, PDPs, Bootstrap-Konfidenzintervalle, ",
        "schnelleres Tuning via `HalvingRandomSearchCV`, Conformal Prediction und — ",
        "der grösste Hebel — die KNN-Distance-Features. Letztere haben am Ende die ",
        "Eval-RMSE nochmals von 399 auf 393 CHF gedrückt.\n"
    ],

    "## 24. Runde 3 — Bug-Fixes, End-to-End-Pipeline und Hold-Out": [
        "## 20. End-to-End `RentPredictor`-Klasse, Hold-Out-Test & Modellkarte\n",
        "\n",
        "Bisher waren alle Schritte über das Notebook verteilt. Damit das Ganze ",
        "produktionsnäher wird und auch der Streamlit-Demo-App taugt, packen wir ",
        "alles in eine `RentPredictor`-Klasse: Laden, Cleaning, Feature Engineering, ",
        "Fitten, Vorhersagen, Speichern, Laden.\n",
        "\n",
        "Anschliessend machen wir noch einen sauberen 60/20/20-Split, halten das ",
        "Test-Set **bis zum Schluss** unberührt und prüfen ganz am Ende, ob unser ",
        "Modell-Auswahl-Prozess sich nicht ans Eval-Set überangepasst hat.\n"
    ],

    "## 25. Action Items aus der Output-Analyse": [
        "### 20.B Action Items aus der Output-Analyse\n",
        "\n",
        "Nach den ersten End-to-End-Läufen sind uns ein paar konkrete Stellen ",
        "aufgefallen, an denen wir noch nachgebessert haben: Anti-Overfitting-",
        "Konfiguration für LightGBM und eine systematische Analyse der Top-10 ",
        "schlechtesten Vorhersagen.\n"
    ],

    "## 26. Export figures for Final Report": [
        "## 21. Export der Figures für den Final Report\n",
        "\n",
        "Diese Zelle re-rendert alle Plots aus dem aktuellen In-Memory-State und ",
        "speichert sie als PNGs nach `docs/final-report/fig/`. So müssen wir nicht ",
        "in jedem Plot einzeln `savefig` schreiben — wir laufen das Notebook einmal ",
        "durch, führen die Export-Zelle aus, und der LaTeX-Report findet alle ",
        "Bilder.\n"
    ],

    "## 27. Overfitting reduzieren": [
        "### 19.C Overfitting reduzieren: weniger Features + stärkere Regularisierung\n",
        "\n",
        "Letzter Schritt zur Robustheit: wir vergleichen drei reduzierte Feature-",
        "Sets und stellen die LightGBM-Parameter konservativer ein (`min_child_samples`, ",
        "`reg_alpha`, `reg_lambda`). Ziel: derselbe Eval-RMSE-Bereich, aber kleinerer ",
        "Train-/Eval-Gap.\n"
    ],

    "## 28. Wide Pipeline": [
        "### 19.D Wide Pipeline: weniger Features, dafür fast doppelt so viele Datenpunkte\n",
        "\n",
        "Ein letzter Versuch: was passiert, wenn wir auf die GWR-/swisstopo-",
        "Enrichment-Felder verzichten und stattdessen alle Listings nutzen können, bei ",
        "denen nur `area`, `rooms`, `east`, `north` und `price_cold` vorhanden sind? ",
        "Wir bekommen ~9.5k Zeilen statt ~4.5k und können prüfen, ob die zusätzlichen ",
        "Daten den Verlust an Features kompensieren.\n"
    ],

    "## Schluss & nächste Schritte": [
        "## Schluss & nächste Schritte\n",
        "\n",
        "Damit ist die Pipeline abgeschlossen. Die wichtigsten Erkenntnisse:\n",
        "\n",
        "- Tree-basierte Modelle schlagen Ridge auf diesem Datensatz deutlich.\n",
        "- LightGBM mit KNN-Features liegt am besten (Eval-RMSE 393 CHF, R² 0.751).\n",
        "- Im teuersten Preis-Quartil ist der Fehler ungefähr doppelt so hoch — hier ",
        "fehlen uns Features wie Stockwerk, Aussicht und Ausbau.\n",
        "- Für DSPRO2 wäre der nächste sinnvolle Schritt, die Beschreibungstexte als ",
        "Quelle für genau diese Features zu nutzen, und eine zweite Plattform ",
        "(Homegate/Immoscout) anzubinden.\n"
    ],
}


# ---------------------------------------------------------------------------
# 2) Patterns identifying the cells we want to drop
# ---------------------------------------------------------------------------
#
# Two families of drops:
#   A) Always dropped — clearly experimental / demo content that does
#      not belong in the final hand-in.
#   B) Compression drops — long, exploratory chapters that overlap with
#      the main pipeline. The script removes the entire chapter
#      (starting from the matching "## " heading until the next
#      "## " heading at the same level).

# A) Always dropped
DROP_IF_SOURCE_CONTAINS: list[str] = [
    # Whole chapter 14c (Demo: Underfitting / Good Fit / Overfitting)
    "14c. Demo: Underfitting / Good Fit / Overfitting",
    "Polynomial Regression als Demo",
]

# B) Compression chapters — dropped only when COMPRESS_NOTEBOOK = True.
# Each entry is a unique substring of the chapter heading that starts
# the section we want to remove. The section ends at the next markdown
# cell that begins with "## " (and is *not* in this list).
COMPRESS_NOTEBOOK = True

COMPRESS_DROP_CHAPTERS: list[str] = [
    # Patterns must match the heading exactly as it appears NOW
    # (i.e. after MARKDOWN_REWRITES has been applied). The previous
    # run of this script already removed many one-off chapters; what
    # remains below targets headings that are still present in the
    # notebook.
    # ----------------------------------------------------------------------
    # Decorative or duplicate visualisations
    # ----------------------------------------------------------------------
    "## 14b. Prediction Curves",          # decorative; covered by 14 + 23.7
    "### 18.1 Finale Prediction",          # redundant with 24.3 sanity checks
    "### 18.2 Optional: Group Split",      # diagnostic only
    # ----------------------------------------------------------------------
    # Chapter 18 (Geo) — keep core EDA + clustering + final geo model
    # ----------------------------------------------------------------------
    "### 21.7 Lineare Modelle: `geo_cluster`",  # only relevant for Ridge
    "### 21.9 Geo-Prediction-Kurve",            # decorative
    "### 21.10 Zusammenfassung Geo",            # redundant prose
    # ----------------------------------------------------------------------
    # Chapter 19 (Iterative Verbesserungen) — keep 22.3 (tuning),
    # 22.5 (learning curve, figure!) and 22.8 (SHAP, figure!).
    # 22.15 used to consolidate the variants from 22.1/22.2/22.4/22.10,
    # but since those chapters are dropped, 22.15 would crash on a
    # fresh kernel restart (NameError on imp_results / results_log /
    # res_strat / res_rfecv). The comprehensive comparison is now in
    # 23.17 (which is kept), so we also drop 22.15.
    # ----------------------------------------------------------------------
    "### 22.1 Missing-Data-Diagnose",
    "### 22.2 Log-transformiertes Target",
    "### 22.4 Stratifizierter Train/Eval-Split",
    "### 22.9 IsolationForest",
    "### 22.15 Konsolidierter Modellvergleich",
    # ----------------------------------------------------------------------
    # 19.C Overfitting reduzieren — the recommendation is already in
    # the report (Section "Reducing Overfit"); keep the comparison
    # but drop the prose-only subsections.
    # ----------------------------------------------------------------------
    "### 27.1 Drei Feature-Sets im Vergleich",
    "### 27.4 Interpretation und Empfehlung",
]


# ---------------------------------------------------------------------------
# 3) Code edits  (substring -> replacement)
# ---------------------------------------------------------------------------
CODE_EDITS: list[tuple[str, str]] = [
    # Importance heatmap: sort descending so the most important features end
    # up at the *top* of the heatmap, not at the bottom.
    (
        "imp_norm = imp_norm.sort_values(by=imp_norm.columns[0], ascending=True)",
        "imp_norm = imp_norm.sort_values(by=imp_norm.columns[0], ascending=False)"
    ),

    # Scatter/highlight points: rot -> blau. Wir wollen die Punkt-Wolken
    # neutral darstellen; rote Highlight-Punkte verleiten den Leser zu der
    # Annahme, dort sei ein Fehler. Blau ist die natürlichere Wahl.
    (
        "color='red', s=50, marker='X', label='Top 10 abs. Fehler'",
        "color='#2980B9', s=50, marker='X', label='Top 10 abs. Fehler'",
    ),

    # Reference lines (median, zero-residual) bleiben rot, weil sie\n
    # tatsächlich eine 'Achtung'-Linie sind und nicht den Datenpunkt
    # darstellen — also lassen wir diese Stellen unverändert.

    # Inline-Export: das Speichern soll möglichst nahe an der jeweiligen
    # Plot-Erzeugung passieren. Wenn eine Zelle schon ein
    # `save_fig(...)`-Aufruf enthält, ist alles in Ordnung. Ist sie
    # plt.show()-only, sollte manuell ein save_fig direkt davor ergänzt
    # werden — dafür ist der Coverage-Check unten.

    # ----------------------------------------------------------------------
    # Konsolidierter Vergleich (23.17) — self-contained machen.
    #
    # Kapitel 22.15 wurde im Cleanup entfernt; damit existieren `summary_blocks`
    # (Kap. 22-Stand) und `HAS_HALVING` / `res_halving` (Kap. 23.10) nicht mehr.
    # 23.17 referenziert sie aber immer noch und crasht beim Lauf-aus-dem-Stand
    # mit NameError. Wir bauen die Vergleichsliste in 23.17 hier neu auf, defensiv
    # gegen alle Varianten, die ggf. fehlen.
    (
        "summary_blocks2 = list(summary_blocks)  # Kapitel 22-Stand\n"
        "\n"
        "if len(stack_fitted) > 0:\n"
        "    summary_blocks2.append(stack_results.assign(Variant='Stacking (23.2)'))\n"
        "if has_full_geo:\n"
        "    summary_blocks2.append(res_knn.assign(Variant='KNN-Distance (23.12)'))\n"
        "if HAS_LGBM and HAS_HALVING:\n"
        "    summary_blocks2.append(res_halving.assign(Variant='LGBM Halving-Tuned (23.10)'))",
        "# Kapitel 22.15 wurde im Cleanup entfernt — das ursprüngliche\n"
        "# `summary_blocks` existiert in dieser Notebook-Version nicht mehr.\n"
        "# Wir bauen die Vergleichsliste hier selbst aus den Varianten zusammen,\n"
        "# die im bereinigten Notebook tatsächlich noch existieren.\n"
        "summary_blocks2 = [main_results.assign(Variant='Baseline (Kap. 10)')]\n"
        "\n"
        "if HAS_LGBM and 'res_tuned' in globals() and tuned_lgbm is not None:\n"
        "    summary_blocks2.append(res_tuned.assign(Variant='Tuned LGBM (22.3)'))\n"
        "\n"
        "if len(stack_fitted) > 0:\n"
        "    summary_blocks2.append(stack_results.assign(Variant='Stacking (23.2)'))\n"
        "if has_full_geo and 'res_knn' in globals():\n"
        "    summary_blocks2.append(res_knn.assign(Variant='KNN-Distance (23.12)'))\n"
        "if HAS_LGBM and globals().get('HAS_HALVING', False) and 'res_halving' in globals():\n"
        "    summary_blocks2.append(res_halving.assign(Variant='LGBM Halving-Tuned (23.10)'))",
    ),

    # ----------------------------------------------------------------------
    # Kapitel 27.2 — self-contained machen.
    #
    # Kapitel 27.1 (das FEATURES_MINIMAL / FEATURES_TOP6 / FEATURES_FULL_KNN
    # sowie _train_df_knn / _eval_df_knn definiert) wurde im Cleanup gedroppt.
    # 27.2 referenziert die Variablen aber weiter und crasht aus dem Stand mit
    # NameError. Wir prependen die Definitionen direkt in die 27.2-Zelle.
    (
        "# === 27.2 Regularisierte LightGBM auf MINIMAL, TOP6, FULL+KNN ===\n",
        "# === 27.2 Regularisierte LightGBM auf MINIMAL, TOP6, FULL+KNN ===\n"
        "# Kapitel 27.1 wurde im Cleanup entfernt — die Feature-Sets, die in 27.2\n"
        "# und 27.3 verwendet werden, definieren wir hier selbst, damit die Zelle\n"
        "# self-contained ist.\n"
        "FEATURES_MINIMAL = ['area', 'knn_price_median', 'east', 'north']\n"
        "FEATURES_TOP6    = ['area', 'knn_price_median', 'knn_price_mean',\n"
        "                     'east', 'north', 'area_per_room']\n"
        "FEATURES_FULL_KNN = FEATURES_ENGINEERED + ['knn_price_mean', 'knn_price_median']\n"
        "\n"
        "# Datenbasis: train_df_knn / eval_df_knn enthalten die KNN-Features (Kap. 23.12);\n"
        "# Fallback auf train_df / eval_df, falls die KNN-Erweiterung übersprungen wurde.\n"
        "_train_df_knn = train_df_knn if 'train_df_knn' in dir() else train_df\n"
        "_eval_df_knn  = eval_df_knn  if 'eval_df_knn'  in dir() else eval_df\n"
        "\n",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return src


def _compress_drop_match(src: str) -> int:
    """Return the heading depth to drop (2 or 3) if `src` starts a
    compression-listed chapter, otherwise 0.

    H2 patterns ("## X.") drop until the next H2 heading.
    H3 patterns ("### X.Y") drop until the next H3 or H2 heading.
    """
    if not COMPRESS_NOTEBOOK:
        return 0
    for pat in COMPRESS_DROP_CHAPTERS:
        if pat in src:
            # Patterns starting with "### " are H3-level drops; "## " is H2.
            if pat.lstrip().startswith("### "):
                return 3
            return 2
    return 0


def should_drop(
    cell: dict,
    in_drop_chapter: bool,
    drop_depth: int,
) -> tuple[bool, bool, int]:
    """
    Returns (drop, in_drop_chapter_after, drop_depth_after).

    Decision order:
      1. If we are currently in a drop window and this cell is a heading
         that closes it, close first.  Then re-evaluate whether the
         closing heading itself starts a new drop window — otherwise we
         would lose drops of two adjacent compression chapters (e.g.
         "## 19." followed by "## 20.", both in the drop list).
      2. Always-drop patterns from DROP_IF_SOURCE_CONTAINS start a
         chapter-wide drop that ends at the next "## " heading.
      3. Compression matches start a drop whose depth comes from the
         pattern (H2 → close at next H2; H3 → close at next H3 or H2).
      4. Plain content cells inside an active drop window are dropped
         silently.
    """
    src = cell_source(cell)
    cell_type = cell.get("cell_type")
    is_h2 = cell_type == "markdown" and src.lstrip().startswith("## ") \
            and not src.lstrip().startswith("### ")
    is_h3 = cell_type == "markdown" and src.lstrip().startswith("### ")

    # Step 1: handle the close of an existing drop window.
    if in_drop_chapter:
        closes = (is_h2) or (is_h3 and drop_depth == 3)
        if closes:
            in_drop_chapter = False
            drop_depth = 0
            # fall through — the closing heading might *itself* start a
            # new drop, so we keep evaluating.

    # Step 2: always-drop start (## 14c ...)
    if any(pat in src for pat in DROP_IF_SOURCE_CONTAINS):
        return True, True, 2

    # Step 3: compression drop start.
    compress_depth = _compress_drop_match(src)
    if compress_depth:
        return True, True, compress_depth

    # Step 4: still inside an unclosed drop window.
    if in_drop_chapter:
        return True, in_drop_chapter, drop_depth

    return False, False, 0


def apply_markdown_rewrite(cell: dict) -> None:
    if cell.get("cell_type") != "markdown":
        return
    src = cell_source(cell)
    for key, new_lines in MARKDOWN_REWRITES.items():
        if key in src:
            cell["source"] = new_lines
            return


def apply_code_edits(cell: dict) -> None:
    if cell.get("cell_type") != "code":
        return
    src = cell_source(cell)
    new_src = src
    for old, new in CODE_EDITS:
        if old in new_src:
            new_src = new_src.replace(old, new)
    if new_src != src:
        cell["source"] = new_src.splitlines(keepends=True)


# ---------------------------------------------------------------------------
# Uniform design: a single matplotlib style cell injected after the imports
# ---------------------------------------------------------------------------
#
# After this cell runs, every subsequent plt.figure / sns.heatmap / etc. uses
# the same style, figsize and colour cycle, so the notebook stops looking like
# "twelve different design systems stitched together".

STYLE_CELL_SENTINEL = "# === Uniform plotting style for the whole notebook"

STYLE_CELL_SOURCE = [
    "# === Uniform plotting style for the whole notebook ===\n",
    "# Wir setzen hier einmal zentral die Style-Defaults, damit jeder Plot\n",
    "# danach denselben Look hat (gleiche Schriftgrösse, gleiche Farben,\n",
    "# gleiche Achsen-Spines). Spart später viel `figsize=...`-Geschwurbel.\n",
    "#\n",
    "# Primärfarbe ist absichtlich Blau (statt Rot). Rot wird in Reports oft\n",
    "# als 'Warnung' interpretiert; unsere Punkt-Wolken sollen aber neutral\n",
    "# wirken, damit die Aussage in der Form des Scatters liegt, nicht in\n",
    "# der Farbe.\n",
    "import matplotlib as mpl\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "sns.set_theme(\n",
    "    context='notebook',\n",
    "    style='whitegrid',\n",
    "    palette='deep',\n",
    "    font='DejaVu Sans',\n",
    "    rc={\n",
    "        'figure.figsize':   (8.0, 4.8),\n",
    "        'figure.dpi':       110,\n",
    "        'savefig.dpi':      150,\n",
    "        'savefig.bbox':     'tight',\n",
    "        'axes.titlesize':   12,\n",
    "        'axes.labelsize':   11,\n",
    "        'axes.spines.top':   False,\n",
    "        'axes.spines.right': False,\n",
    "        'legend.frameon':    False,\n",
    "        'grid.alpha':        0.35,\n",
    "        'scatter.edgecolors': 'none',\n",
    "    },\n",
    ")\n",
    "# Color cycle: blau zuerst, dann andere neutrale Töne.\n",
    "mpl.rcParams['axes.prop_cycle'] = mpl.cycler(\n",
    "    color=['#2980B9', '#2C3E50', '#27AE60', '#F39C12',\n",
    "           '#8E44AD', '#16A085', '#7F8C8D', '#C0392B']\n",
    ")\n",
    "# Default scatter colour wird oft aus C0 geholt; viele Notebook-Zellen\n",
    "# setzen aber explizit `color='red'` oder `c='red'` für die Punkte. Das\n",
    "# CODE_EDITS-Mapping in humanize_notebook.py ersetzt diese Stellen, damit\n",
    "# die Scatterplots blau statt rot werden.\n",
]


def inject_style_cell(nb: dict) -> bool:
    """Insert the uniform-style cell directly after the imports cell.

    Idempotent: if the sentinel is already present anywhere in the
    notebook, nothing is added.
    """
    cells = nb["cells"]
    for c in cells:
        if STYLE_CELL_SENTINEL in cell_source(c):
            return False  # already injected

    # Find the imports cell (the first code cell that imports pandas/numpy).
    insert_after = None
    for i, c in enumerate(cells):
        if c.get("cell_type") != "code":
            continue
        src = cell_source(c)
        if "import pandas" in src or "import numpy" in src:
            insert_after = i
            break
    if insert_after is None:
        return False

    style_cell = {
        "cell_type": "code",
        "metadata": {},
        "source": STYLE_CELL_SOURCE,
        "outputs": [],
        "execution_count": None,
    }
    cells.insert(insert_after + 1, style_cell)
    return True


# ---------------------------------------------------------------------------
# PNG-export sanity check: which figures the final report expects vs. which
# the export cell actually writes
# ---------------------------------------------------------------------------

REPORT_REQUIRED_PNGS = [
    # Core figures of the report
    "geo_price_map.png",
    "price_distribution.png",
    "geo_clusters.png",
    "rmse_train_eval.png",
    "bootstrap_ci.png",
    "actual_vs_predicted.png",
    "residuals.png",
    "qq_plot.png",
    "error_by_priceband.png",
    "feature_importance_heatmap.png",
    "pdp_top6.png",
    # Wide-pipeline figures (added with Chapter 28)
    "wide_vs_standard.png",
    "wide_actual_vs_predicted.png",
    # New round of figures added in the second pass over the report
    "learning_curve.png",          # Chapter 22.5 — produces it; needs save_fig
    "shap_summary.png",            # Chapter 22.8 — produces it; needs save_fig
    "spatial_residuals.png",       # New: residual magnitude on a Switzerland map
    "feature_correlation.png",     # New: heatmap of correlations on the model set

    # Streamlit-app walkthrough screenshots (manually placed in fig/).
    # Filenames are timestamps from when the user saved them; we keep
    # them as-is so they line up with the file names on disk.
    "2026-05-19_13h23_27.png",     # Step 1 — Modell auswählen
    "2026-05-19_13h22_41.png",     # Step 2 — Adresse suchen
    "2026-05-19_13h23_12.png",     # Step 3 — Wohnung auswählen
    "2026-05-19_13h24_23.png",     # Step 4 — Auswertung / Model Performance
    "2026-05-19_13h23_45.png",     # Manual fallback (1/2): structural sliders
    "2026-05-19_13h23_59.png",     # Manual fallback (2/2): location sliders
]


def check_png_export_coverage(nb: dict) -> list[str]:
    """Return the list of PNG filenames that the report needs but that
    are *not* referenced anywhere in the notebook's code cells.

    We deliberately do not try to auto-generate the export code — the
    actual plotting calls live in the notebook's in-memory state and
    cannot be reconstructed from text alone — but we do warn loudly so
    that any missing export is fixed manually.
    """
    all_code = "\n".join(
        cell_source(c) for c in nb["cells"] if c.get("cell_type") == "code"
    )
    missing = [png for png in REPORT_REQUIRED_PNGS if png not in all_code]
    return missing


def list_resulting_structure(nb: dict, limit: int = 40) -> list[str]:
    """Return the post-cleanup chapter outline (H2 and H3 headings) so
    the user can quickly verify that the right things were kept and
    the right things removed."""
    headings = []
    for c in nb["cells"]:
        if c.get("cell_type") != "markdown":
            continue
        src = cell_source(c).lstrip()
        if src.startswith("## ") and not src.startswith("### "):
            first_line = src.splitlines()[0].rstrip()
            headings.append(first_line)
        elif src.startswith("### "):
            first_line = src.splitlines()[0].rstrip()
            headings.append("  " + first_line)
    if len(headings) > limit:
        return headings[: limit - 1] + [f"  ... and {len(headings) - limit + 1} more"]
    return headings


def check_inline_savefig(nb: dict) -> list[str]:
    """Find code cells that create a figure (plt.figure / fig, ax = ...)
    but never call `save_fig`, `fig.savefig`, or `plt.savefig`.

    These are the cells where the user has explicitly asked for an
    inline export: instead of relying on the consolidated Chapter 21
    export at the end, every plotting cell should save its own PNG
    next to where it is created.  The script does not auto-edit these
    cells (the figure name is project-specific), but it prints a list
    of the first ~80 characters of each such cell so they can be
    spotted in the notebook quickly.
    """
    PLOT_HINTS  = ("plt.figure(", "fig, ax", "fig, axes", "fig = plt.figure")
    SAVE_HINTS  = ("save_fig(", "fig.savefig(", "plt.savefig(")
    EXPORT_CELL = "Export der Figures"  # the consolidated export markdown

    seen_export_marker = False
    offenders: list[str] = []
    for c in nb["cells"]:
        src = cell_source(c)
        if EXPORT_CELL in src:
            seen_export_marker = True
            continue
        if seen_export_marker:
            continue  # cells past the consolidated export are exempt
        if c.get("cell_type") != "code":
            continue
        if not any(h in src for h in PLOT_HINTS):
            continue
        if any(h in src for h in SAVE_HINTS):
            continue
        # Strip leading whitespace lines for a more useful preview.
        preview = " | ".join(
            line.strip() for line in src.splitlines() if line.strip()
        )[:120]
        offenders.append(preview)
    return offenders


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not NB_PATH.exists():
        raise SystemExit(f"Notebook not found: {NB_PATH}")

    # Backup
    if not BACKUP_PATH.exists():
        shutil.copy2(NB_PATH, BACKUP_PATH)
        print(f"Backup written to {BACKUP_PATH}")

    with NB_PATH.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    new_cells: list[dict] = []
    in_drop_chapter = False
    drop_depth = 0
    dropped = 0
    rewritten_md = 0
    rewritten_code = 0

    for cell in nb["cells"]:
        drop, in_drop_chapter, drop_depth = should_drop(
            cell, in_drop_chapter, drop_depth
        )
        if drop:
            dropped += 1
            continue

        before = cell_source(cell)
        apply_markdown_rewrite(cell)
        apply_code_edits(cell)
        after = cell_source(cell)

        if before != after:
            if cell.get("cell_type") == "markdown":
                rewritten_md += 1
            else:
                rewritten_code += 1

        new_cells.append(cell)

    nb["cells"] = new_cells

    # Uniform-design: inject the one shared matplotlib style cell.
    style_injected = inject_style_cell(nb)

    # Sanity check the PNG-export coverage.
    missing_pngs = check_png_export_coverage(nb)

    # Sanity check inline savefig coverage.
    plot_cells_without_save = check_inline_savefig(nb)

    with NB_PATH.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"Dropped cells       : {dropped}")
    print(f"Rewritten markdown  : {rewritten_md}")
    print(f"Rewritten code      : {rewritten_code}")
    print(f"Style cell injected : {style_injected}")
    if missing_pngs:
        print()
        print("WARNING: the following PNG files are referenced by the final report")
        print("but no code cell in the notebook writes them. Add them to the export")
        print("cell at the end (Chapter 21 'Export der Figures') manually:")
        for png in missing_pngs:
            print(f"    - {png}")
    else:
        print("PNG export coverage : OK (all report PNGs are produced)")

    if plot_cells_without_save:
        print()
        print("NOTE: the following plotting cells do NOT save their figure inline.")
        print("Consider adding a `save_fig('<name>.png', fig)` directly under the")
        print("plot — that way each plot's PNG is exported the moment it is")
        print(f"created, instead of relying on the bulk export at the end")
        print(f"({len(plot_cells_without_save)} cells found):")
        for prev in plot_cells_without_save[:20]:
            print(f"    - {prev}")
        if len(plot_cells_without_save) > 20:
            print(f"    ... and {len(plot_cells_without_save) - 20} more")
    else:
        print("Inline savefig      : OK (every plot cell saves its own PNG)")

    # Show the resulting notebook outline so the user can spot anything
    # that still looks wrong.
    print()
    print("Resulting outline:")
    for line in list_resulting_structure(nb):
        print("    " + line)

    print(f"Saved cleaned-up notebook to {NB_PATH}")


if __name__ == "__main__":
    main()
