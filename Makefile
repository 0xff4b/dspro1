# ---------------------------------------------------------------------------
# DSPRO1 — Predicting Apartment Rental Prices in Switzerland
# Team 8 — Elias Martinelli & Timo Schlumpf
# ---------------------------------------------------------------------------
#
# Häufige Workflows als `make`-Targets gebündelt — damit die Befehlsketten
# nicht jedes Mal aus dem Kopf zusammengesucht werden müssen.
#
#   make help            kurze Liste aller Targets
#   make report          baut das Final-Report-PDF (pdflatex + bibtex + 2x pdflatex)
#   make app             startet die Streamlit-Demo-App (mit cache clear)
#   make notebook-fix    wendet die Post-Cleanup-Patches auf model_v3_clean.ipynb an
#   make notebook-clean  re-run von humanize_notebook.py (mit Backup)
#   make clean-report    räumt LaTeX-Aux-Dateien im final-report-Ordner auf
#
# Konvention: alle Targets gehen davon aus, dass `make` aus dem Repo-Root
# gestartet wird (also aus dem Ordner, in dem dieses Makefile liegt).
# ---------------------------------------------------------------------------

# ---- Konfiguration --------------------------------------------------------
REPORT_DIR   := docs/final-report
REPORT_NAME  := DISPRO1_FinalReport_Team8_PredictingApartmentRentalPrices_DRAFT
REPORT_TEX   := $(REPORT_NAME).tex
REPORT_PDF   := $(REPORT_NAME).pdf

NOTEBOOK_DIR := src/notebooks
APP          := src/app.py

PDFLATEX     := pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX       := bibtex

PYTHON       ?= python

# .PHONY: keine dieser Targets erzeugt eine gleichnamige Datei.
.PHONY: help report app notebook-fix notebook-clean clean-report clean-all

# ---- help (default) -------------------------------------------------------
help:
	@echo "Verfügbare Targets:"
	@echo "  make report          Final-Report-PDF bauen (pdflatex + bibtex + 2x pdflatex)"
	@echo "  make app             Streamlit-Demo starten (mit cache clear)"
	@echo "  make notebook-fix    fix_summary_blocks.py auf model_v3_clean.ipynb anwenden"
	@echo "  make notebook-clean  humanize_notebook.py erneut laufen lassen"
	@echo "  make clean-report    LaTeX-Aux-Dateien (.aux, .log, .toc, ...) löschen"
	@echo "  make clean-all       clean-report + LaTeX-PDF entfernen"
.DEFAULT_GOAL := help

# ---- Final-Report ---------------------------------------------------------
# pdflatex zweimal nach bibtex laufen lassen, damit Querverweise + Bib-Einträge
# sauber aufgelöst sind. -halt-on-error: bei TeX-Fehler sofort raus, sonst
# verschwindet die eigentliche Fehlermeldung in 200 Zeilen Folgewarnungen.
report:
	cd $(REPORT_DIR) && $(PDFLATEX) $(REPORT_TEX)
	cd $(REPORT_DIR) && $(BIBTEX)   $(REPORT_NAME)
	cd $(REPORT_DIR) && $(PDFLATEX) $(REPORT_TEX)
	cd $(REPORT_DIR) && $(PDFLATEX) $(REPORT_TEX)
	@echo
	@echo "PDF gebaut: $(REPORT_DIR)/$(REPORT_PDF)"

# ---- Streamlit-App --------------------------------------------------------
# `streamlit cache clear` vorweg, damit nach einem Modell-Retrain die App
# garantiert mit dem neuen Joblib-Stand startet und nicht aus dem Cache lebt.
app:
	streamlit cache clear
	streamlit run $(APP)

# ---- Notebook-Maintenance -------------------------------------------------
notebook-fix:
	cd $(NOTEBOOK_DIR) && $(PYTHON) fix_summary_blocks.py

notebook-clean:
	cd $(NOTEBOOK_DIR) && $(PYTHON) humanize_notebook.py

# ---- Cleanup --------------------------------------------------------------
# LaTeX-Müll, der bei jedem pdflatex-Lauf entsteht. Das PDF selbst bleibt liegen,
# damit ein versehentliches `make clean-report` nicht die fertige Abgabe killt.
clean-report:
	cd $(REPORT_DIR) && rm -f \
		$(REPORT_NAME).aux \
		$(REPORT_NAME).bbl \
		$(REPORT_NAME).blg \
		$(REPORT_NAME).fdb_latexmk \
		$(REPORT_NAME).fls \
		$(REPORT_NAME).log \
		$(REPORT_NAME).out \
		$(REPORT_NAME).toc

clean-all: clean-report
	cd $(REPORT_DIR) && rm -f $(REPORT_PDF)
