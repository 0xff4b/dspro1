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
#   make app             startet die Streamlit-Demo-App (mit Projektumgebung)
#
# Konvention: alle Targets gehen davon aus, dass `make` aus dem Repo-Root
# gestartet wird (also aus dem Ordner, in dem dieses Makefile liegt).
# ---------------------------------------------------------------------------

# ---- Konfiguration --------------------------------------------------------
REPORT_DIR   := docs/dspro1/final-report
REPORT_NAME  := DISPRO1_FinalReport_Team8_PredictingApartmentRentalPrices
REPORT_TEX   := $(REPORT_NAME).tex
REPORT_PDF   := $(REPORT_NAME).pdf

PYTHON       ?= python3.12

PDFLATEX     := pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX       := bibtex

# .PHONY: keine dieser Targets erzeugt eine gleichnamige Datei.
.PHONY: help report setup app notebook doctor

# ---- help (default) -------------------------------------------------------
help:
	@echo "Verfügbare Targets:"
	@echo "  make report    Final-Report-PDF bauen (pdflatex + bibtex + 2x pdflatex)"
	@echo "  make app       Umgebung vorbereiten und Streamlit starten"
	@echo "  make setup     Vollstaendige Python-Umgebung vorbereiten"
	@echo "  make notebook  Umgebung vorbereiten und JupyterLab starten"
	@echo "  make doctor    Installation ohne Aenderungen pruefen"
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

# ---- Lokale Umgebung und Anwendungen --------------------------------------
setup:
	$(PYTHON) setup.py

app:
	$(PYTHON) setup.py --profile app --start app

notebook:
	$(PYTHON) setup.py --start notebook

doctor:
	$(PYTHON) setup.py --doctor
