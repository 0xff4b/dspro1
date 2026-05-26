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
#   make app             startet die Streamlit-Demo-App (mit cache clear = effektiv restart)
#
# Konvention: alle Targets gehen davon aus, dass `make` aus dem Repo-Root
# gestartet wird (also aus dem Ordner, in dem dieses Makefile liegt).
# ---------------------------------------------------------------------------

# ---- Konfiguration --------------------------------------------------------
REPORT_DIR   := docs/final-report
REPORT_NAME  := DISPRO1_FinalReport_Team8_PredictingApartmentRentalPrices_DRAFT
REPORT_TEX   := $(REPORT_NAME).tex
REPORT_PDF   := $(REPORT_NAME).pdf

APP          := src/app.py

PDFLATEX     := pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX       := bibtex

# .PHONY: keine dieser Targets erzeugt eine gleichnamige Datei.
.PHONY: help report app

# ---- help (default) -------------------------------------------------------
help:
	@echo "Verfügbare Targets:"
	@echo "  make report    Final-Report-PDF bauen (pdflatex + bibtex + 2x pdflatex)"
	@echo "  make app       Streamlit-Demo starten / restarten (mit cache clear)"
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
# garantiert mit dem neuen Joblib-Stand startet und nicht aus dem Cache lebt
# — effektiv ein Restart.
app:
	streamlit cache clear
	streamlit run $(APP)
