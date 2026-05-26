"""Streamlit-Demo fuer unser DSPRO1-Mietpreis-Modell.

Starten mit:
    streamlit run src/app.py

Die App ist im Grunde eine grafische Oberflaeche fuer die `RentPredictor`-
Klasse aus dem Notebook (Kapitel 24.2). Beim ersten Start wird das Modell
einmal trainiert und in einem Streamlit-Cache abgelegt; jeder weitere
Klick laeuft dann ohne Re-Training.

Den eigentlichen Ablauf habe ich 1:1 aus dem Notebook uebernommen, damit
die Vorhersage hier identisch ist wie das, was wir im Bericht zeigen:

    Adresse  -->  EGID / Koordinaten (geo.admin SearchServer)
    EGID     -->  GWR-Gebaeudedaten (gbauj, ganzwhg, garea)
    Koord.   -->  Swisstopo-Layer (Hoehe, OeV, Solar, Population)
    Cleaning wie in final_records.ipynb
    Predict via RentPredictor.predict(...)

Wenn jemand das Repo neu checkt: einfach `streamlit run src/app.py`
ausfuehren, die App holt sich `model.csv` automatisch.

— Elias, Team 8
"""
from __future__ import annotations

import datetime as dt
import math
import re
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


# --- Pfade ------------------------------------------------------------------
# Wir lassen die App relativ zum Repo-Root suchen, damit ich sie sowohl lokal
# als auch in einer aufgeraeumten Repo-Kopie starten kann.
APP_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DATA_PATH    = APP_DIR / "external-sources" / "output_csv" / "model.csv"
MODEL_PATH   = PROJECT_ROOT / "models" / "rent_predictor_streamlit.joblib"


# --- RentPredictor ----------------------------------------------------------
# Das ist exakt dieselbe Klasse wie im Notebook (Kap. 24.2). Ich habe sie hier
# noch einmal inline, damit `app.py` ohne den Notebook-Code direkt laufen
# kann — sonst muesste ich beim Demo-Setup auch noch das Notebook
# importieren, und das ist es mir nicht wert.
class RentPredictor:
    """End-to-End-Pipeline fuer die Mietpreis-Vorhersage.

    Reihenfolge im fit:
        clean -> engineer -> geo-cluster -> knn-features -> imputer -> model

    Im predict laeuft dieselbe Pipeline, aber natuerlich ohne neues Fitten.
    """

    def __init__(
        self,
        target_col: str = "price",
        feature_cols=None,
        n_geo_clusters: int = 8,
        knn_k: int = 10,
        outlier_min_area: int = 10,
        outlier_min_price: int = 300,
        reference_year: int = 2026,
        random_state: int = 42,
        model=None,
    ):
        self.target_col        = target_col
        self.feature_cols      = feature_cols
        self.n_geo_clusters    = n_geo_clusters
        self.knn_k             = knn_k
        self.outlier_min_area  = outlier_min_area
        self.outlier_min_price = outlier_min_price
        self.reference_year    = reference_year
        self.random_state      = random_state
        self.model             = model

    def _clean(self, df, training=True):
        out = df.copy()
        if training:
            mask = out["area"] >= self.outlier_min_area
            if self.target_col in out.columns:
                mask &= out[self.target_col] >= self.outlier_min_price
            out = out[mask].drop_duplicates().reset_index(drop=True)
        return out

    def _engineer(self, df):
        out = df.copy()
        if "year_built" in out.columns:
            out["building_age"] = self.reference_year - out["year_built"]
        if "rooms" in out.columns and "area" in out.columns:
            out["area_per_room"] = np.where(
                out["rooms"] > 0, out["area"] / out["rooms"], np.nan
            )
        if "apartments" in out.columns and "land_area" in out.columns:
            out["land_area_per_apartment"] = np.where(
                out["apartments"] > 0, out["land_area"] / out["apartments"], np.nan,
            )
        return out

    def _geo_cluster(self, df, fit=False):
        cols = ["east", "north"]
        if not all(c in df.columns for c in cols):
            return df
        if fit:
            self._geo_pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("km", KMeans(n_clusters=self.n_geo_clusters,
                              random_state=self.random_state, n_init=10)),
            ])
            self._geo_pipe.fit(df[cols])
        out = df.copy()
        out["geo_cluster"] = self._geo_pipe.predict(out[cols])
        return out

    def _knn_features(self, df, fit=False):
        cols = ["east", "north"]
        if not all(c in df.columns for c in cols):
            return df
        if fit:
            self._coord_scaler = StandardScaler().fit(df[cols])
            self._train_coords = self._coord_scaler.transform(df[cols])
            self._train_prices = df[self.target_col].values
            self._nbrs = NearestNeighbors(n_neighbors=self.knn_k + 1).fit(
                self._train_coords
            )
            _, idx = self._nbrs.kneighbors(self._train_coords)
            idx = idx[:, 1:]
            out = df.copy()
            out["knn_price_mean"]   = self._train_prices[idx].mean(axis=1)
            out["knn_price_median"] = np.median(self._train_prices[idx], axis=1)
            return out
        coords_q = self._coord_scaler.transform(df[cols])
        _, idx = self._nbrs.kneighbors(coords_q, n_neighbors=self.knn_k)
        out = df.copy()
        out["knn_price_mean"]   = self._train_prices[idx].mean(axis=1)
        out["knn_price_median"] = np.median(self._train_prices[idx], axis=1)
        return out

    def fit(self, df):
        df_c = self._clean(df, training=True)
        df_e = self._engineer(df_c)
        df_g = self._geo_cluster(df_e, fit=True)
        df_k = self._knn_features(df_g, fit=True)

        if self.feature_cols is None:
            base = ["east", "north", "elevation", "area", "rooms", "year_built",
                    "apartments", "land_area", "population", "oev", "solar",
                    "building_age", "area_per_room", "land_area_per_apartment",
                    "geo_cluster", "knn_price_mean", "knn_price_median"]
            self.feature_cols = [c for c in base if c in df_k.columns]

        self._imputer = SimpleImputer(strategy="median")
        X = self._imputer.fit_transform(df_k[self.feature_cols])
        y = df_k[self.target_col].values

        if self.model is None:
            if HAS_LGBM:
                self.model = LGBMRegressor(
                    n_estimators=500, learning_rate=0.05,
                    random_state=self.random_state, n_jobs=-1, verbose=-1,
                )
            else:
                self.model = RandomForestRegressor(
                    n_estimators=300, random_state=self.random_state,
                    n_jobs=-1, min_samples_leaf=2,
                )
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, df):
        if not getattr(self, "_is_fitted", False):
            raise RuntimeError("RentPredictor wurde noch nicht gefittet.")
        df_c = self._clean(df, training=False)
        df_e = self._engineer(df_c)
        df_g = self._geo_cluster(df_e, fit=False)
        df_k = self._knn_features(df_g, fit=False)
        for c in self.feature_cols:
            if c not in df_k.columns:
                df_k[c] = np.nan
        X_arr = self._imputer.transform(df_k[self.feature_cols])

        # Falls ein Modell mit Feature-Namen trainiert wurde (z.B. LightGBM),
        # geben wir auch beim Predict ein DataFrame mit denselben Namen weiter.
        if hasattr(self.model, "feature_names_in_"):
            X_df = pd.DataFrame(X_arr, columns=list(self.feature_cols), index=df_k.index)
            ordered = list(self.model.feature_names_in_)
            for col in ordered:
                if col not in X_df.columns:
                    X_df[col] = np.nan
            return self.model.predict(X_df[ordered])

        return self.model.predict(X_arr)


# --- Pipeline-Funktionen ----------------------------------------------------
# Reihenfolge: Adresse -> EGID -> GWR -> Swisstopo -> Features
# Logik 1:1 aus den drei Notebooks:
#   - gwr_egid_db_sync.ipynb         (search_address, fetch_gwr_feature)
#   - swisstopo_enrich_db_sync_v2    (geocode, get_elevation, identify, parse_*)
#   - final_records.ipynb            (Pflichtfelder + Spalten-Reihenfolge)

# API-Endpunkte (1:1 aus den Notebooks uebernommen, damit die App genau
# dieselben Daten holt, mit denen das Modell trainiert wurde)
API_SEARCH_URL   = "https://api3.geo.admin.ch/rest/services/api/SearchServer"
API_FIND_URL     = "https://api3.geo.admin.ch/rest/services/api/MapServer/find"
API_IDENTIFY_URL = "https://api3.geo.admin.ch/rest/services/api/MapServer/identify"
API_HEIGHT_URL   = "https://api3.geo.admin.ch/rest/services/height"
API_BASE         = "https://api3.geo.admin.ch"

# Offizielle/öffentliche XML-Quellen für EGID-Abfragen.
# 1) Direkter XML-Download der Housing-Stat EGID-Abfrage
# 2) Fallback: MADD/eCH-0206 Webservice
HOUSING_STAT_EGID_XML_URL = "https://www.housing-stat.ch/de/data/query/egid.xml"
MADD_ECH_API_URL          = "https://madd.bfs.admin.ch/eCH-0206"

IDENTIFY_LAYERS = "all:" + ",".join([
    "ch.are.erreichbarkeit-oev",
    "ch.bfe.solarenergie-eignung-daecher",
])
POP_LAYER = "all:ch.bfs.volkszaehlung-bevoelkerungsstatistik_einwohner"

# Spaltenreihenfolge der Modell-CSV (final_records.ipynb FINAL_DATASET_QUERY)
RAW_MODEL_COLUMNS = [
    "area_sqm", "rooms", "population", "oev_score",
    "solar_class", "elevation_m", "lv95_east", "lv95_north",
    "gbauj", "ganzwhg", "garea",
]

# Pflichtspalten (status='usable' aus final_records.ipynb)
REQUIRED_NON_NULL = [
    "address", "area_sqm", "rooms", "population", "egid",
    "gbauj", "ganzwhg", "garea", "oev_score", "solar_class", "elevation_m",
]


def normalize_address(addr: Any) -> str:
    """Whitespace normalisieren — identisch zu den Notebooks."""
    if not isinstance(addr, str):
        return ""
    return re.sub(r"\s+", " ", addr.strip())


def _safe_num(x, default=None) -> Optional[float]:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _safe_int(x, default=None) -> Optional[int]:
    v = _safe_num(x, None)
    if v is None:
        return default
    try:
        return int(round(v))
    except Exception:
        return default


def _request_json(url: str, params: dict, timeout: int = 15) -> Optional[dict]:
    """HTTP-Wrapper mit einem Retry — analog zu _request/_get_json in den Notebooks."""
    for attempt in range(2):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                try:
                    return r.json()
                except Exception:
                    return None
            return None
        except requests.exceptions.ConnectionError:
            if attempt == 0:
                time.sleep(2)
        except requests.exceptions.Timeout:
            if attempt == 0:
                time.sleep(1)
        except Exception:
            return None
    return None


# --------------------------------------------------------------------------
# Public Pipeline-API (Funktionsnamen wie vom User gefordert)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def lookup_egid(address: str) -> Dict[str, Any]:
    """Adresse → EGID + Koordinaten + GWR-Link.

    Aus gwr_egid_db_sync.ipynb (search_address): GeoAdmin SearchServer mit
    origins=address. Parsed featureId zu EGID, sammelt Koordinaten und
    optional einen direkten gwr_link für load_gwr_data.

    Raises
    ------
    ValueError: Adresse leer oder kein Treffer.
    """
    addr = normalize_address(address)
    if not addr:
        raise ValueError("Adresse ist leer.")

    data = _request_json(API_SEARCH_URL, {
        "searchText": addr,
        "type":       "locations",
        "origins":    "address",
        "sr":         2056,
        "limit":      1,
    })
    if not data:
        raise requests.exceptions.RequestException("API-Timeout beim SearchServer.")
    if not data.get("results"):
        raise ValueError(f"Adresse nicht gefunden: '{addr}'")

    best = data["results"][0]
    attrs = best.get("attrs", {}) or {}

    # SearchServer: y=East, x=North (LV95)
    east  = _safe_num(attrs.get("y"))
    north = _safe_num(attrs.get("x"))

    # GWR-Link aus den Result-Links extrahieren
    gwr_link = None
    for link in attrs.get("links", []) or []:
        if link.get("title") == "ch.bfs.gebaeude_wohnungs_register":
            href = link.get("href")
            if href:
                gwr_link = href
                break

    # EGID aus featureId (Format: <EGID>_<EWID>)
    feature_id = attrs.get("featureId") or attrs.get("feature_id")
    egid = None
    if feature_id:
        try:
            egid = int(str(feature_id).split("_")[0])
        except (ValueError, IndexError):
            pass

    if east is None or north is None:
        raise ValueError(f"Keine Koordinaten für '{addr}' verfügbar.")

    label_clean = (attrs.get("label", addr) or addr).replace("<b>", "").replace("</b>", "")

    return {
        "address":     label_clean,
        "egid":        egid,
        "lv95_east":   east,
        "lv95_north":  north,
        "gwr_link":    gwr_link,
        "feature_id":  feature_id,
    }


@st.cache_data(show_spinner=False)
def load_gwr_data(
    egid: Optional[int] = None,
    gwr_link: Optional[str] = None,
) -> Dict[str, Any]:
    """EGID → GWR-Gebäudeattribute (gbauj, ganzwhg, garea, ...).

    Wichtig: `gbauj` ist das Baujahr des Gebäudes. Es wird bewusst nicht aus
    `yearOfConstruction` der einzelnen Wohnung überschrieben.
    """
    if not egid and not gwr_link:
        raise ValueError("Weder EGID noch gwr_link angegeben.")

    feature = None

    # 1) GeoAdmin-Link aus lookup_egid bevorzugen
    if gwr_link:
        url = gwr_link if gwr_link.startswith("http") else f"{API_BASE}{gwr_link}"
        data = _request_json(url, {"returnGeometry": "false"}, timeout=20)
        if data:
            feature = data.get("feature", data)

    # 2) GeoAdmin Find by EGID
    if feature is None and egid is not None:
        data = _request_json(API_FIND_URL, {
            "layer":          "ch.bfs.gebaeude_wohnungs_register",
            "searchText":     str(egid),
            "searchField":    "egid",
            "returnGeometry": "false",
        }, timeout=20)
        if data and data.get("results"):
            feature = data["results"][0]

    attrs: Dict[str, Any] = {}
    if feature is not None:
        attrs_raw = feature.get("attributes", {}) or {}
        attrs = {str(k).lower(): v for k, v in attrs_raw.items()}

    # Werte aus GeoAdmin, falls vorhanden
    result = {
        "egid":     _safe_int(attrs.get("egid", egid)),
        "gbauj":    _safe_int(attrs.get("gbauj")),
        "gbaup":    _safe_int(attrs.get("gbaup")),
        "ganzwhg":  _safe_int(attrs.get("ganzwhg")),
        "garea":    _safe_num(attrs.get("garea")),
        "_attrs":   attrs,
    }

    # 3) Fallback/Ergänzung aus MADD-XML.
    #    Hier kommt das Baujahr aus building/dateOfConstruction,
    #    nicht aus dwelling/yearOfConstruction.
    if egid is not None and (
        result["gbauj"] is None
        or result["ganzwhg"] is None
        or result["garea"] is None
    ):
        try:
            xml_text, madd_debug = _fetch_madd_xml_for_egid(int(egid), timeout=20)
            if xml_text:
                building = _parse_building_from_madd_xml(xml_text)
                if result["egid"] is None:
                    result["egid"] = _safe_int(building.get("egid"), egid)
                if result["gbauj"] is None:
                    result["gbauj"] = _safe_int(building.get("gbauj"))
                if result["ganzwhg"] is None:
                    result["ganzwhg"] = _safe_int(building.get("ganzwhg"))
                if result["garea"] is None:
                    result["garea"] = _safe_num(building.get("garea"))
                result["_madd_debug"] = madd_debug
        except Exception as exc:
            result["_madd_error"] = f"{type(exc).__name__}: {exc}"

    if (
        result["gbauj"] is None
        and result["ganzwhg"] is None
        and result["garea"] is None
        and feature is None
    ):
        raise ValueError(f"GWR-Daten nicht gefunden (egid={egid}).")

    return result

# GWR-WSTWK Stockwerk-Codes (BFS-Standard)
WSTWK_CODE_MAP: Dict[int, str] = {
    3100: "Sockelgeschoss",
    3300: "UG",
    3401: "1. UG",
    3402: "2. UG",
    3403: "3. UG",
    3413: "EG",
    3500: "EG",
    3501: "1. OG",
    3502: "2. OG",
    3503: "3. OG",
    3504: "4. OG",
    3505: "5. OG",
    3506: "6. OG",
    3507: "7. OG",
    3508: "8. OG",
    3601: "1. DG",
    3602: "2. DG",
}


def parse_gwr_floor(raw_floor: Any) -> Optional[str]:
    """Heuristisches Parsing der Stockwerk-Codes aus GWR/MADD.

    MADD/eCH liefert für Wohnungen teilweise Codes wie 3101, 3102, ...
    Diese werden als 1. Stock, 2. Stock usw. angezeigt.
    """
    if raw_floor is None or raw_floor == "":
        return None

    s = str(raw_floor).strip()
    if not s:
        return None

    code = _safe_int(s)
    if code is None:
        return s

    # Wichtig für MADD XML: 3101, 3102, 3103, ...
    if 3101 <= code <= 3199:
        return f"{code - 3100}. Stock"

    if code in WSTWK_CODE_MAP:
        return WSTWK_CODE_MAP[code]
    if code == 0:
        return "EG"
    if 1 <= code <= 20:
        return f"{code}. OG"
    if code < 0:
        return f"{abs(code)}. UG"

    return str(code)

APARTMENT_FIELD_KEYS = (
    "ewid", "wstwk", "wflae", "wazim", "wbez", "wstat",
    "stockwerk", "flaeche", "zimmer", "bezeichnung",
    "floor", "area", "rooms", "label",
    "administrativedwellingno", "noofhabitablerooms", "surfaceareaofdwelling",
)


def _is_scalar(v) -> bool:
    """True falls v ein einzelner Skalar ist (keine Liste, kein Dict)."""
    return v is not None and not isinstance(v, (list, dict, tuple, set))


def _is_dwelling_record(d: dict) -> bool:
    """Heuristik: dict sieht wie eine einzelne Wohnung aus.

    Lenient: mindestens 2 Wohnungs-Felder als Skalar.  Wir verlangen *nicht*
    explizit EWID, weil manche APIs EWID weglassen oder anders benennen.
    Zwei Skalar-Felder verhindern Aggregat-Wrapper (wo Felder als Listen
    vorliegen) als False-Positives.
    """
    if not isinstance(d, dict):
        return False
    klow = {str(k).lower(): v for k, v in d.items()}
    n_scalar = sum(1 for k in APARTMENT_FIELD_KEYS if _is_scalar(klow.get(k)))
    return n_scalar >= 2


def _extract_dwelling(d: dict) -> dict:
    klow = {str(k).lower(): v for k, v in d.items()}

    def pick(*keys):
        for k in keys:
            v = klow.get(k)
            if _is_scalar(v):
                return v
        return None

    floor_raw = pick("wstwk", "stockwerk", "floor")
    return {
        "ewid":        _safe_int(pick("ewid")),
        "label":       pick("wbez", "bezeichnung", "label", "administrativedwellingno"),
        "floor_raw":   floor_raw,
        "floor_label": parse_gwr_floor(floor_raw),
        "rooms":       _safe_num(pick("wazim", "zimmer", "rooms", "noofhabitablerooms")),
        "area":        _safe_num(pick("wflae", "flaeche", "area", "surfaceareaofdwelling")),
    }


def _walk_dwellings(obj, seen: set) -> list:
    """Rekursive Suche nach Wohnungs-Records, dedupliziert über EWID oder Field-Tuple."""
    out: list = []
    if isinstance(obj, dict):
        if _is_dwelling_record(obj):
            d = _extract_dwelling(obj)
            # Dedup-Key: EWID falls vorhanden, sonst Tuple aus charakteristischen Feldern
            ewid = d.get("ewid")
            dedup_key = ("ewid", ewid) if ewid else (
                "tup", d.get("floor_label"), d.get("rooms"),
                d.get("area"), d.get("label"),
            )
            if dedup_key not in seen:
                seen.add(dedup_key)
                out.append(d)
            # Nicht weiter in einen Dwelling-Record reinrekursieren
            return out
        for v in obj.values():
            out.extend(_walk_dwellings(v, seen))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            out.extend(_walk_dwellings(item, seen))
    return out


def _parse_dwellings_from_html(html: str) -> list:
    """Heuristisch HTML-Tabelle aus GeoAdmin htmlPopup parsen.

    GeoAdmin liefert für Gebäude eine HTML-Tabelle der Wohnungen. Wir suchen
    nach <tr>-Zeilen, identifizieren die Header (EWID/Stockwerk/Zimmer/Fläche)
    und mappen die Daten.
    """
    out: list = []
    if not html or len(html) < 50:
        return out

    # Finde alle <tr>-Zeilen
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.IGNORECASE | re.DOTALL)
    if not rows:
        return out

    def _strip(cell: str) -> str:
        cell = re.sub(r"<[^>]+>", "", cell)
        cell = cell.replace("&nbsp;", " ").replace("&amp;", "&")
        return cell.strip()

    # Erste Tabelle mit Header identifizieren
    field_map: Dict[str, int] = {}
    data_rows: list = []
    for row in rows:
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row,
                           re.IGNORECASE | re.DOTALL)
        cells = [_strip(c) for c in cells]
        if not cells:
            continue
        # Header-Zeile?
        if not field_map:
            row_lower = " ".join(c.lower() for c in cells)
            looks_like_header = any(k in row_lower for k in
                                     ("ewid", "stockwerk", "wstwk", "wbez",
                                      "zimmer", "wazim", "fläche", "wflae"))
            if looks_like_header:
                for idx, col in enumerate(cells):
                    cl = col.lower()
                    if "ewid" in cl:
                        field_map["ewid"] = idx
                    elif "stockwerk" in cl or "wstwk" in cl or "etage" in cl:
                        field_map["floor_raw"] = idx
                    elif "zimmer" in cl or "wazim" in cl:
                        field_map["rooms"] = idx
                    elif "fläche" in cl or "flaeche" in cl or "wflae" in cl:
                        field_map["area"] = idx
                    elif "bezeich" in cl or "wbez" in cl or "wohnung" in cl:
                        field_map["label"] = idx
                continue
        if field_map:
            data_rows.append(cells)

    if not field_map or not data_rows:
        return out

    for row in data_rows:
        d = {"ewid": None, "label": None, "floor_raw": None,
             "floor_label": None, "rooms": None, "area": None}
        try:
            if "ewid" in field_map and field_map["ewid"] < len(row):
                d["ewid"] = _safe_int(row[field_map["ewid"]])
            if "label" in field_map and field_map["label"] < len(row):
                v = row[field_map["label"]]
                d["label"] = v if v else None
            if "floor_raw" in field_map and field_map["floor_raw"] < len(row):
                v = row[field_map["floor_raw"]]
                d["floor_raw"]   = v
                d["floor_label"] = parse_gwr_floor(v)
            if "rooms" in field_map and field_map["rooms"] < len(row):
                v = row[field_map["rooms"]].replace(",", ".")
                d["rooms"] = _safe_num(re.sub(r"[^\d.\-]", "", v))
            if "area" in field_map and field_map["area"] < len(row):
                v = row[field_map["area"]].replace(",", ".")
                d["area"] = _safe_num(re.sub(r"[^\d.\-]", "", v))
        except Exception:
            continue
        # Mindestens ein verwertbares Feld
        if any(d.get(k) is not None for k in ("ewid", "rooms", "area", "floor_label", "label")):
            out.append(d)

    return out


# --------------------------------------------------------------------------
# MADD / eCH-0206 XML: EGID -> Gebäude + Wohnungen
# --------------------------------------------------------------------------
def _xml_local_name(tag: str) -> str:
    """Entfernt XML-Namespaces: '{namespace}EGID' -> 'EGID'."""
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _xml_direct_child(parent, name: str):
    """Direktes Kind-Element nach lokalem Namen suchen."""
    if parent is None:
        return None
    name_l = name.lower()
    for child in list(parent):
        if _xml_local_name(child.tag).lower() == name_l:
            return child
    return None


def _xml_first_text(parent, *names: str) -> Optional[str]:
    """Rekursiv ersten Text für einen lokalen XML-Namen finden."""
    if parent is None:
        return None
    wanted = {n.lower() for n in names}
    for el in parent.iter():
        if _xml_local_name(el.tag).lower() in wanted:
            txt = (el.text or "").strip()
            if txt:
                return txt
    return None


def _looks_like_xml(text: str) -> bool:
    s = (text or "").lstrip()
    return s.startswith("<?xml") or s.startswith("<maddResponse") or "<maddResponse" in s[:500]


def _build_madd_request_xml(egid: int) -> str:
    """Minimale eCH-0206 maddRequest-Anfrage für EGID/building."""
    now = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    msg_id = str(uuid.uuid4())
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<eCH-0206:maddRequest
    xmlns:eCH-0206="http://www.ech.ch/xmlns/eCH-0206/2"
    xmlns:eCH-0058="http://www.ech.ch/xmlns/eCH-0058/5"
    xmlns:eCH-0129="http://www.ech.ch/xmlns/eCH-0129/5"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <eCH-0206:requestHeader>
    <eCH-0206:messageId>{msg_id}</eCH-0206:messageId>
    <eCH-0206:businessReferenceId>{msg_id}</eCH-0206:businessReferenceId>
    <eCH-0206:requestingApplication>
      <eCH-0058:manufacturer>HSLU</eCH-0058:manufacturer>
      <eCH-0058:product>RentPredictorStreamlit</eCH-0058:product>
      <eCH-0058:productVersion>1.0</eCH-0058:productVersion>
    </eCH-0206:requestingApplication>
    <eCH-0206:requestDate>{now}</eCH-0206:requestDate>
  </eCH-0206:requestHeader>
  <eCH-0206:requestContext>building</eCH-0206:requestContext>
  <eCH-0206:requestQuery>
    <eCH-0206:EGID>{int(egid)}</eCH-0206:EGID>
  </eCH-0206:requestQuery>
</eCH-0206:maddRequest>'''


def _fetch_madd_xml_for_egid(egid: int, timeout: int = 20) -> tuple[str, Dict[str, Any]]:
    """Automatischer XML-Abruf für eine EGID.

    Reihenfolge:
    1. Direkter XML-Download der Housing-Stat EGID-Abfrage.
    2. Fallback: MADD/eCH-0206 als POST mit XML-Body.
    3. Fallback: MADD/eCH-0206 als GET mit egid-Parameter.
    """
    headers_xml = {
        "Accept": "application/xml,text/xml,*/*",
        "User-Agent": "rent-predictor-streamlit/1.0",
    }
    attempts: list[Dict[str, Any]] = []

    # 1) Direkter XML-Endpunkt passend zur EGID-Webabfrage
    try:
        r = requests.get(
            HOUSING_STAT_EGID_XML_URL,
            params={"egid": int(egid)},
            headers=headers_xml,
            timeout=timeout,
        )
        info = {
            "method": "GET",
            "url": r.url,
            "status": r.status_code,
            "content_type": r.headers.get("Content-Type"),
            "text_start": r.text[:160],
        }
        attempts.append(info)
        if r.status_code == 200 and _looks_like_xml(r.text):
            return r.text, {"success": True, "used": info, "attempts": attempts}
    except Exception as exc:
        attempts.append({
            "method": "GET",
            "url": HOUSING_STAT_EGID_XML_URL,
            "exception": f"{type(exc).__name__}: {exc}",
        })

    # 2) MADD/eCH-0206 POST
    xml_body = _build_madd_request_xml(egid)
    try:
        r = requests.post(
            MADD_ECH_API_URL,
            data=xml_body.encode("utf-8"),
            headers={
                "Content-Type": "application/xml; charset=utf-8",
                "Accept": "application/xml,text/xml,*/*",
                "User-Agent": "rent-predictor-streamlit/1.0",
            },
            timeout=timeout,
        )
        info = {
            "method": "POST",
            "url": r.url,
            "status": r.status_code,
            "content_type": r.headers.get("Content-Type"),
            "text_start": r.text[:160],
        }
        attempts.append(info)
        if r.status_code == 200 and _looks_like_xml(r.text):
            return r.text, {"success": True, "used": info, "attempts": attempts}
    except Exception as exc:
        attempts.append({
            "method": "POST",
            "url": MADD_ECH_API_URL,
            "exception": f"{type(exc).__name__}: {exc}",
        })

    # 3) MADD/eCH-0206 GET
    try:
        r = requests.get(
            MADD_ECH_API_URL,
            params={"egid": int(egid)},
            headers=headers_xml,
            timeout=timeout,
        )
        info = {
            "method": "GET",
            "url": r.url,
            "status": r.status_code,
            "content_type": r.headers.get("Content-Type"),
            "text_start": r.text[:160],
        }
        attempts.append(info)
        if r.status_code == 200 and _looks_like_xml(r.text):
            return r.text, {"success": True, "used": info, "attempts": attempts}
    except Exception as exc:
        attempts.append({
            "method": "GET",
            "url": MADD_ECH_API_URL,
            "exception": f"{type(exc).__name__}: {exc}",
        })

    return "", {"success": False, "attempts": attempts}


def _parse_building_from_madd_xml(xml_text: str) -> Dict[str, Any]:
    """Gebäudeattribute aus MADD XML lesen.

    `gbauj` stammt aus building/dateOfConstruction/dateOfConstruction.
    `ganzwhg` wird aus der Anzahl dwellingItem gelesen.
    `garea` entspricht surfaceAreaOfBuilding.
    """
    root = ET.fromstring(xml_text)

    dwelling_count = sum(
        1 for el in root.iter()
        if _xml_local_name(el.tag) == "dwellingItem"
    )

    for building_item in root.iter():
        if _xml_local_name(building_item.tag) != "buildingItem":
            continue

        building = _xml_direct_child(building_item, "building")
        if building is None:
            continue

        date_node = _xml_direct_child(building, "dateOfConstruction")

        return {
            "egid":     _safe_int(_xml_first_text(building_item, "EGID")),
            "gbauj":    _safe_int(_xml_first_text(date_node, "dateOfConstruction")),
            "gbaup":    _safe_int(_xml_first_text(date_node, "periodOfConstruction")),
            "ganzwhg":  dwelling_count or None,
            "garea":    _safe_num(_xml_first_text(building, "surfaceAreaOfBuilding")),
            "building_status":   _safe_int(_xml_first_text(building, "buildingStatus")),
            "building_category": _safe_int(_xml_first_text(building, "buildingCategory")),
            "building_class":    _safe_int(_xml_first_text(building, "buildingClass")),
            "number_of_floors":  _safe_int(_xml_first_text(building, "numberOfFloors")),
        }

    return {
        "egid":     None,
        "gbauj":    None,
        "gbaup":    None,
        "ganzwhg":  dwelling_count or None,
        "garea":    None,
    }


def _parse_dwellings_from_madd_xml(xml_text: str) -> list:
    """Wohnungen aus MADD/eCH-0206 XML parsen.

    Das Wohnungs-Baujahr wird bewusst als `year_built_dwelling` geführt und
    nicht als Modell-Baujahr verwendet. Für das Modell bleibt `gbauj`
    aus dem Gebäude massgebend.
    """
    out: list = []
    seen: set = set()

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out

    for item in root.iter():
        if _xml_local_name(item.tag) != "dwellingItem":
            continue

        ewid = _safe_int(_xml_first_text(item, "EWID"))
        admin_no = _xml_first_text(item, "administrativeDwellingNo")
        floor_raw = _xml_first_text(item, "floor")

        d = {
            "ewid":                ewid,
            "label":               admin_no,
            "floor_raw":           _safe_int(floor_raw, floor_raw),
            "floor_label":         parse_gwr_floor(floor_raw),
            "rooms":               _safe_num(_xml_first_text(item, "noOfHabitableRooms")),
            "area":                _safe_num(_xml_first_text(item, "surfaceAreaOfDwelling")),
            "year_built_dwelling": _safe_int(_xml_first_text(item, "yearOfConstruction")),
            "kitchen":             _safe_int(_xml_first_text(item, "kitchen")),
            "dwelling_status":     _safe_int(_xml_first_text(item, "dwellingStatus")),
        }

        key = ("ewid", ewid) if ewid else (
            "tup", d.get("floor_label"), d.get("rooms"),
            d.get("area"), d.get("label"),
        )
        if key not in seen:
            seen.add(key)
            out.append(d)

    return out


@st.cache_data(show_spinner=False)
def load_gwr_dwellings_with_debug(egid: int) -> tuple:
    """EGID → (Wohnungsliste, Debug-Info).

    Primär wird automatisch die XML/API-Antwort von Housing-Stat/MADD gelesen.
    Die alten GeoAdmin-Varianten bleiben nur als Fallback drin.
    """
    debug: Dict[str, Any] = {"egid": egid, "attempts": []}
    if not egid:
        return [], debug

    seen: set = set()
    out: list = []

    # === 1. MADD / Housing-Stat XML: echte Wohnungsdetails ===
    try:
        xml_text, madd_debug = _fetch_madd_xml_for_egid(int(egid), timeout=20)
        attempt = {
            "source": "housing-stat / MADD XML",
            "success": bool(xml_text),
            **madd_debug,
        }
        if xml_text:
            parsed = _parse_dwellings_from_madd_xml(xml_text)
            attempt["xml_len"] = len(xml_text)
            attempt["dwellings_parsed"] = len(parsed)

            for d in parsed:
                ewid = d.get("ewid")
                key = ("ewid", ewid) if ewid else (
                    "tup", d.get("floor_label"), d.get("rooms"),
                    d.get("area"), d.get("label"),
                )
                if key not in seen:
                    seen.add(key)
                    out.append(d)
        debug["attempts"].append(attempt)
    except Exception as exc:
        debug["attempts"].append({
            "source": "housing-stat / MADD XML",
            "exception": f"{type(exc).__name__}: {exc}",
        })

    headers = {
        "Accept": "application/json",
        "User-Agent": "rent-predictor-streamlit/1.0",
    }

    # === 2. GeoAdmin Feature-Endpoint (Fallback) ===
    if not out:
        feat_url = (
            f"https://api3.geo.admin.ch/rest/services/ech/MapServer/"
            f"ch.bfs.gebaeude_wohnungs_register/{int(egid)}"
            f"?returnGeometry=false&lang=de"
        )
        attempt = {"source": "geo.admin.ch feature", "url": feat_url}
        try:
            r = requests.get(feat_url, headers=headers, timeout=20)
            attempt["status"] = r.status_code
            if r.status_code == 200:
                try:
                    data = r.json()
                    if isinstance(data, dict):
                        attempt["top_keys"] = list(data.keys())[:10]
                    before = len(out)
                    out.extend(_walk_dwellings(data, seen))
                    attempt["dwellings_found"] = len(out) - before
                except Exception as e:
                    attempt["json_error"] = str(e)
        except Exception as e:
            attempt["exception"] = f"{type(e).__name__}: {e}"
        debug["attempts"].append(attempt)

    # === 3. GeoAdmin htmlPopup (Fallback) ===
    if not out:
        for popup_kind in ("extendedHtmlPopup", "htmlPopup"):
            popup_url = (
                f"https://api3.geo.admin.ch/rest/services/ech/MapServer/"
                f"ch.bfs.gebaeude_wohnungs_register/{int(egid)}/{popup_kind}"
                f"?lang=de"
            )
            attempt = {"source": f"geo.admin.ch {popup_kind}", "url": popup_url}
            try:
                r = requests.get(
                    popup_url,
                    headers={"Accept": "text/html",
                             "User-Agent": headers["User-Agent"]},
                    timeout=20,
                )
                attempt["status"] = r.status_code
                if r.status_code == 200:
                    parsed = _parse_dwellings_from_html(r.text)
                    attempt["html_len"] = len(r.text)
                    attempt["dwellings_parsed"] = len(parsed)
                    for d in parsed:
                        ewid = d.get("ewid")
                        key = ("ewid", ewid) if ewid else (
                            "tup", d.get("floor_label"), d.get("rooms"),
                            d.get("area"), d.get("label"),
                        )
                        if key not in seen:
                            seen.add(key)
                            out.append(d)
            except Exception as e:
                attempt["exception"] = f"{type(e).__name__}: {e}"
            debug["attempts"].append(attempt)
            if out:
                break

    # === 4. GeoAdmin Find (letzter Fallback) ===
    if not out:
        attempt = {"source": "geo.admin.ch find", "egid": egid}
        try:
            data = _request_json(API_FIND_URL, {
                "layer":          "ch.bfs.gebaeude_wohnungs_register",
                "searchText":     str(int(egid)),
                "searchField":    "egid",
                "returnGeometry": "false",
                "limit":          50,
            }, timeout=20)
            attempt["got_data"] = data is not None
            if data is not None:
                if isinstance(data, dict):
                    attempt["top_keys"] = list(data.keys())[:10]
                before = len(out)
                out.extend(_walk_dwellings(data, seen))
                attempt["dwellings_found"] = len(out) - before
        except Exception as e:
            attempt["exception"] = f"{type(e).__name__}: {e}"
        debug["attempts"].append(attempt)

    # Sortierung: EWID zuerst, sonst Stockwerk/Label
    def _sort_key(d):
        ewid = d.get("ewid")
        if ewid is not None:
            return (0, ewid)
        fr = d.get("floor_raw")
        if isinstance(fr, (int, float)):
            return (1, fr)
        if isinstance(fr, str) and fr.strip():
            return (2, fr)
        return (3, "")

    out.sort(key=_sort_key)
    debug["total_dwellings"] = len(out)
    return out, debug
def make_manual_entry_dwelling(area_default: float = 75.0,
                                 rooms_default: float = 3.0) -> dict:
    """Letzter Fallback: ein einziger generischer Eintrag für manuelle Eingabe.

    Wird verwendet, wenn weder die GWR-API noch die Synthese aus ganzwhg
    Wohnungen liefern. Damit hat das Dropdown immer mindestens einen Eintrag
    und der Nutzer kann Fläche / Zimmer / Stockwerk frei eingeben.
    """
    return {
        "ewid":        None,
        "label":       "Manuelle Eingabe (keine GWR-Daten verfügbar)",
        "floor_raw":   None,
        "floor_label": None,
        "rooms":       rooms_default,
        "area":        area_default,
    }


def synthesize_dwellings_from_building(gwr_building: Dict[str, Any]) -> list:
    """Fallback: aus `ganzwhg` und `garea` synthetische Wohnungs-Einträge bauen.

    Wenn keine echte API Wohnungen liefert, generieren wir N generische Einträge,
    bei denen `area` der Gebäude-Durchschnitt ist (`garea / ganzwhg`).
    Stockwerk und Zimmer bleiben offen, der Nutzer trägt sie manuell nach.
    """
    if not gwr_building:
        return []
    ganzwhg = _safe_int(gwr_building.get("ganzwhg"))
    garea   = _safe_num(gwr_building.get("garea"))
    if not ganzwhg or ganzwhg <= 0:
        return []
    avg_area = (garea / ganzwhg) if (garea and ganzwhg) else None
    return [
        {
            "ewid":        None,
            "label":       f"Wohnung {i+1} (Schätzung aus GWR-Total)",
            "floor_raw":   None,
            "floor_label": None,
            "rooms":       None,
            "area":        avg_area,
        }
        for i in range(int(ganzwhg))
    ]


def load_gwr_dwellings(egid: int) -> list:
    """Thin wrapper, nur die Liste — für bestehende Aufrufer."""
    out, _ = load_gwr_dwellings_with_debug(egid)
    return out


@st.cache_data(show_spinner=False)
def load_swisstopo_data(east: float, north: float) -> Dict[str, Any]:
    """LV95-Koordinaten → Höhe + ÖV-Score + Solar-Klasse + Bevölkerung.

    Aus swisstopo_enrich_db_sync_v2.ipynb (enrich_address ohne den
    geocode-Step, der schon in lookup_egid passiert ist).
    """
    # Höhe ü. M.
    h = _request_json(API_HEIGHT_URL, {"easting": east, "northing": north}, timeout=15)
    elevation = _safe_num((h or {}).get("height"))

    # ÖV-Score + Solar-Klasse via identify (tolerance=1, erste Treffer)
    ident = _request_json(API_IDENTIFY_URL, {
        "geometry":       f"{east},{north}",
        "geometryType":   "esriGeometryPoint",
        "layers":         IDENTIFY_LAYERS,
        "tolerance":      1,
        "returnGeometry": "false",
        "sr":             2056,
        "imageDisplay":   "100,100,96",
        "mapExtent":      f"{east-10},{north-10},{east+10},{north+10}",
    }, timeout=20)

    oev_score, solar_class = None, None
    for item in (ident or {}).get("results") or []:
        layer = item.get("layerBodId", "")
        attr  = item.get("attributes", {}) or {}
        if oev_score is None and layer == "ch.are.erreichbarkeit-oev":
            oev_score = _safe_num(attr.get("oev_erreichb_ewap"))
        elif solar_class is None and layer == "ch.bfe.solarenergie-eignung-daecher":
            solar_class = _safe_int(attr.get("klasse"))
        if oev_score is not None and solar_class is not None:
            break

    # Bevölkerung — nächstgelegene Hektarzelle
    pop_resp = _request_json(API_IDENTIFY_URL, {
        "geometry":       f"{east},{north}",
        "geometryType":   "esriGeometryPoint",
        "layers":         POP_LAYER,
        "tolerance":      1,
        "returnGeometry": "false",
        "sr":             2056,
        "imageDisplay":   "100,100,96",
        "mapExtent":      f"{east-10},{north-10},{east+10},{north+10}",
    }, timeout=15)

    population = None
    for item in (pop_resp or {}).get("results") or []:
        attr = item.get("attributes", {}) or {}
        n_val = _safe_int(attr.get("number"))
        y_val = attr.get("i_year")
        if n_val is None:
            continue
        if y_val is None or y_val == 2024:
            population = n_val
            break

    return {
        "elevation_m": elevation,
        "oev_score":   oev_score,
        "solar_class": solar_class,
        "population":  population,
    }


def assemble_features(
    *,
    address: str,
    area_sqm: float,
    rooms: float,
    floor: Optional[str] = None,
    egid_info: Dict[str, Any],
    gwr_info: Dict[str, Any],
    swisstopo_info: Dict[str, Any],
) -> pd.DataFrame:
    """Kombiniert alle Quellen zu einer Roh-Zeile mit der Spaltenstruktur
    aus final_records.ipynb (Modell-CSV).

    `floor` wird mitgeführt (Stockwerk), ist aktuell aber kein Modellfeature.
    """
    raw = {
        "address":      address,
        "area_sqm":     _safe_num(area_sqm),
        "rooms":        _safe_num(rooms),
        "lv95_east":    _safe_num(egid_info.get("lv95_east")),
        "lv95_north":   _safe_num(egid_info.get("lv95_north")),
        "egid":         _safe_int(egid_info.get("egid")),
        "gbauj":        _safe_int(gwr_info.get("gbauj")),
        "ganzwhg":      _safe_int(gwr_info.get("ganzwhg")),
        "garea":        _safe_num(gwr_info.get("garea")),
        "elevation_m":  _safe_num(swisstopo_info.get("elevation_m")),
        "oev_score":    _safe_num(swisstopo_info.get("oev_score")),
        "solar_class":  _safe_int(swisstopo_info.get("solar_class")),
        "population":   _safe_int(swisstopo_info.get("population")),
        "floor":        floor,  # informativ
    }
    return pd.DataFrame([raw])


def clean_and_finalize_records(
    raw_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Bereinigung + Harmonisierung wie in final_records.ipynb.

    1. Pflichtspalten dürfen nicht NULL sein (Status='usable')
    2. Typkonvertierungen wie im Notebook
    3. Spaltenreihenfolge wie in model.csv
    4. Rename auf Trainings-Spaltennamen (area_sqm → area, etc.)
    """
    df = raw_df.copy()
    status: Dict[str, Any] = {"warnings": [], "missing": []}

    # Status-Check (final_records.ipynb 'usable'-Filter)
    for col in REQUIRED_NON_NULL:
        if col in df.columns and df[col].isna().any():
            status["missing"].append(col)

    if status["missing"]:
        status["warnings"].append(
            f"Pflichtspalten fehlend ('usable'-Filter aus final_records.ipynb): "
            f"{status['missing']}"
        )

    # Typen wie im Notebook
    for c in ["area_sqm", "rooms", "population", "egid",
              "gbauj", "ganzwhg", "solar_class"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["lv95_east", "lv95_north", "elevation_m", "oev_score", "garea"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Spaltenreihenfolge wie in model.csv (FINAL_DATASET_QUERY)
    model_cols_raw = [c for c in RAW_MODEL_COLUMNS if c in df.columns]
    df_model = df[model_cols_raw].copy()

    # Rename auf Trainings-Namen (siehe COLUMN_RENAMES weiter unten)
    df_model = df_model.rename(columns=COLUMN_RENAMES)

    return df_model, status


# --- Daten- und Modell-Loader -----------------------------------------------
# Streamlit-Caching: das Modell wird einmal trainiert (geht ein paar Sekunden)
# und danach aus dem Cache wiederverwendet. Andernfalls wuerde es bei jedem
# Klick erneut trainieren, und die App waere unbenutzbar langsam.
COLUMN_RENAMES = {
    "area_sqm":    "area",
    "rooms":       "rooms",
    "price_cold":  "price",
    "population":  "population",
    "oev_score":   "oev",
    "solar_class": "solar",
    "elevation_m": "elevation",
    "lv95_east":   "east",
    "lv95_north":  "north",
    "gbauj":       "year_built",
    "ganzwhg":     "apartments",
    "garea":       "land_area",
}


@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Daten nicht gefunden: {DATA_PATH}. "
            "Stelle sicher, dass src/external-sources/output_csv/model.csv existiert."
        )
    df = pd.read_csv(DATA_PATH).rename(columns=COLUMN_RENAMES)
    return df


@st.cache_resource(show_spinner="Trainiere Modell (einmalig, ca. 10 Sekunden) ...")
def get_predictor() -> tuple[RentPredictor, str]:
    """Lädt gecachten Predictor oder trainiert frisch."""
    df = load_data()

    # Versuche, ein bereits trainiertes Streamlit-Artefakt zu laden
    if MODEL_PATH.exists():
        try:
            artifact = joblib.load(MODEL_PATH)
            return artifact["predictor"], f"Cache: {MODEL_PATH.name}"
        except Exception as exc:
            st.warning(f"Konnte gecachtes Modell nicht laden ({exc}), trainiere neu.")

    # Frisch trainieren
    rp = RentPredictor()
    rp.fit(df)

    # Optional: Cache schreiben
    try:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"predictor": rp}, MODEL_PATH)
    except Exception:
        pass  # nicht kritisch

    return rp, "frisch trainiert"



# --- UI: Streamlit-Dashboard ------------------------------------------------
# Ab hier ist alles "nur" Frontend. Die Logik (Adresse -> Features -> Predict)
# steckt komplett in den Pfad-/Pipeline-Funktionen weiter oben; hier wird nur
# noch eingegeben, gerendert und angezeigt.
APP_VERSION = "Dashboard v4.3"

PRES_NAVY = "#1F2A44"
PRES_BLUE = "#263B73"
PRES_RED = "#FF454F"
PRES_BG = "#F7F8FB"
PRES_MUTED = "#6B7280"

PROJECT_METRICS = {
    "rmse": 393.0,
    "r2": 0.751,
    "baseline_rmse": 847.0,
    "band_mae": {
        "Cheap": 216.0,
        "Medium-low": 176.0,
        "Medium-high": 221.0,
        "Expensive": 445.0,
    },
}

MODEL_COMPARISON = pd.DataFrame([
    {"Model": "Dummy mean", "RMSE Eval": 847, "R²": 0.000},
    {"Model": "Ridge", "RMSE Eval": 560, "R²": None},
    {"Model": "RandomForest", "RMSE Eval": 425, "R²": None},
    {"Model": "GradientBoosting", "RMSE Eval": 425, "R²": None},
    {"Model": "XGBoost", "RMSE Eval": 421, "R²": None},
    {"Model": "LightGBM", "RMSE Eval": 399, "R²": None},
    {"Model": "LightGBM + KNN", "RMSE Eval": 393, "R²": 0.751},
])


def inject_css() -> None:
    st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient(180deg, #ffffff 0%, {PRES_BG} 100%);
    }}
    .block-container {{
        max-width: 1360px;
        padding-top: 2.0rem;
        padding-bottom: 3rem;
    }}
    div[data-testid="stSidebar"] {{
        background: #EEF2F7;
        border-right: 1px solid #DEE4EE;
    }}
    h1, h2, h3 {{
        color: {PRES_NAVY};
        letter-spacing: .01em;
    }}
    .hero {{
        background: linear-gradient(135deg, {PRES_NAVY} 0%, {PRES_BLUE} 70%, #375AA4 100%);
        color: white;
        padding: 28px 32px;
        border-radius: 26px;
        box-shadow: 0 18px 50px rgba(31,42,68,.18);
        margin-bottom: 20px;
    }}
    .hero h1 {{
        color: white;
        margin: 0;
        font-size: 2.35rem;
    }}
    .hero p {{
        margin: 8px 0 0 0;
        color: rgba(255,255,255,.82);
        font-size: 1.05rem;
    }}
    .version-pill {{
        display: inline-block;
        margin-top: 14px;
        padding: 5px 10px;
        border-radius: 999px;
        background: rgba(255,255,255,.16);
        border: 1px solid rgba(255,255,255,.30);
        color: white;
        font-size: .82rem;
    }}
    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 16px;
        margin: 14px 0 18px 0;
    }}
    .kpi-card {{
        background: white;
        border: 1px solid #E6E9F0;
        border-radius: 20px;
        padding: 18px 20px;
        box-shadow: 0 12px 30px rgba(31,42,68,.07);
    }}
    .kpi-label {{
        color: {PRES_MUTED};
        font-size: .84rem;
        margin-bottom: 6px;
    }}
    .kpi-value {{
        color: {PRES_NAVY};
        font-size: 1.75rem;
        font-weight: 760;
        line-height: 1.12;
    }}
    .kpi-note {{
        color: {PRES_MUTED};
        font-size: .78rem;
        margin-top: 6px;
    }}
    .section-card {{
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 22px;
        padding: 22px 24px;
        box-shadow: 0 12px 32px rgba(31,42,68,.06);
        margin: 14px 0;
    }}
    .success-box {{
        background: #EAF8F0;
        border: 1px solid #C9EFD8;
        color: #116636;
        border-radius: 16px;
        padding: 14px 16px;
        margin: 12px 0;
        font-weight: 650;
    }}
    .warning-box {{
        background: #FFF7E6;
        border: 1px solid #FFE0A3;
        color: #7A4E00;
        border-radius: 16px;
        padding: 14px 16px;
        margin: 12px 0;
    }}
    div.stButton > button:first-child {{
        background: {PRES_RED};
        color: white;
        border: 0;
        border-radius: 14px;
        padding: .65rem 1.2rem;
        font-weight: 750;
    }}
    div.stButton > button:first-child:hover {{
        background: #E73741;
        color: white;
        border: 0;
    }}
    </style>
    """, unsafe_allow_html=True)


def fmt_chf(v: Any, decimals: int = 0) -> str:
    try:
        return f"{float(v):,.{decimals}f}".replace(",", "'") + " CHF"
    except Exception:
        return "—"


def clean_label(label: str) -> str:
    return re.sub(r"<[^>]+>", "", label or "").replace("  ", " ").strip()


def kpi_cards(items: list[tuple[str, str, str]]) -> None:
    html = ['<div class="kpi-grid">']
    for label, value, note in items:
        html.append(
            f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


@st.cache_data(show_spinner=False, ttl=3600)
def geoadmin_address_suggestions(query: str, limit: int = 10) -> list[Dict[str, Any]]:
    q = normalize_address(query)
    if len(q) < 3:
        return []
    data = _request_json(API_SEARCH_URL, {
        "searchText": q,
        "type": "locations",
        "origins": "address",
        "sr": 2056,
        "limit": int(limit),
    }, timeout=12)

    out: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in (data or {}).get("results") or []:
        attrs = item.get("attrs", {}) or {}
        label = clean_label(attrs.get("label") or attrs.get("detail") or q)
        if not label or label in seen:
            continue
        seen.add(label)
        feature_id = attrs.get("featureId") or attrs.get("feature_id")
        egid = None
        if feature_id:
            try:
                egid = int(str(feature_id).split("_")[0])
            except Exception:
                egid = None
        out.append({
            "label": label,
            "egid": egid,
            "lv95_east": _safe_num(attrs.get("y")),
            "lv95_north": _safe_num(attrs.get("x")),
            "feature_id": feature_id,
        })
    return out


class GenericJoblibModel:
    """Adapter für Notebook-Artefakte: {'model': ..., 'features'/'feature_cols': ...}."""

    def __init__(self, artifact: Any, name: str = "joblib model"):
        self.artifact = artifact
        self.name = name
        if isinstance(artifact, dict):
            self.model = artifact.get("model") or artifact.get("estimator") or artifact.get("pipeline")
            self.feature_cols = list(
                artifact.get("features")
                or artifact.get("feature_cols")
                or artifact.get("feature_names")
                or []
            )
            self.meta = artifact
        else:
            self.model = artifact
            self.feature_cols = list(getattr(artifact, "feature_cols", []) or getattr(artifact, "feature_names_in_", []) or [])
            self.meta = {}

        if self.model is None and hasattr(artifact, "predict"):
            self.model = artifact
        if not self.feature_cols and hasattr(self.model, "feature_names_in_"):
            self.feature_cols = list(self.model.feature_names_in_)

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "building_age" not in out.columns and "year_built" in out.columns:
            out["building_age"] = 2026 - pd.to_numeric(out["year_built"], errors="coerce")
        if "area_per_room" not in out.columns and {"area", "rooms"}.issubset(out.columns):
            rooms_num = pd.to_numeric(out["rooms"], errors="coerce")
            area_num = pd.to_numeric(out["area"], errors="coerce")
            out["area_per_room"] = np.where(rooms_num > 0, area_num / rooms_num, np.nan)
        if "land_area_per_apartment" not in out.columns and {"land_area", "apartments"}.issubset(out.columns):
            a_num = pd.to_numeric(out["apartments"], errors="coerce")
            l_num = pd.to_numeric(out["land_area"], errors="coerce")
            out["land_area_per_apartment"] = np.where(a_num > 0, l_num / a_num, np.nan)

        # lgbm_wide_4feat speichert KNN-Hilfsstrukturen.
        if isinstance(self.artifact, dict):
            scaler = self.artifact.get("knn_scaler")
            nbrs = self.artifact.get("knn_index")
            prices = self.artifact.get("knn_prices")
            if scaler is not None and nbrs is not None and prices is not None and {"east", "north"}.issubset(out.columns):
                try:
                    coords = scaler.transform(out[["east", "north"]])
                    _, idx = nbrs.kneighbors(coords, n_neighbors=min(10, len(prices)))
                    price_arr = np.asarray(prices)
                    out["knn_price_mean"] = price_arr[idx].mean(axis=1)
                    out["knn_price_median"] = np.median(price_arr[idx], axis=1)
                except Exception:
                    pass

        return out

    def predict(self, df: pd.DataFrame):
        if self.model is None or not hasattr(self.model, "predict"):
            raise RuntimeError(f"Artefakt '{self.name}' enthält kein predict-fähiges Modell.")
        X_full = self._prepare_features(df)
        features = list(self.feature_cols)
        if not features and hasattr(self.model, "feature_names_in_"):
            features = list(self.model.feature_names_in_)
        if not features:
            features = [c for c in X_full.columns if c != "price"]

        for col in features:
            if col not in X_full.columns:
                X_full[col] = np.nan
        X = X_full[features]

        if hasattr(self.model, "feature_names_in_"):
            ordered = list(self.model.feature_names_in_)
            for col in ordered:
                if col not in X.columns:
                    X[col] = np.nan
            X = X[ordered]

        return self.model.predict(X)


def model_search_dirs() -> list[Path]:
    return list(dict.fromkeys([
        PROJECT_ROOT / "models",
        APP_DIR / "models",
        APP_DIR / "notebooks" / "models",
        PROJECT_ROOT / "src" / "notebooks" / "models",
        Path("/home/sa_linux/code/Elias-Martinelli/dspro1/src/notebooks/models"),
    ]))


def discover_model_options() -> list[Dict[str, Any]]:
    """Findet verfügbare Notebook-Modelle.

    Wichtig:
    - `best_model_v3.joblib` ist der neue Default.
    - `rent_predictor_streamlit.joblib` wird bewusst ausgeblendet, weil dieses
      lokale Cache-Artefakt veraltet/falsch sein kann.
    - Der alte Auto-Train-Fallback erscheint nur, falls gar kein Notebook-Modell
      gefunden wird.
    """
    options: list[Dict[str, Any]] = []
    seen: set[str] = set()

    hidden_filenames = {
        "rent_predictor_streamlit.joblib",
    }

    priority = {
        "best_model_v3.joblib": 0,
        "lgbm_wide_4feat.joblib": 1,
        "rent_predictor_v3.joblib": 2,
        "lgbm_minimal_4feat.joblib": 3,
    }

    for d in model_search_dirs():
        try:
            files = sorted(d.glob("*.joblib")) if d.exists() else []
        except Exception:
            files = []

        for path in files:
            if path.name in hidden_filenames:
                continue

            rp = str(path.resolve())
            if rp in seen:
                continue
            seen.add(rp)

            name = path.name
            label = name
            if name == "best_model_v3.joblib":
                label = "best_model_v3.joblib · Default / Best Eval"
            elif name == "lgbm_wide_4feat.joblib":
                label = "lgbm_wide_4feat.joblib · Production Pick"
            elif name == "lgbm_minimal_4feat.joblib":
                label = "lgbm_minimal_4feat.joblib · Minimal"
            elif name == "rent_predictor_v3.joblib":
                label = "rent_predictor_v3.joblib · Notebook Full Predictor"

            options.append({
                "key": rp,
                "label": label,
                "path": path,
                "is_default": name == "best_model_v3.joblib",
                "sort_rank": priority.get(name, 50),
            })

    # Bestes Modell zuerst; danach definierte Reihenfolge und Name.
    options.sort(key=lambda x: (x.get("sort_rank", 50), Path(str(x.get("path", ""))).name))

    # Nur wenn keine Notebook-Modelle gefunden wurden: alter Fallback.
    if not options:
        options.append({
            "key": "__default__",
            "label": "Fallback · aktueller RentPredictor aus model.csv",
            "path": None,
            "is_default": True,
            "sort_rank": 999,
        })

    return options


@st.cache_resource(show_spinner=False)
def load_joblib_artifact(path_str: str) -> Any:
    return joblib.load(path_str)


def load_selected_predictor(model_key: str) -> tuple[Any, Dict[str, Any]]:
    if model_key == "__default__":
        predictor, source = get_predictor()
        return predictor, {
            "label": "Default · aktueller RentPredictor",
            "source": source,
            "path": str(MODEL_PATH),
            "kind": type(getattr(predictor, "model", predictor)).__name__,
            "rmse_eval": PROJECT_METRICS["rmse"],
            "r2_eval": PROJECT_METRICS["r2"],
        }

    artifact = load_joblib_artifact(model_key)
    if isinstance(artifact, dict) and "predictor" in artifact:
        pred = artifact["predictor"]
        metrics = artifact.get("test_metrics") or {}
        return pred, {
            "label": Path(model_key).name,
            "source": "joblib predictor",
            "path": model_key,
            "kind": type(getattr(pred, "model", pred)).__name__,
            "rmse_eval": artifact.get("rmse_eval") or metrics.get("rmse") or PROJECT_METRICS["rmse"],
            "r2_eval": artifact.get("r2_eval") or metrics.get("r2") or PROJECT_METRICS["r2"],
            "feature_cols": artifact.get("feature_cols"),
        }

    adapter = GenericJoblibModel(artifact, name=Path(model_key).name)
    meta = {
        "label": Path(model_key).name,
        "source": "joblib model",
        "path": model_key,
        "kind": type(getattr(adapter, "model", adapter)).__name__,
        "features": getattr(adapter, "feature_cols", []),
        "rmse_eval": PROJECT_METRICS["rmse"],
        "r2_eval": PROJECT_METRICS["r2"],
    }
    if isinstance(artifact, dict):
        meta.update({
            "rmse_eval": artifact.get("rmse_eval") or artifact.get("rmse") or meta["rmse_eval"],
            "r2_eval": artifact.get("r2_eval") or artifact.get("r2") or meta["r2_eval"],
            "overfit_gap": artifact.get("overfit_gap") or artifact.get("gap"),
            "n_train": artifact.get("n_train"),
            "n_eval": artifact.get("n_eval"),
        })
    return adapter, meta


def safe_predict(model_obj: Any, df: pd.DataFrame) -> np.ndarray:
    return np.asarray(model_obj.predict(df.copy()), dtype=float)


def prediction_band(prediction: float, data: Optional[pd.DataFrame] = None) -> tuple[str, float]:
    maes = PROJECT_METRICS["band_mae"]
    try:
        if data is not None and "price" in data.columns and data["price"].notna().sum() > 10:
            q25, q50, q75 = data["price"].quantile([0.25, 0.50, 0.75]).tolist()
            if prediction <= q25:
                return "Cheap", maes["Cheap"]
            if prediction <= q50:
                return "Medium-low", maes["Medium-low"]
            if prediction <= q75:
                return "Medium-high", maes["Medium-high"]
            return "Expensive", maes["Expensive"]
    except Exception:
        pass
    if prediction < 1500:
        return "Cheap", maes["Cheap"]
    if prediction < 2300:
        return "Medium-low", maes["Medium-low"]
    if prediction < 3500:
        return "Medium-high", maes["Medium-high"]
    return "Expensive", maes["Expensive"]


def hero() -> None:
    st.markdown(f"""
    <div class="hero">
        <h1>Mietpreis-Schätzer Schweiz</h1>
        <p>Adresse → EGID/GWR → Swisstopo → Modellvergleich → transparente Kaltmiet-Schätzung</p>
        <span class="version-pill">Version: {APP_VERSION}</span>
    </div>
    """, unsafe_allow_html=True)


def selected_dwelling() -> Optional[dict]:
    dwellings = st.session_state.get("lookup_dwellings") or []
    idx = int(st.session_state.get("dwelling_selector_idx", 0) or 0)
    if 0 <= idx < len(dwellings):
        return dwellings[idx]
    return None


def apply_dwelling_to_inputs(sel: dict) -> None:
    if not sel:
        return
    if sel.get("area") is not None:
        try:
            st.session_state.lookup_area = max(10, min(500, int(round(float(sel["area"])))))
        except Exception:
            pass
    if sel.get("rooms") is not None:
        try:
            st.session_state.lookup_rooms = max(0.5, min(15.0, float(sel["rooms"])))
        except Exception:
            pass
    if sel.get("floor_label"):
        st.session_state.lookup_floor = sel.get("floor_label")


def on_dwelling_change() -> None:
    sel = selected_dwelling()
    if sel:
        apply_dwelling_to_inputs(sel)


def fmt_dwelling(idx: int) -> str:
    d = (st.session_state.get("lookup_dwellings") or [])[idx]
    ew = d.get("ewid") or "?"
    label = d.get("label") or ""
    floor = d.get("floor_label") or "—"
    r = d.get("rooms")
    a = d.get("area")
    r_str = f"{r:.1f}".rstrip("0").rstrip(".") if r else "?"
    a_str = f"{a:.0f}" if a else "?"
    label_part = f" · {label}" if label else ""
    return f"EWID {ew}{label_part} · {floor} · {r_str} Zi · {a_str} m²"


def address_picker() -> Optional[str]:
    st.markdown("### 🔎 Offizielle Adresse suchen")
    st.caption("Tippe Strasse/Hausnummer. Ort oder PLZ ist optional; der Vorschlag kommt von GeoAdmin.")
    selected_label: Optional[str] = None

    try:
        from streamlit_searchbox import st_searchbox  # type: ignore

        def search(term: str) -> list[str]:
            return [x["label"] for x in geoadmin_address_suggestions(term, limit=10)]

        selected_label = st_searchbox(
            search,
            key="live_address_search",
            placeholder="z. B. Kirchhaldenstrasse 36b oder Kronenbergstrasse 5",
            label="Adresse suchen",
            clear_on_submit=False,
        )
        if selected_label:
            st.session_state.selected_address_label = selected_label

    except Exception:
        query = st.text_input(
            "Adresse suchen",
            value=st.session_state.get("address_query", ""),
            placeholder="z. B. Kirchhaldenstrasse 36b oder Kronenbergstrasse 5",
            key="address_query",
        )
        suggestions = geoadmin_address_suggestions(query, limit=10)
        if suggestions:
            labels = [s["label"] for s in suggestions]
            selected_label = st.selectbox("Offizielle Adresse auswählen", labels, key="address_suggestion_select")
            st.session_state.selected_address_label = selected_label
        elif len(normalize_address(query)) >= 3:
            st.info("Noch kein offizieller Vorschlag gefunden. Ergänze Ort oder Hausnummer, falls nötig.")

    return selected_label or st.session_state.get("selected_address_label")


def run_address_lookup(selected_label: str) -> None:
    try:
        load_gwr_dwellings_with_debug.clear()
    except Exception:
        pass

    st.session_state.lookup_area = 75
    st.session_state.lookup_rooms = 3.0
    st.session_state.lookup_floor = "—"
    st.session_state.dwelling_selector_idx = 0
    st.session_state.lookup_dwellings_debug = None
    st.session_state.lookup_dwellings_synthesized = False

    with st.spinner("Adresse via GeoAdmin auflösen ..."):
        egid_info = lookup_egid(selected_label)

    dwellings: list = []
    debug_info: Dict[str, Any] = {}
    if egid_info.get("egid"):
        with st.spinner(f"Wohnungen für EGID {egid_info['egid']} laden ..."):
            try:
                dwellings, debug_info = load_gwr_dwellings_with_debug(int(egid_info["egid"]))
            except Exception as exc:
                debug_info = {"error": f"{type(exc).__name__}: {exc}"}

    # Kein künstliches Synthetisieren. Wenn nichts kommt: genau ein Default-Eintrag.
    if not dwellings:
        dwellings = [make_manual_entry_dwelling(area_default=75.0, rooms_default=3.0)]
        st.session_state.lookup_dwellings_synthesized = True
        debug_info["fallback_manual_entry"] = True

    st.session_state.lookup_egid_info = egid_info
    st.session_state.lookup_dwellings = dwellings
    st.session_state.lookup_dwellings_debug = debug_info
    st.session_state.lookup_address = selected_label
    apply_dwelling_to_inputs(dwellings[0])


def render_prediction_page(model_obj: Any, model_meta: Dict[str, Any]) -> None:
    st.markdown("## Schätzung")
    st.markdown("Adresse auswählen, Wohnung wählen, Werte prüfen und Kaltmiete schätzen.")

    selected_label = address_picker()
    c1, c2 = st.columns([4, 1])
    with c1:
        if selected_label:
            st.markdown(
                f"<div class='success-box'>Ausgewählt: <b>{selected_label}</b><br>"
                "Gebäude- und Wohnungsdaten werden automatisch geladen.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<div class='warning-box'>Wähle zuerst eine offizielle GeoAdmin-Adresse aus dem Dropdown.</div>", unsafe_allow_html=True)
    with c2:
        reload_clicked = st.button("🔄 Daten neu laden", type="primary", disabled=not bool(selected_label), width="stretch")

    # Dynamischer Lookup: Sobald eine offizielle Adresse ausgewählt ist,
    # werden EGID/GWR/Wohnungen automatisch geladen. Der Button ist nur noch
    # zum manuellen Neuladen da.
    if selected_label and (
        reload_clicked
        or st.session_state.get("auto_lookup_address") != selected_label
        or st.session_state.get("lookup_egid_info") is None
    ):
        try:
            run_address_lookup(selected_label)
            st.session_state.auto_lookup_address = selected_label
        except Exception as e:
            st.error(f"Adress-/Gebäude-Lookup fehlgeschlagen: {type(e).__name__}: {e}")

    egid_info_state = st.session_state.get("lookup_egid_info")
    dwellings_state = st.session_state.get("lookup_dwellings") or []
    if egid_info_state is None:
        return

    if st.session_state.get("lookup_dwellings_synthesized", False):
        st.info("Für diese EGID wurden keine Wohnungsdetails gefunden. Die App nutzt Default 75 m² / 3 Zimmer, überschreibbar.")
    else:
        st.success(f"Gefunden: {egid_info_state.get('address')} · EGID {egid_info_state.get('egid')} · {len(dwellings_state)} Wohnung(en)")

    if dwellings_state:
        if not (0 <= st.session_state.get("dwelling_selector_idx", 0) < len(dwellings_state)):
            st.session_state.dwelling_selector_idx = 0
        st.selectbox(
            "🏠 Wohnung auswählen",
            range(len(dwellings_state)),
            format_func=fmt_dwelling,
            key="dwelling_selector_idx",
            on_change=on_dwelling_change,
        )

    floor_options = list(dict.fromkeys(
        ["—", "EG", "1. Stock", "2. Stock", "3. Stock", "4. Stock", "5. Stock",
         "1. OG", "2. OG", "3. OG", "4. OG", "5. OG", "UG", "1. UG", "2. UG", "1. DG", "2. DG"]
        + [d.get("floor_label") for d in dwellings_state if d.get("floor_label")]
    ))
    if st.session_state.lookup_floor not in floor_options:
        floor_options.append(st.session_state.lookup_floor)

    c1, c2, c3 = st.columns(3)
    c1.number_input("Wohnfläche (m²)", min_value=10, max_value=500, step=1, key="lookup_area")
    c2.number_input("Zimmer", min_value=0.5, max_value=15.0, step=0.5, key="lookup_rooms")
    c3.selectbox("Stockwerk", floor_options, key="lookup_floor")

    # Automatische Live-Schätzung: Jede Änderung an Wohnung, Fläche, Zimmer,
    # Stockwerk oder Modell triggert einen Streamlit-Rerun und damit eine neue
    # Schätzung. Kein separater "Analysieren"-Klick nötig.
    st.markdown("### ⚡ Live-Schätzung")
    try:
        selected_dw = selected_dwelling() or {}

        with st.spinner("Schätzung aktualisieren ..."):
            try:
                gwr_info = load_gwr_data(
                    egid=egid_info_state.get("egid"),
                    gwr_link=egid_info_state.get("gwr_link"),
                )
            except ValueError:
                gwr_info = {"egid": egid_info_state.get("egid"), "gbauj": None, "ganzwhg": None, "garea": None}

            # Gebäude-Baujahr bleibt primär; nur wenn leer, Wohnungs-Baujahr verwenden.
            if not gwr_info.get("gbauj") and selected_dw.get("year_built_dwelling"):
                gwr_info["gbauj"] = selected_dw.get("year_built_dwelling")

            swisstopo_info = load_swisstopo_data(
                east=egid_info_state["lv95_east"],
                north=egid_info_state["lv95_north"],
            )

            raw_df = assemble_features(
                address=egid_info_state["address"],
                area_sqm=st.session_state.lookup_area,
                rooms=st.session_state.lookup_rooms,
                floor=None if st.session_state.lookup_floor == "—" else st.session_state.lookup_floor,
                egid_info=egid_info_state,
                gwr_info=gwr_info or {},
                swisstopo_info=swisstopo_info or {},
            )
            model_df, lookup_status = clean_and_finalize_records(raw_df)
            pred_lookup = float(safe_predict(model_obj, model_df)[0])
            band, band_mae = prediction_band(pred_lookup, load_data())
            rmse = float(model_meta.get("rmse_eval") or PROJECT_METRICS["rmse"])

        st.session_state.last_prediction = {
            "prediction": pred_lookup,
            "address": egid_info_state.get("address"),
            "egid": egid_info_state.get("egid"),
            "area": float(st.session_state.lookup_area),
            "rooms": float(st.session_state.lookup_rooms),
            "floor": st.session_state.lookup_floor,
            "price_per_sqm": pred_lookup / float(st.session_state.lookup_area),
            "model": model_meta.get("label"),
            "rmse": rmse,
            "band": band,
            "band_mae": band_mae,
            "model_df": model_df,
        }

        kpi_cards([
            ("Geschätzte Kaltmiete", fmt_chf(pred_lookup), f"Modell: {model_meta.get('label', '—')}"),
            ("Preis pro m²", f"{pred_lookup / float(st.session_state.lookup_area):.0f} CHF/m²", f"{st.session_state.lookup_area:.0f} m², {st.session_state.lookup_rooms:g} Zimmer"),
            ("Typische Unsicherheit", f"± {fmt_chf(rmse)}", "RMSE aus Projekt-Evaluation / Artefakt"),
            ("Preisband", f"{band}", f"typ. MAE ± {fmt_chf(band_mae)}"),
        ])

        st.caption("Die Schätzung aktualisiert sich automatisch, sobald du Wohnung, Fläche, Zimmer, Adresse oder Modell änderst.")

        for w in lookup_status.get("warnings", []):
            st.warning(w)

        with st.expander("Details: Modell-Eingabe und API-Daten"):
            st.markdown("**Bereinigte Modell-Eingabe**")
            st.dataframe(model_df.T.rename(columns={0: "Wert"}), width="stretch")
            st.markdown("**API-Rohdaten**")
            api_rows = [
                {"Quelle": "lookup_egid", **{k: v for k, v in egid_info_state.items() if k != "feature_id"}},
                {"Quelle": "load_gwr_data", **{k: v for k, v in (gwr_info or {}).items() if k not in ("_attrs", "_madd_debug")}},
                {"Quelle": "load_swisstopo_data", **(swisstopo_info or {})},
            ]
            st.dataframe(pd.DataFrame(api_rows).T, width="stretch")

    except Exception as e:
        st.error(f"Live-Schätzung fehlgeschlagen: {type(e).__name__}: {e}")

    if dwellings_state:
        with st.expander(f"Alle {len(dwellings_state)} Wohnung(en) im Gebäude"):
            dw_df = pd.DataFrame(dwellings_state)
            cols = [c for c in ["ewid", "label", "floor_label", "rooms", "area", "year_built_dwelling"] if c in dw_df.columns]
            st.dataframe(dw_df[cols].rename(columns={
                "ewid": "EWID", "label": "Bezeichnung", "floor_label": "Stockwerk",
                "rooms": "Zimmer", "area": "Fläche (m²)", "year_built_dwelling": "Wohnungs-Baujahr",
            }), width="stretch")


def sample_predictions(model_obj: Any, n: int = 700) -> tuple[pd.DataFrame, Optional[str]]:
    try:
        df = load_data().copy()
        if "price" not in df.columns:
            return pd.DataFrame(), "Spalte 'price' fehlt in model.csv."
        df = df.dropna(subset=["price", "area", "rooms", "east", "north"]).reset_index(drop=True)
        if len(df) > n:
            df = df.sample(n=n, random_state=42).reset_index(drop=True)
        X = df.drop(columns=["price"], errors="ignore")
        y_pred = safe_predict(model_obj, X)
        out = df[["price", "area", "rooms", "east", "north"]].copy()
        out["predicted"] = y_pred
        out["residual"] = out["price"] - out["predicted"]
        return out, None
    except Exception as exc:
        return pd.DataFrame(), f"Performance-Sample konnte nicht berechnet werden: {type(exc).__name__}: {exc}"


def plot_price_distribution(df: pd.DataFrame, last: Optional[dict]) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(df["price"].dropna(), bins=35, alpha=0.78, color=PRES_BLUE)
    ax.set_title("Preisverteilung im Modell-Datensatz")
    ax.set_xlabel("Kaltmiete CHF")
    ax.set_ylabel("Anzahl")
    if last:
        ax.axvline(last["prediction"], color=PRES_RED, linewidth=3, label="Letzte Schätzung")
        ax.scatter([last["prediction"]], [0], color=PRES_RED, s=120, zorder=5)
        ax.legend()
    st.pyplot(fig, clear_figure=True)


def plot_actual_vs_predicted(df_pred: pd.DataFrame, last: Optional[dict]) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(df_pred["price"], df_pred["predicted"], s=18, alpha=0.38, color=PRES_BLUE)
    lo = float(min(df_pred["price"].min(), df_pred["predicted"].min()))
    hi = float(max(df_pred["price"].max(), df_pred["predicted"].max()))
    ax.plot([lo, hi], [lo, hi], color=PRES_RED, linestyle="--", linewidth=1.7, label="perfekt")
    if last:
        p = float(last["prediction"])
        ax.scatter([p], [p], color=PRES_RED, s=160, edgecolor="white", linewidth=1.5, zorder=5, label="Neue Schätzung")
    ax.set_title("Actual vs. Predicted")
    ax.set_xlabel("Actual CHF")
    ax.set_ylabel("Predicted CHF")
    ax.legend()
    st.pyplot(fig, clear_figure=True)


def render_performance_page(model_obj: Any, model_meta: Dict[str, Any]) -> None:
    st.markdown("## Model Performance")
    last = st.session_state.get("last_prediction")
    rmse = float(model_meta.get("rmse_eval") or PROJECT_METRICS["rmse"])
    r2 = model_meta.get("r2_eval", PROJECT_METRICS["r2"])

    if last:
        kpi_cards([
            ("Letzte Schätzung", fmt_chf(last["prediction"]), last.get("address", "")),
            ("Erwartete Genauigkeit", f"± {fmt_chf(last.get('band_mae', rmse))}", f"Preisband-MAE: {last.get('band', '—')}"),
            ("Globaler RMSE", f"± {fmt_chf(rmse)}", "quadratischer Durchschnittsfehler"),
            ("R²", f"{float(r2):.3f}" if r2 is not None else "—", "Varianz-Erklärung"),
        ])
    else:
        kpi_cards([
            ("Bestes Projektmodell", "LightGBM + KNN", "aus finaler Evaluation"),
            ("RMSE", fmt_chf(rmse), "niedriger ist besser"),
            ("R²", f"{float(r2):.3f}" if r2 is not None else "—", "höher ist besser"),
            ("Baseline RMSE", fmt_chf(PROJECT_METRICS["baseline_rmse"]), "Mean-Prediction"),
        ])

    df_pred, err = sample_predictions(model_obj)
    c1, c2 = st.columns([1, 1])
    with c1:
        try:
            plot_price_distribution(load_data(), last)
        except Exception as exc:
            st.info(f"Preisverteilung nicht verfügbar: {exc}")
    with c2:
        if err:
            st.info(err)
        elif not df_pred.empty:
            plot_actual_vs_predicted(df_pred, last)

    st.markdown("### Modellvergleich aus Projekt-Evaluation")
    st.dataframe(MODEL_COMPARISON, width="stretch", hide_index=True)

    if not df_pred.empty:
        try:
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
            y = df_pred["price"].values
            p = df_pred["predicted"].values
            sample_metrics = pd.DataFrame([{
                "MAE Sample": mean_absolute_error(y, p),
                "RMSE Sample": float(np.sqrt(mean_squared_error(y, p))),
                "MedAE Sample": median_absolute_error(y, p),
                "R² Sample": r2_score(y, p),
            }])
            st.markdown("### Live-Check auf einem Sample aus `model.csv`")
            st.caption("Diagnosewert auf verfügbaren Daten, nicht der offizielle Hold-out-Wert.")
            st.dataframe(sample_metrics.round(3), width="stretch", hide_index=True)
        except Exception:
            pass

    band_df = pd.DataFrame([{"Preisband": k, "MAE CHF": v} for k, v in PROJECT_METRICS["band_mae"].items()])
    st.markdown("### Fehler nach Preisband")
    st.dataframe(band_df, width="stretch", hide_index=True)


def render_model_data_page(model_meta: Dict[str, Any], model_options: list[Dict[str, Any]]) -> None:
    st.markdown("## Modell & Daten")
    kpi_cards([
        ("Aktives Modell", str(model_meta.get("label", "—")), str(model_meta.get("kind", ""))),
        ("Quelle", str(model_meta.get("source", "—")), str(model_meta.get("path", ""))[:80]),
        ("Datenpfad", "model.csv", str(DATA_PATH)),
        ("Version", APP_VERSION, "sichtbarer Script-Check"),
    ])

    st.markdown("### Gefundene Modell-Artefakte")
    rows = [{"Label": opt["label"], "Pfad": str(opt.get("path") or "Default Auto-Cache"), "Default": bool(opt.get("is_default"))} for opt in model_options]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.markdown("### Modell-Metadaten")
    st.json({k: v for k, v in model_meta.items() if k not in ("features",)})
    if model_meta.get("features"):
        st.write(model_meta.get("features"))

    with st.expander("Datenvorschau `model.csv`"):
        try:
            df = load_data()
            st.dataframe(df.head(25), width="stretch")
            st.write(f"Zeilen: {len(df):,} · Spalten: {len(df.columns):,}".replace(",", "'"))
        except Exception as exc:
            st.error(f"Daten konnten nicht geladen werden: {exc}")


def render_manual_sidebar_prediction(model_obj: Any) -> None:
    st.sidebar.header("🏢 Manuelle Schätzung")
    area = st.sidebar.slider("Wohnfläche (m²)", 20, 250, 75)
    rooms = st.sidebar.slider("Zimmer", 1, 8, 3)
    year_built = st.sidebar.slider("Baujahr", 1900, 2026, 1990)

    st.sidebar.header("🏗️ Gebäude")
    apartments = st.sidebar.slider("Wohnungen im Gebäude", 1, 100, 8)
    land_area = st.sidebar.slider("Grundstücksfläche (m²)", 50, 2000, 300)

    st.sidebar.header("📍 Lage")
    presets = {
        "Zürich (HB)": (2683000, 1247000, 408),
        "Bern (HB)": (2600000, 1200000, 540),
        "Luzern (HB)": (2666000, 1211000, 435),
        "Genf (HB)": (2500000, 1118000, 375),
        "Basel (SBB)": (2611000, 1267000, 270),
        "Lugano (Centro)": (2717500, 1095500, 273),
        "Zermatt": (2624500, 1097000, 1620),
        "Custom": None,
    }
    preset = st.sidebar.selectbox("Stadt-Preset", list(presets.keys()))
    east_d, north_d, elev_d = presets[preset] or (2683000, 1247000, 408)
    east = st.sidebar.number_input("LV95 East", value=east_d, step=1000)
    north = st.sidebar.number_input("LV95 North", value=north_d, step=1000)
    elevation = st.sidebar.number_input("Höhe (m ü. M.)", value=elev_d, step=10)

    st.sidebar.header("🌍 Lagedaten")
    population = st.sidebar.slider("Bevölkerung Hektar", 1, 600, 100)
    oev = st.sidebar.slider("ÖV-Erschliessung (Score)", 0, 100000, 4000)
    solar = st.sidebar.slider("Solar-Klasse (1=schlecht, 5=top)", 1, 5, 3)

    input_df = pd.DataFrame([{
        "east": east, "north": north, "elevation": elevation, "area": area,
        "rooms": rooms, "year_built": year_built, "apartments": apartments,
        "land_area": land_area, "population": population, "oev": oev, "solar": solar,
    }])

    try:
        pred = float(safe_predict(model_obj, input_df)[0])
        st.sidebar.metric("Manuelle Kaltmiete", fmt_chf(pred), f"{pred / area:.0f} CHF/m²")
    except Exception as exc:
        st.sidebar.warning(f"Manuelle Schätzung nicht verfügbar: {type(exc).__name__}")


# ---- App Start ----
st.set_page_config(
    page_title="Mietpreis-Schätzer Schweiz",
    page_icon="🏠",
    layout="wide",
)

inject_css()
hero()

# Session defaults
st.session_state.setdefault("lookup_egid_info", None)
st.session_state.setdefault("lookup_dwellings", [])
st.session_state.setdefault("lookup_address", "")
st.session_state.setdefault("lookup_area", 75)
st.session_state.setdefault("lookup_rooms", 3.0)
st.session_state.setdefault("lookup_floor", "—")
st.session_state.setdefault("dwelling_selector_idx", 0)
st.session_state.setdefault("auto_lookup_address", None)

model_options = discover_model_options()
model_option_map = {m["key"]: m for m in model_options}

selected_model_key = st.sidebar.selectbox(
    "🤖 Modell auswählen",
    [m["key"] for m in model_options],
    index=0,
    format_func=lambda k: model_option_map[k]["label"],
)

try:
    predictor, model_meta = load_selected_predictor(selected_model_key)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Modell konnte nicht geladen werden: {type(e).__name__}: {e}")
    st.stop()

st.sidebar.success(f"Modell aktiv: {model_meta.get('label')} · {model_meta.get('kind', 'Model')}")
st.sidebar.caption(f"{APP_VERSION} · Script OK")
st.sidebar.divider()

render_manual_sidebar_prediction(predictor)

page = st.radio(
    "Navigation",
    ["Schätzung", "Model Performance", "Modell & Daten"],
    horizontal=True,
    label_visibility="collapsed",
    key="top_navigation",
)

if page == "Schätzung":
    render_prediction_page(predictor, model_meta)
elif page == "Model Performance":
    render_performance_page(predictor, model_meta)
else:
    render_model_data_page(model_meta, model_options)
