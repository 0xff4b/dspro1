"""
fix_summary_blocks.py
=====================

One-shot Patch-Skript für `model_v3_clean.ipynb`.

Im letzten Cleanup-Lauf von `humanize_notebook.py` wurden mehrere
"Definitions"-Zellen gedroppt, deren Variablen aber von späteren Zellen
weiterhin gelesen werden. Beim Lauf aus dem Stand crasht das Notebook
dadurch mit verschiedenen `NameError`s.

Bekannte Regressionen, die dieses Skript korrigiert:

1. **Zelle 23.17 (`v3-c23-summary`)** — Konsolidierter Vergleich.
   Kapitel 22.15 (das `summary_blocks` aufgebaut hat) und Kapitel 23.10
   (das `HAS_HALVING` / `res_halving` definiert hat) sind beide entfernt.
   Patch: die Vergleichsliste wird in 23.17 selbst aufgebaut, defensiv
   gegen alle Varianten, die ggf. fehlen.

2. **Zelle 27.2 (`v3-c27-2`)** — Regularisiertes LGBM auf drei
   Feature-Sets. Kapitel 27.1 (das `FEATURES_MINIMAL`, `FEATURES_TOP6`,
   `FEATURES_FULL_KNN` sowie `_train_df_knn` / `_eval_df_knn` definiert
   hat) ist entfernt. Patch: die Feature-Sets und Daten-Aliase werden am
   Anfang der 27.2-Zelle definiert.

Aufruf (vom Repo-Root):

    cd src/notebooks
    python fix_summary_blocks.py

Idempotent: für jede Patch-Regel wird über einen Marker-String geprüft,
ob die Zelle schon repariert ist. Ein zweiter Lauf macht dann nichts.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

NB_PATH = Path(__file__).resolve().parent / "model_v3_clean.ipynb"
BAK_PATH = NB_PATH.with_name("model_v3_clean.prepatch.ipynb")


# ---------------------------------------------------------------------------
# Patch-Definitionen
# ---------------------------------------------------------------------------

@dataclass
class Patch:
    """Eine Patch-Regel.

    `cell_id`         Vorrangig: die ID der Ziel-Zelle (z.B. 'v3-c23-summary').
    `fallback_needle` Falls die ID nicht mehr stimmt: ein eindeutiger Substring
                      im Quelltext der Ziel-Zelle.
    `already_patched` Substring, an dem erkannt wird, dass der Patch bereits
                      angewendet wurde (Idempotenz).
    `rewrite`         Funktion, die den alten Quelltext (str) bekommt und den
                      neuen Quelltext (str) zurückgibt. Outputs/Execution-Count
                      werden separat geleert (siehe `clear_outputs`).
    `clear_outputs`   Wenn True, werden cell['outputs'] und cell['execution_count']
                      zurückgesetzt — sinnvoll, wenn die Zelle einen alten
                      NameError-Traceback enthält.
    """

    cell_id: str
    fallback_needle: str
    already_patched: str
    rewrite: Callable[[str], str]
    clear_outputs: bool = True


# --- Patch 1: 23.17 — summary_blocks / HAS_HALVING ------------------------

PATCH_23_17_MARKER = "# Kapitel 22.15 wurde im Cleanup entfernt"

NEW_SOURCE_23_17 = """\
# Kapitel 22.15 wurde im Cleanup entfernt — das ursprüngliche
# `summary_blocks` existiert in dieser Notebook-Version nicht mehr.
# Wir bauen die Vergleichsliste hier selbst aus den Varianten zusammen,
# die im bereinigten Notebook tatsächlich noch existieren.
summary_blocks2 = [main_results.assign(Variant='Baseline (Kap. 10)')]

if HAS_LGBM and 'res_tuned' in globals() and tuned_lgbm is not None:
    summary_blocks2.append(res_tuned.assign(Variant='Tuned LGBM (22.3)'))

if len(stack_fitted) > 0:
    summary_blocks2.append(stack_results.assign(Variant='Stacking (23.2)'))
if has_full_geo and 'res_knn' in globals():
    summary_blocks2.append(res_knn.assign(Variant='KNN-Distance (23.12)'))
if HAS_LGBM and globals().get('HAS_HALVING', False) and 'res_halving' in globals():
    summary_blocks2.append(res_halving.assign(Variant='LGBM Halving-Tuned (23.10)'))

summary_v23 = (
    pd.concat(summary_blocks2, ignore_index=True)
    [['Variant', 'Model', 'RMSE Train', 'RMSE Eval', 'R² Eval', 'Overfitting Gap']]
    .sort_values('RMSE Eval')
    .reset_index(drop=True)
)
print('Vollständiger Vergleich aller Varianten (Kap. 10 / 22 / 23):')
display(summary_v23.head(25))

best_overall = summary_v23.iloc[0]
print(f'\\n🏆 Bester Run insgesamt: {best_overall["Variant"]} → {best_overall["Model"]}')
print(f'    RMSE Eval = {best_overall["RMSE Eval"]:.1f} CHF, R² = {best_overall["R² Eval"]:.3f}')
"""


def rewrite_23_17(_old: str) -> str:
    return NEW_SOURCE_23_17


# --- Patch 2: 27.2 — FEATURES_MINIMAL / TOP6 / FULL_KNN -------------------

PATCH_27_2_MARKER = "# Kapitel 27.1 wurde im Cleanup entfernt"

PREPEND_27_2 = """\
# Kapitel 27.1 wurde im Cleanup entfernt — die Feature-Sets, die in 27.2
# und 27.3 verwendet werden, definieren wir hier selbst, damit die Zelle
# self-contained ist.
FEATURES_MINIMAL = ['area', 'knn_price_median', 'east', 'north']
FEATURES_TOP6    = ['area', 'knn_price_median', 'knn_price_mean',
                     'east', 'north', 'area_per_room']
FEATURES_FULL_KNN = FEATURES_ENGINEERED + ['knn_price_mean', 'knn_price_median']

# Datenbasis: train_df_knn / eval_df_knn enthalten die KNN-Features (Kap. 23.12).
# Falls die KNN-Erweiterung übersprungen wurde, fallen wir auf train_df / eval_df
# zurück (dann fehlen halt die knn_price_*-Spalten und entsprechende Sets werden
# in der Schleife unten via 'missing columns' übersprungen).
_train_df_knn = train_df_knn if 'train_df_knn' in dir() else train_df
_eval_df_knn  = eval_df_knn  if 'eval_df_knn'  in dir() else eval_df

"""


def rewrite_27_2(old: str) -> str:
    # Einfügen direkt nach der `# === 27.2 ...`-Banner-Zeile, oder — falls die
    # nicht (mehr) da ist — ganz am Anfang.
    banner = "# === 27.2 Regularisierte LightGBM auf MINIMAL, TOP6, FULL+KNN ===\n"
    if banner in old:
        return old.replace(banner, banner + PREPEND_27_2, 1)
    return PREPEND_27_2 + old


# --- Patch 3: Export-Zelle (Kap. 21) — 4 fehlende Final-Report-Figures -----

PATCH_EXPORT_MARKER = "# === [auto] zusätzliche Figures für den Final Report ==="

# Diese Bilder werden im Final-Report (.tex) per \autofigure referenziert,
# aber von der Export-Zelle bisher NICHT erzeugt:
#   - feature_correlation.png
#   - learning_curve.png
#   - shap_summary.png
#   - spatial_residuals.png
#
# Wir hängen den Generator-Code direkt vor das abschliessende
# `print('\nFertig. Figures liegen in', FIG_DIR.resolve())` der Export-Zelle.
EXTRA_FIGURES_BLOCK = """\

# === [auto] zusätzliche Figures für den Final Report ===
# Vier Plots, die der Final-Report (.tex) via \\autofigure referenziert,
# aber die im ursprünglichen Export-Code fehlten. Jeder Block ist defensiv:
# fehlende Variablen / Libraries werden abgefangen, damit ein Fehler in
# einem Block die anderen drei nicht killt.

# --- feature_correlation.png --------------------------------------------
try:
    _corr_cols_pref = ['area', 'rooms', 'building_age', 'area_per_room',
                       'east', 'north', 'knn_price_median', 'knn_price_mean']
    _corr_cols = [c for c in _corr_cols_pref if c in train_df.columns]
    if len(_corr_cols) >= 3:
        _corr_df = train_df[_corr_cols + [TARGET_COL]].corr(numeric_only=True)
        _ordered = _corr_df[TARGET_COL].drop(TARGET_COL).abs().sort_values(ascending=False).index.tolist()
        _sub = _corr_df.loc[_ordered + [TARGET_COL], _ordered + [TARGET_COL]]
        fig, ax = plt.subplots(figsize=(7.5, 6))
        sns.heatmap(_sub, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1, square=True, ax=ax,
                    cbar_kws={'label': 'Pearson correlation'})
        ax.set_title('Feature correlation with price_cold')
        plt.tight_layout()
        _save(fig, 'feature_correlation.png')
except Exception as _e:
    print(f'  SKIP feature_correlation.png ({_e})')

# --- learning_curve.png -------------------------------------------------
try:
    from sklearn.model_selection import learning_curve as _learning_curve

    _lc_candidates = ['LightGBM', 'GradientBoosting', 'RandomForest']
    _lc_pick = next((m for m in _lc_candidates
                     if m in make_models() and (not m == 'LightGBM' or HAS_LGBM)),
                    None)
    if _lc_pick is not None:
        _lc_model = make_models()[_lc_pick]
        _sizes_rel, _train_scores, _val_scores = _learning_curve(
            _lc_model,
            train_df[main_features],
            train_df[TARGET_COL],
            train_sizes=np.linspace(0.15, 1.0, 6),
            cv=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            random_state=RANDOM_STATE,
            shuffle=True,
        )
        _train_rmse = -_train_scores.mean(axis=1)
        _val_rmse   = -_val_scores.mean(axis=1)
        _train_std  =  _train_scores.std(axis=1)
        _val_std    =  _val_scores.std(axis=1)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(_sizes_rel, _train_rmse, 'o-', label='Train RMSE', color='#1F4E79')
        ax.plot(_sizes_rel, _val_rmse,   's-', label='CV RMSE',    color='#E30613')
        ax.fill_between(_sizes_rel, _train_rmse - _train_std, _train_rmse + _train_std, alpha=0.15, color='#1F4E79')
        ax.fill_between(_sizes_rel, _val_rmse   - _val_std,   _val_rmse   + _val_std,   alpha=0.15, color='#E30613')
        ax.set_xlabel('Training-set size')
        ax.set_ylabel('RMSE (CHF)')
        ax.set_title(f'Learning curve — {_lc_pick}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _save(fig, 'learning_curve.png')
except Exception as _e:
    print(f'  SKIP learning_curve.png ({_e})')

# --- shap_summary.png ---------------------------------------------------
try:
    import shap as _shap
    _shap_model = None
    if 'main_fitted' in dir() and best_name in main_fitted:
        _shap_model = main_fitted[best_name]
    if _shap_model is not None:
        _shap_X = train_df[main_features].sample(
            min(1500, len(train_df)), random_state=RANDOM_STATE,
        )
        try:
            _explainer = _shap.TreeExplainer(_shap_model)
            _sv = _explainer.shap_values(_shap_X)
        except Exception:
            _explainer = _shap.Explainer(_shap_model, _shap_X)
            _sv = _explainer(_shap_X).values
        plt.figure(figsize=(8, 6))
        _shap.summary_plot(_sv, _shap_X, show=False, plot_size=None)
        fig = plt.gcf()
        plt.tight_layout()
        _save(fig, 'shap_summary.png')
    else:
        print('  SKIP shap_summary.png (best model not available in main_fitted)')
except ImportError:
    print('  SKIP shap_summary.png (shap not installed — pip install shap)')
except Exception as _e:
    print(f'  SKIP shap_summary.png ({_e})')

# --- spatial_residuals.png ----------------------------------------------
try:
    if {'east', 'north'} <= set(eval_df.columns):
        _abs_res = np.abs(_residuals)
        _scale_ref = max(np.percentile(_abs_res, 95), 1.0)
        _sizes = np.clip(8 + (_abs_res / _scale_ref) * 50, 4, 90)
        fig, ax = plt.subplots(figsize=(8.5, 7))
        sc = ax.scatter(eval_df['east'], eval_df['north'],
                        c=_abs_res, s=_sizes, alpha=0.55, cmap='magma',
                        edgecolors='none')
        plt.colorbar(sc, ax=ax, label='Absolute residual (CHF)')
        ax.set_xlabel('East coordinate (LV95)')
        ax.set_ylabel('North coordinate (LV95)')
        ax.set_title('Geographic distribution of absolute residuals')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _save(fig, 'spatial_residuals.png')
    else:
        print('  SKIP spatial_residuals.png (eval_df has no east/north)')
except Exception as _e:
    print(f'  SKIP spatial_residuals.png ({_e})')

"""


def rewrite_export_cell(old: str) -> str:
    final_print = "print('\\nFertig. Figures liegen in', FIG_DIR.resolve())"
    if final_print in old:
        # Vor dem abschliessenden Print einfügen.
        return old.replace(final_print, EXTRA_FIGURES_BLOCK + final_print, 1)
    # Fallback: einfach am Ende anhängen.
    return old + EXTRA_FIGURES_BLOCK


PATCHES: list[Patch] = [
    Patch(
        cell_id="v3-c23-summary",
        fallback_needle="summary_blocks2 = list(summary_blocks)",
        already_patched=PATCH_23_17_MARKER,
        rewrite=rewrite_23_17,
        clear_outputs=True,
    ),
    Patch(
        cell_id="v3-c27-2",
        fallback_needle="# === 27.2 Regularisierte LightGBM auf MINIMAL",
        already_patched=PATCH_27_2_MARKER,
        rewrite=rewrite_27_2,
        clear_outputs=True,
    ),
    Patch(
        cell_id="v3-c26-1",
        fallback_needle="# === Kapitel 26: Export aller Final-Report-Figures ===",
        already_patched=PATCH_EXPORT_MARKER,
        rewrite=rewrite_export_cell,
        clear_outputs=True,
    ),
]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def cell_source_str(cell: dict) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return src


def find_target_cell(nb: dict, patch: Patch) -> dict | None:
    for cell in nb["cells"]:
        if cell.get("id") == patch.cell_id:
            return cell
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        if patch.fallback_needle in cell_source_str(cell):
            return cell
    return None


def apply_patch(nb: dict, patch: Patch) -> str:
    """Returns one of 'patched', 'already_patched', 'not_found'."""
    target = find_target_cell(nb, patch)
    if target is None:
        return "not_found"

    src = cell_source_str(target)
    if patch.already_patched in src:
        return "already_patched"

    new_src = patch.rewrite(src)
    # Cell sources in .ipynb sind Listen von Zeilen mit \n am Ende.
    target["source"] = new_src.splitlines(keepends=True)
    if patch.clear_outputs:
        target["outputs"] = []
        target["execution_count"] = None
    return "patched"


def main() -> None:
    if not NB_PATH.exists():
        raise SystemExit(f"Notebook not found: {NB_PATH}")

    with NB_PATH.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    # Snapshot-Backup vor dem ersten echten Patch-Lauf.
    if not BAK_PATH.exists():
        shutil.copy2(NB_PATH, BAK_PATH)
        print(f"Backup written to {BAK_PATH.name}")

    any_patched = False
    for patch in PATCHES:
        status = apply_patch(nb, patch)
        label = patch.cell_id
        if status == "patched":
            print(f"[OK] patched cell '{label}'")
            any_patched = True
        elif status == "already_patched":
            print(f"[ -] cell '{label}' already patched")
        else:
            print(f"[!!] cell '{label}' not found — skipped")

    if any_patched:
        with NB_PATH.open("w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
            f.write("\n")
        print(f"\nWrote {NB_PATH.name}. Restart the kernel and run from the top.")
    else:
        print("\nNo changes written.")


if __name__ == "__main__":
    main()
