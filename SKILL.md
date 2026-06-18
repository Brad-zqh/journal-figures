---
name: shap-figures
description: Produce top-journal / Nature-style figures for ML, SHAP, GAM and GIS / big-data work — unified coolwarm palette (blue=low/negative, red=high/positive), big clear serif fonts, labelled (a)(b)(c) composites. Use when the user wants to beautify or generate publication-quality plots: SHAP beeswarm/summary, feature-importance bars + nightingale rose, GAM dependence with CI & thresholds, model predicted-vs-actual scatter, spatial SHAP point maps with north arrow / scale bar, monthly trend lines, ablation grids, or multi-panel combined figures. Provides a ready-to-import helper library (scripts/shap_viz.py).
---

# Journal-grade SHAP figures (coolwarm toolkit)

A reusable toolkit for publication-quality ML / SHAP / GAM / GIS visualisation.
Use the bundled `scripts/shap_viz.py` (axes-level helpers — you own the
`Figure`/`GridSpec`, the helpers draw into your `Axes`).

## When to use
The user asks to "beautify", "make journal-quality / Nature-style", or generate:
SHAP beeswarm/summary, importance (bar + rose), GAM dependence, model accuracy
scatter, spatial SHAP maps, monthly trend, ablation, or combined multi-panel figures.

## Design rules (keep consistent)
- **Coolwarm everywhere**: blue = low / negative, red = high / positive.
  `CMAP_FV` soft default; `CMAP_DEEP` for beeswarm (deeper ends); `CMAP_SEQ` warm
  sequential (density / importance); standard `coolwarm` for maps & model scatter.
- **Categories**: pick 3 colours (e.g. red / blue / grey) and colour feature names,
  bars and rose wedges consistently.
- **Big, clear serif fonts** (Times New Roman → DejaVu Serif fallback); hairline
  spines; no chart-junk; titles can live in filenames.
- **Composites** get compact `(a) … / (b) … / (c) …` labels via `stack_images(labels=…)`.
- Dependence panels annotate **R²**, **p**, and a Positive/Negative/Threshold legend
  *inside* each subplot. Keep feature order **consistent** across grouped panels.
- Maps: light CartoDB Positron basemap, north arrow + scale bar, deeper coolwarm.
- Save with `bbox_inches="tight"`, dpi 300–400.

## How to use the library
```python
import sys; sys.path.insert(0, os.path.expanduser(
    "~/.claude/skills/shap-figures/scripts"))
import shap_viz as jv
jv.set_style()                       # global serif / sizes

# SHAP beeswarm (one panel)
jv.beeswarm(ax, shap_mat, feat_vals, names, cmap=jv.CMAP_DEEP,
            xlim=(lo, hi), name_colors=colors)   # xlim trims extreme points
jv.add_colorbar(ax, fig, cmap=jv.CMAP_DEEP)

# Importance bar + nightingale rose
jv.importance_panel(ax, importances, names, colors, letter="a", title="Revenue")

# GAM dependence (needs pygam)
jv.gam_dependence(ax, x, shap_x, name="Vegetation", x_transform=to_original_units)

# Spatial SHAP map (xs, ys in EPSG:3857 if basemap=True)
norm = jv.coolwarm_point_map(ax, xs, ys, shap_vals, basemap=True)
jv.north_arrow(ax); jv.scale_bar(ax, length_units=20000, label="20 km")

# Combine saved panels into a labelled composite
jv.stack_images(paths, "combined.png", vertical=True,
                labels=jv.dep_labels(["Revenue", "ADR", "RevPAR"]))
```

## Function reference (`shap_viz.py`)
- `set_style(serif, base)` — global rcParams.
- `beeswarm(ax, shap_mat, feat_vals, names, cmap, dot, xlim, spread_frac, name_colors)`.
- `add_colorbar(ax, fig, cmap, label, full_height)` — Low/High slim bar.
- `importance_panel(ax, importances, names, colors, letter, title, with_rose, fs)`.
- `gam_fit(x, y)` → grid, curve, CI, crossings, R², p. `gam_dependence(ax, x, y, …)`.
- `coolwarm_point_map(ax, xs, ys, values, pct, basemap)` → TwoSlopeNorm.
- `north_arrow(ax)`, `scale_bar(ax, length_units, label)`.
- `stack_images(paths, out, vertical, labels, label_frac)`, `dep_labels(names)`.
- `cat_gradient(base_colors, n)` — graded ramp; `despine(ax)`; palette constants
  `CMAP_FV / CMAP_DEEP / CMAP_SEQ / CAT_COLOR / INK / GRID`.

## Notes
- Optional deps: `pygam` (dependence), `contextily` + `pyproj` (basemaps), `Pillow`
  (`stack_images`). The library imports them lazily so unrelated calls don't fail.
- Helpers are dataset-agnostic: pass arrays / values, not file paths.
- When editing Jupyter cells programmatically, read content from a file then write —
  don't put back-tick text inside `python3 -c "…"` (the shell eats back-ticks).
