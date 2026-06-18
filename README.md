# journal-figures

A [Claude Code](https://claude.com/claude-code) **Skill** for producing
top-journal / Nature-style figures for **ML, SHAP, GAM and GIS / big-data** work.

Unified **coolwarm** palette (blue = low / negative, red = high / positive),
big clear serif fonts, and labelled `(a) (b) (c)` composites. Ships a
ready-to-import, dataset-agnostic helper library.

## What it draws
- SHAP **beeswarm / summary** (deeper-end coolwarm, extreme-point trimming, per-panel colorbar)
- Feature **importance** bars + nightingale rose (category-coloured)
- **GAM dependence** with 95% CI, red/blue fills, thresholds, in-panel R² & p
- Model **predicted-vs-actual** scatter (density-shaded)
- Spatial **SHAP point maps** with north arrow + scale bar (coolwarm diverging)
- **Monthly trend** lines, **ablation** grids
- Labelled **multi-panel composites** (`stack_images`)

## Install
Clone into your personal Claude Code skills folder:

```bash
git clone https://github.com/Brad-zqh/journal-figures \
  ~/.claude/skills/journal-figures
```

Restart your Claude Code session, then trigger with `/journal-figures` or just
ask for "journal-quality SHAP / dependence / map figures".

## Use the library directly
```python
import sys, os
sys.path.insert(0, os.path.expanduser("~/.claude/skills/journal-figures/scripts"))
import journal_viz as jv

jv.set_style()
fig, ax = plt.subplots()
jv.gam_dependence(ax, x, shap_x, name="Vegetation")     # GAM + CI + thresholds
```

See [`SKILL.md`](SKILL.md) for the full design rules and function reference.

## Requirements
`numpy`, `pandas`, `matplotlib`, `scipy`. Optional (lazy-imported):
`pygam` (dependence), `contextily` + `pyproj` (basemaps), `Pillow` (`stack_images`).

## License
MIT
