"""
journal_viz.py — Journal-grade figure toolkit (coolwarm theme)
==============================================================
Dataset-agnostic helpers for Nature/top-journal style ML / SHAP / GIS figures.
Everything is axes-level (you own the Figure/GridSpec), so it composes freely.

Conventions
-----------
- Unified coolwarm: BLUE = low / negative, RED = high / positive.
- Big, clear serif fonts; hairline spines; no clutter.
- Multi-panel composites are labelled (a) / (b) / (c) ...

Quick start
-----------
    import journal_viz as jv
    jv.set_style()
    fig, ax = plt.subplots()
    jv.gam_dependence(ax, x, shap_x, name="Vegetation")

Optional deps: pygam (GAM dependence), contextily + pyproj (basemaps).
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

# --------------------------------------------------------------------- palette
INK, GRID = "#2B2B2B", "#D9D9D9"
CMAP_FV = LinearSegmentedColormap.from_list(
    "soft_coolwarm", ["#5566C8", "#7E8FDD", "#AEBCEF", "#DDE2F2", "#F2F2F2",
                      "#F3CEC6", "#EC9C90", "#DC6356", "#C83A33"])
CMAP_DEEP = LinearSegmentedColormap.from_list(
    "deep_coolwarm", ["#15166A", "#2A37A8", "#5E74D6", "#A9B8EC", "#EDEFF4",
                      "#F1C2B7", "#E07764", "#C01F33", "#6E0010"])
CMAP_SEQ = LinearSegmentedColormap.from_list(
    "cw_warm", ["#EDEEF2", "#E5A187", "#D65244", "#B40426", "#7A0019"])
# generic 3-class category colours (override as needed)
CAT_COLOR = {"A": "#B40426", "B": "#3B4CC0", "C": "#7B7B7B"}


def set_style(serif="Times New Roman", base=14):
    """Global rcParams — clean serif, hairline spines."""
    avail = {f.name for f in mpl.font_manager.fontManager.ttflist}
    if serif not in avail:
        serif = "DejaVu Serif"
    mpl.rcParams.update({
        "font.family": serif, "mathtext.fontset": "stix",
        "font.size": base, "axes.titlesize": base + 2, "axes.labelsize": base + 1,
        "axes.edgecolor": INK, "axes.linewidth": 1.0, "axes.facecolor": "white",
        "axes.grid": False, "xtick.color": INK, "ytick.color": INK,
        "xtick.labelsize": base - 1, "ytick.labelsize": base - 1,
        "legend.fontsize": base - 2, "legend.frameon": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "savefig.dpi": 400, "figure.dpi": 110, "svg.fonttype": "none",
    })


def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def cat_gradient(base_colors, n):
    """n dark→light colours along a ramp (for graded bars / lines)."""
    cm = LinearSegmentedColormap.from_list("g", base_colors)
    return [cm(t) for t in np.linspace(0.0, 0.85, max(n, 1))]


# ----------------------------------------------------------------- beeswarm
def beeswarm(ax, shap_mat, feat_vals, feat_names, cmap=None, dot=14, xlim=None,
             spread_frac=0.5, max_points=2500, name_colors=None, rng=0):
    """SHAP beeswarm. shap_mat,(n,k); feat_vals,(n,k) raw values for colour.
    xlim=(lo,hi) drops extreme points so the bulk fills more area."""
    cmap = cmap or CMAP_FV
    rs = np.random.default_rng(rng)
    n, k = shap_mat.shape
    if n > max_points:
        sel = rs.choice(n, max_points, replace=False)
        shap_mat, feat_vals = shap_mat[sel], feat_vals[sel]
    order = list(range(k))[::-1]
    for yi, j in enumerate(order):
        if yi % 2 == 0:
            ax.axhspan(yi - 0.5, yi + 0.5, color="#F4F5F8", zorder=0)
        sv = shap_mat[:, j]; fv = feat_vals[:, j].astype(float)
        if xlim is not None:
            keep = (sv >= xlim[0]) & (sv <= xlim[1]); sv, fv = sv[keep], fv[keep]
        lo, hi = np.nanpercentile(fv, [5, 95])
        c = np.clip((fv - lo) / (hi - lo + 1e-9), 0, 1)
        nb = 90
        bins = np.linspace(sv.min(), sv.max() + 1e-9, nb + 1)
        idx = np.clip(np.digitize(sv, bins) - 1, 0, nb - 1)
        yoff = np.zeros_like(sv)
        for b in range(nb):
            m = np.where(idx == b)[0]
            if len(m) == 0:
                continue
            sp = min(0.9 * spread_frac, 0.022 * np.sqrt(len(m)))
            o = np.linspace(-sp, sp, len(m))
            yoff[m] = o[np.argsort(np.argsort(np.abs(o)))]
        ax.scatter(sv, yi + yoff, c=c, cmap=cmap, s=dot, alpha=0.88,
                   linewidths=0.12, edgecolors="white", zorder=2, vmin=0, vmax=1)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.axvline(0, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.set_yticks(range(k))
    ax.set_yticklabels([feat_names[j] for j in order])
    if name_colors is not None:
        for tick, j in zip(ax.get_yticklabels(), order):
            tick.set_color(name_colors[j])
    ax.set_ylim(-0.6, k - 0.4)
    despine(ax, keep=("bottom",)); ax.tick_params(axis="y", length=0)


def add_colorbar(ax, fig, cmap=None, label="Feature value", full_height=True):
    """Slim Low/High colorbar glued to the right of an axes."""
    cmap = cmap or CMAP_FV
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
    cax = ax.inset_axes([1.012, 0.0 if full_height else 0.06,
                         0.022, 1.0 if full_height else 0.88])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_ticks([0, 1]); cb.set_ticklabels(["Low", "High"])
    cb.set_label(label, rotation=270, labelpad=13)
    cb.outline.set_linewidth(0.5)
    return cb


# ---------------------------------------------------- importance bar + rose
def _txt_on(c):
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(c)
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.62 else INK


def importance_panel(ax, importances, names, colors, letter=None, title="",
                     with_rose=True, fs=18):
    """Horizontal importance bars (names outside) + optional nightingale rose
    inset (names inside wedges). `importances`,`names`,`colors` aligned, sorted
    descending by the caller (or not — we sort here)."""
    imp = np.asarray(importances, float)
    o = np.argsort(-imp)
    imp, names, colors = imp[o], [names[i] for i in o], [colors[i] for i in o]
    n = len(imp); yv = np.arange(n)[::-1]
    ax.barh(yv, imp, color=colors, edgecolor="white", lw=0.6, height=0.74)
    vmax = imp.max()
    for y, v in zip(yv, imp):
        ax.text(v + vmax * 0.015, y, f"{v:.3g}", va="center", fontsize=fs - 4, color=INK)
    ax.set_yticks(yv); ax.set_yticklabels(names, fontsize=fs)
    for tick, c in zip(ax.get_yticklabels(), colors):
        tick.set_color(c)
    ax.set_xlim(0, vmax * 1.24); ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("mean |SHAP value|", fontsize=fs - 1)
    ax.tick_params(axis="x", labelsize=fs - 4)
    if letter or title:
        ax.text(0.0, 1.02, f"({letter}) {title}".strip(), transform=ax.transAxes,
                fontsize=fs + 3, fontweight="bold")
    despine(ax, keep=("bottom",)); ax.tick_params(axis="y", length=0)
    if with_rose:
        share = imp / imp.sum()
        axr = ax.inset_axes([0.40, 0.0, 0.66, 0.66], projection="polar")
        w = 2 * np.pi / n; theta = np.arange(n) * w + np.pi / 2; hole = 0.32
        rl = np.sqrt(share); rl = rl / rl.max()
        axr.bar(theta, rl, width=w * 0.93, color=colors, edgecolor="white",
                lw=0.8, align="edge", bottom=hole)
        axr.set_theta_direction(-1); axr.set_xticks([]); axr.set_yticks([])
        axr.spines["polar"].set_visible(False); axr.set_ylim(0, hole + 1.0)
        axr.patch.set_alpha(0)
        for t, nm, rr, cc in zip(theta, names, rl, colors):
            ang = t + w / 2; rot = np.degrees(ang)
            if 90 < (rot % 360) < 270:
                rot += 180
            axr.text(ang, hole + rr * 0.55, nm, rotation=rot, rotation_mode="anchor",
                     ha="center", va="center", fontsize=fs - 8, fontweight="bold",
                     color=_txt_on(cc))


# ------------------------------------------------------- GAM dependence panel
def gam_fit(x, y, n_grid=200):
    """Return grid, mean curve, 95% CI, zero-crossings, pseudo-R², p-value."""
    from pygam import LinearGAM, s
    x = np.asarray(x, float); y = np.asarray(y, float)
    nuniq = len(np.unique(x))
    xx = np.linspace(np.nanmin(x), np.nanmax(x), n_grid)
    if nuniq <= 2:
        means = [y[x == v].mean() for v in np.unique(x)]
        return xx, np.interp(xx, np.unique(x), means), None, np.array([]), np.nan, np.nan
    rs = np.random.default_rng(0)
    idx = rs.choice(len(x), min(8000, len(x)), replace=False)
    nspl = int(np.clip(nuniq, 6, 20))
    gam = LinearGAM(s(0, n_splines=nspl)).fit(x[idx].reshape(-1, 1), y[idx])
    pred = gam.predict(xx.reshape(-1, 1))
    ci = gam.confidence_intervals(xx.reshape(-1, 1), width=0.95)
    cross = xx[:-1][np.diff(np.sign(pred)) != 0]
    try:
        r2 = gam.statistics_["pseudo_r2"]["explained_deviance"]
    except Exception:
        r2 = np.nan
    try:
        pval = float(np.atleast_1d(gam.statistics_["p_values"])[0])
    except Exception:
        pval = np.nan
    return xx, pred, ci, cross, r2, pval


def gam_dependence(ax, x, y, name="", color=None, small=False, legend=True,
                   x_transform=None):
    """SHAP dependence panel: GAM fit + CI + red(pos)/blue(neg) fill + thresholds,
    annotated with R² and p inside the panel. x_transform(x)->original units."""
    NAVY, CIB, POS, NEG, THR = "#27408B", "#9AA7C2", "#EBA0A0", "#A6BEE8", "#7A0019"
    xg, pred, ci, cross, r2, pval = gam_fit(x, y)
    xp = x_transform(xg) if x_transform else xg
    ax.fill_between(xp, pred, 0, where=pred >= 0, color=POS, alpha=0.78, interpolate=True, zorder=1)
    ax.fill_between(xp, pred, 0, where=pred < 0, color=NEG, alpha=0.78, interpolate=True, zorder=1)
    if ci is not None:
        ax.fill_between(xp, ci[:, 0], ci[:, 1], color=CIB, alpha=0.45, lw=0, zorder=2)
    ax.plot(xp, pred, color=NAVY, lw=2.6, zorder=4)
    ax.axhline(0, color="#7a7a7a", lw=1.0, ls=(0, (4, 3)), zorder=3)
    for cx in cross[:4]:
        cxp = x_transform(cx) if x_transform else cx
        ax.axvline(cxp, color=THR, lw=1.1, ls="--", zorder=3)
        ax.annotate(f"{cxp:.2f}", xy=(cxp, 0), xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5 if small else 9.5,
                    color=THR, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=THR, lw=0.8))
    if np.isfinite(r2):
        ptxt = "p < 0.01" if (not np.isfinite(pval) or pval < 0.01) else f"p = {pval:.2g}"
        ax.text(0.04, 0.95, f"$R^2$ = {max(r2,0)*100:.1f}%\n{ptxt}", transform=ax.transAxes,
                va="top", ha="left", fontsize=10 if small else 11, fontweight="bold", color=INK)
    if name:
        ax.set_title(name, loc="left", fontsize=12.5 if small else 14,
                     fontweight="bold", color=color or INK)
    ax.set_xlabel("Feature value", fontsize=11 if small else 12.5)
    ax.set_ylabel("SHAP value", fontsize=11 if small else 12.5)
    despine(ax); ax.tick_params(labelsize=10 if small else 11.5)
    if legend:
        leg = [Line2D([0], [0], color=NAVY, lw=2.6, label="GAM fit"),
               Patch(facecolor=POS, label="Positive"), Patch(facecolor=NEG, label="Negative"),
               Line2D([0], [0], color=THR, lw=1.2, ls="--", label="Threshold")]
        ax.legend(handles=leg, loc="lower right", fontsize=7.5 if small else 8.5,
                  handlelength=1.2, labelspacing=0.25, borderpad=0.35,
                  framealpha=0.85, facecolor="white", edgecolor="0.8")


# ----------------------------------------------------------- map decorations
def north_arrow(ax, loc=(0.93, 0.88), size=0.10, fs=18):
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    cx = x0 + (x1 - x0) * loc[0]; cy = y0 + (y1 - y0) * loc[1]
    h = (y1 - y0) * size; w = h * 0.34
    top, bot, mid = cy + h * 0.45, cy - h * 0.35, cy - h * 0.05
    ax.fill([cx, cx - w / 2, cx], [top, bot, mid], color=INK, zorder=20)
    ax.fill([cx, cx + w / 2, cx], [top, bot, mid], facecolor="white",
            edgecolor=INK, lw=0.6, zorder=20)
    ax.text(cx, top + h * 0.1, "N", ha="center", va="bottom", fontsize=fs,
            fontweight="bold", color=INK, zorder=20)


def scale_bar(ax, length_units, n_seg=4, loc=(0.06, 0.06), label="", fs=11):
    """Black/white scale bar. length_units is in axis (projected) units."""
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    xs = x0 + (x1 - x0) * loc[0]; ys = y0 + (y1 - y0) * loc[1]
    h = (y1 - y0) * 0.013; seg = length_units / n_seg
    for i in range(n_seg):
        c = "black" if i % 2 == 0 else "white"
        ax.add_patch(Rectangle((xs + seg * i, ys), seg, h, facecolor=c,
                               edgecolor=INK, lw=0.4, zorder=20))
    ax.text(xs, ys + h * 1.6, "0", ha="center", va="bottom", fontsize=fs, zorder=20)
    ax.text(xs + length_units, ys + h * 1.6, label, ha="center", va="bottom",
            fontsize=fs, fontweight="bold", zorder=20)


def coolwarm_point_map(ax, xs, ys, values, point_size=24, pct=85, cmap=None,
                       basemap=False, web_mercator=True):
    """Diverging SHAP point map (values centred at 0). xs,ys must be in EPSG:3857
    if basemap=True. Returns the TwoSlopeNorm used."""
    cmap = cmap or plt.get_cmap("coolwarm")
    vmax = np.nanpercentile(np.abs(values), pct) or 1e-6
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    if basemap:
        try:
            import contextily as ctx
            ax.scatter(xs, ys, c=np.clip(values, -vmax, vmax), cmap=cmap, norm=norm,
                       s=point_size, alpha=0.95, linewidths=0.25, edgecolors="white", zorder=5)
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels,
                            attribution="", zorder=0)
        except Exception as e:
            print("basemap skipped:", e)
    else:
        ax.scatter(xs, ys, c=np.clip(values, -vmax, vmax), cmap=cmap, norm=norm,
                   s=point_size, alpha=0.95, linewidths=0.25, edgecolors="white", zorder=5)
    ax.set_xticks([]); ax.set_yticks([])
    return norm


# ----------------------------------------------------- labelled image stacking
def _pil_font(size):
    from PIL import ImageFont
    for p in ["/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
              "/System/Library/Fonts/Supplemental/Times New Roman.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    try:
        return ImageFont.truetype(mpl.font_manager.findfont("DejaVu Serif"), size)
    except Exception:
        return ImageFont.load_default()


def stack_images(paths, out, vertical=True, pad_frac=0.010, labels=None,
                 label_frac=0.040):
    """Compose saved figures onto one canvas; optional compact (a)/(b)/(c) labels."""
    from PIL import Image, ImageDraw
    imgs = [Image.open(p).convert("RGB") for p in paths]
    if vertical:
        W = max(i.width for i in imgs); pad = int(W * pad_frac)
        lab_h = int(W * label_frac) if labels else 0
        font = _pil_font(int(lab_h * 0.66)) if labels else None
        H = sum(i.height for i in imgs) + (lab_h + pad) * len(imgs) + pad
        cv = Image.new("RGB", (W + 2 * pad, H), "white"); d = ImageDraw.Draw(cv); y = pad
        for k, im in enumerate(imgs):
            if labels:
                d.text((pad + int(W * 0.01), y), labels[k], fill=(26, 26, 26), font=font); y += lab_h
            cv.paste(im, ((W - im.width) // 2 + pad, y)); y += im.height + pad
    else:
        H = max(i.height for i in imgs); pad = int(H * pad_frac)
        lab_h = int(H * label_frac) if labels else 0
        font = _pil_font(int(lab_h * 0.66)) if labels else None
        W = sum(i.width for i in imgs) + pad * (len(imgs) + 1)
        cv = Image.new("RGB", (W, H + lab_h + 2 * pad), "white"); d = ImageDraw.Draw(cv); x = pad
        for k, im in enumerate(imgs):
            if labels:
                d.text((x + int(H * 0.01), int(pad * 0.4)), labels[k], fill=(26, 26, 26), font=font)
            cv.paste(im, (x, lab_h + pad)); x += im.width + pad
    cv.save(out, dpi=(300, 300)); return out


def dep_labels(names):
    """['Revenue','ADR'] -> ['(a) Revenue','(b) ADR']."""
    return [f"({chr(97 + k)}) {n}" for k, n in enumerate(names)]
