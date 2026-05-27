"""Build Figure 1 — residualized binscatter showing heat × stressor amplification.

For each stressor, partial out the demographic controls + fixed effects from
both CES-D z and daily mean temperature (Frisch-Waugh-Lovell), then plot the
residualised relationship separately for the exposed vs unexposed groups.
If heat amplifies the stressor effect, the residualised heat-CES-D slope is
steeper for the exposed group than for the unexposed group. The slope
difference equals the heat × stressor interaction β from the full regression.

Method (per stressor):
  Step 1. Residualise cesd_z on (controls + FE).
  Step 2. Residualise tmean_c on (controls + FE).
  Step 3. Within each subset (stressor = 0 / stressor exposed), bin the
          residualised heat into 20 quantile bins and compute mean
          (cesd_resid, heat_resid) per bin.
  Step 4. Plot the bin means as dots + a linear fit through each group's
          residualised data with 95% CI band (cluster-robust at kabupaten).

Exposed definition (continuous stressors): top quartile of POSITIVE shock
values — concentrates on observations that actually faced a meaningful shock.

Outputs:
  output/figures/figure1_interaction.pdf
  output/figures/figure1_interaction.png
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pyfixest as pf
import matplotlib as mpl
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "data" / "generated"
FIG = PROJECT / "output" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT / "code" / "analysis"))
from importlib import import_module
mod14 = import_module("14_unified_refined")
PALM_3MO_DECLINE = mod14.PALM_3MO_DECLINE

# --- Style ---
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman", "DejaVu Serif"],
    "font.size":   10,
    "axes.titlesize":  11,
    "axes.labelsize":  10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "axes.linewidth":  0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
})

COLOR_UNEXPOSED = "#5b8db8"
COLOR_EXPOSED   = "#c45a4f"

CONTROLS = "age + female + edu_yrs + married + widowed"


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(OUT / "analysis_dataset.parquet")
    fin  = pd.read_parquet(OUT / "financial_shocks.parquet")
    fin2 = pd.read_parquet(OUT / "financial_shocks_v2.parquet")
    df = df.merge(fin[["pidlink", "wave", "job_loss_within_yr"]],
                  on=["pidlink", "wave"], how="left")
    df = df.merge(fin2[["pidlink", "wave", "palm_farmer_hh", "transport_share"]],
                  on=["pidlink", "wave"], how="left")
    df["female"] = (df.sex == "F").astype(int)
    df = df.dropna(subset=[
        "cesd_raw", "tmean_c", "kab_code", "month", "year", "wave",
        "age", "female", "edu_yrs", "married", "widowed",
        "job_loss_within_yr", "palm_farmer_hh", "transport_share",
        "interview_date",
    ])
    df["cesd_z"] = df.groupby("wave")["cesd_raw"].transform(lambda s: (s - s.mean()) / s.std())
    df["intvw_ym"] = list(zip(df.interview_date.dt.year, df.interview_date.dt.month))
    df["palm_shock"] = df.palm_farmer_hh * df.intvw_ym.map(PALM_3MO_DECLINE).fillna(0.0)
    df["post_subsidy"] = (df.interview_date >= pd.Timestamp("2014-11-18")).astype(int)
    df["fuel_shock"] = df.post_subsidy * df.transport_share
    return df


def _restrict_singleton_kab(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.kab_code.value_counts()
    return df[df.kab_code.isin(counts[counts > 1].index)].copy()


def residualize(df: pd.DataFrame, var: str, fe: str) -> np.ndarray:
    """Partial out controls + FE from `var`."""
    formula = f"{var} ~ {CONTROLS} | {fe}"
    m = pf.feols(formula, data=df, vcov="iid")
    return df[var].values - m.predict(newdata=df)


def define_exposed(df: pd.DataFrame, stressor: str) -> np.ndarray:
    """Exposed group definition.

      Job loss : stressor == 1 (binary).
      Palm     : >= 25th percentile of positive shock values (broad group).
      Fuel     : top quartile of transport_share within post-cut sub-sample
                 (sharper contrast — see filter_for_panel for the
                 pre-restriction to post-cut only).
    """
    if stressor == "job_loss_within_yr":
        return (df[stressor] == 1).values
    if stressor == "fuel_shock":
        # Within (already post-cut-restricted) data, exposed = top quartile
        # of transport_share
        ts = df.transport_share
        threshold = ts.quantile(0.75)
        return (ts >= threshold).values
    # Continuous palm shock: top 75% of positive shock values (broad)
    s = df[stressor]
    pos = s[s > 0]
    if len(pos) == 0:
        return np.zeros(len(df), dtype=bool)
    threshold = pos.quantile(0.25)
    return (s >= threshold).values


def filter_for_panel(df: pd.DataFrame, stressor: str) -> pd.DataFrame:
    """Optionally restrict the working sample before residualisation.

    For fuel cut we restrict to post-cut IFLS5 observations only, so the
    binscatter compares high-transport vs low-transport households *within the
    same calendar period*, isolating the fuel-exposure dimension."""
    if stressor == "fuel_shock":
        return df[df.post_subsidy == 1].copy()
    return df


def binscatter_means(x: np.ndarray, y: np.ndarray, n_bins: int = 20):
    if len(x) == 0:
        return np.array([]), np.array([])
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(x, quantiles)
    edges[0]  -= 1e-9
    edges[-1] += 1e-9
    bin_idx = np.digitize(x, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    xb = np.full(n_bins, np.nan)
    yb = np.full(n_bins, np.nan)
    for k in range(n_bins):
        mask = bin_idx == k
        if mask.sum() > 0:
            xb[k] = x[mask].mean()
            yb[k] = y[mask].mean()
    return xb, yb


def linear_fit_with_ci(df: pd.DataFrame, y_col: str, x_col: str,
                       cluster_col: str, x_grid: np.ndarray):
    """Fit y = a + b·x with cluster-robust SEs and return predicted line + 95% CI."""
    if len(df) < 5:
        zeros = np.zeros_like(x_grid)
        return zeros, zeros, zeros
    m = pf.feols(f"{y_col} ~ {x_col}", data=df, vcov={"CRV1": cluster_col})
    coefs = m.coef()
    V = pd.DataFrame(m._vcov, index=coefs.index, columns=coefs.index)
    a = coefs.get("Intercept", 0.0)
    b = coefs.get(x_col, 0.0)
    y_hat = a + b * x_grid
    var_a  = V.loc["Intercept", "Intercept"] if "Intercept" in V.index else 0.0
    var_b  = V.loc[x_col, x_col]              if x_col in V.index else 0.0
    cov_ab = V.loc["Intercept", x_col]        if ("Intercept" in V.index and x_col in V.index) else 0.0
    var_yhat = var_a + (x_grid ** 2) * var_b + 2.0 * x_grid * cov_ab
    se_yhat  = np.sqrt(np.clip(var_yhat, 0, None))
    return y_hat, y_hat - 1.96 * se_yhat, y_hat + 1.96 * se_yhat


def fit_full_interaction(df: pd.DataFrame, stressor: str, extra: str, fe: str):
    df_work = df.copy()
    df_work["heat_c_dev"] = df_work.tmean_c - df_work.tmean_c.mean()
    formula = f"cesd_z ~ heat_c_dev * {stressor}{extra} + {CONTROLS} | {fe}"
    m = pf.feols(formula, data=df_work, vcov={"CRV1": "kab_code"})
    inter = f"heat_c_dev:{stressor}"
    return m.coef()[inter], m.pvalue()[inter]


def stars(p):
    if pd.isna(p): return ""
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def main() -> None:
    df = load_data()
    df = _restrict_singleton_kab(df)
    sub_ifls5 = _restrict_singleton_kab(df[df.wave == "IFLS5"].copy())

    FE_POOLED = "month + year + wave + kab_code"
    FE_IFLS5  = "month + year + kab_code"

    panels = [
        {"label": "Job loss × Heat",
         "stressor": "job_loss_within_yr",
         "extra": "", "data": df, "fe": FE_POOLED,
         "unexposed_lbl": "Not recently job-lost",
         "exposed_lbl":   "Recently job-lost (≤12 mo)",
        },
        {"label": "Palm shock × Heat",
         "stressor": "palm_shock",
         "extra": " + palm_farmer_hh", "data": df, "fe": FE_POOLED,
         "unexposed_lbl": "No palm shock",
         "exposed_lbl":   "Palm farmer w/ price decline",
        },
        {"label": "Fuel cut × Heat (within post-cut)",
         "stressor": "fuel_shock",
         "extra": " + transport_share", "data": sub_ifls5, "fe": FE_IFLS5,
         "unexposed_lbl": "Post-cut, low transport share (bottom 75%)",
         "exposed_lbl":   "Post-cut, high transport share (top 25%)",
        },
    ]

    results = []
    for p in panels:
        # Pre-filter the working data (fuel cut → post-cut only)
        d = filter_for_panel(p["data"].copy(), p["stressor"])
        d["cesd_resid"] = residualize(d, "cesd_z", p["fe"])
        d["heat_resid"] = residualize(d, "tmean_c", p["fe"])
        # Drop NaN residuals (can occur for singleton FE cells)
        d = d.dropna(subset=["cesd_resid", "heat_resid"]).copy()
        d["exposed"] = define_exposed(d, p["stressor"])
        un = d[~d.exposed]
        ex = d[d.exposed]
        un_x, un_y = binscatter_means(un.heat_resid.values, un.cesd_resid.values, n_bins=20)
        ex_x, ex_y = binscatter_means(ex.heat_resid.values, ex.cesd_resid.values, n_bins=20)
        # Full interaction from Table 1 spec (unrestricted, for annotation)
        beta_i, p_i = fit_full_interaction(p["data"], p["stressor"], p["extra"], p["fe"])
        results.append({
            "panel": p, "un_df": un, "ex_df": ex,
            "un_x": un_x, "un_y": un_y, "ex_x": ex_x, "ex_y": ex_y,
            "beta_i": beta_i, "p_i": p_i,
            "n_un": len(un), "n_ex": len(ex),
        })

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3))
    for ax, r in zip(axes, results):
        p = r["panel"]
        # x-range from 1st-99th percentiles of the *underlying observation-level*
        # residualised heat (wider than just the bin centers; nan-safe)
        all_heat = np.concatenate([r["un_df"].heat_resid.values, r["ex_df"].heat_resid.values])
        all_heat = all_heat[np.isfinite(all_heat)]
        if len(all_heat) == 0:
            x_min, x_max = -1.0, 1.0
        else:
            x_lo = np.nanpercentile(all_heat, 1)
            x_hi = np.nanpercentile(all_heat, 99)
            x_pad = 0.05 * (x_hi - x_lo)
            x_min, x_max = x_lo - x_pad, x_hi + x_pad
        x_line = np.linspace(x_min, x_max, 100)

        # CI bands
        un_fit, un_lo, un_hi = linear_fit_with_ci(
            r["un_df"], "cesd_resid", "heat_resid", "kab_code", x_line
        )
        ex_fit, ex_lo, ex_hi = linear_fit_with_ci(
            r["ex_df"], "cesd_resid", "heat_resid", "kab_code", x_line
        )
        ax.fill_between(x_line, un_lo, un_hi, color=COLOR_UNEXPOSED, alpha=0.15, lw=0,
                        zorder=1)
        ax.fill_between(x_line, ex_lo, ex_hi, color=COLOR_EXPOSED,   alpha=0.15, lw=0,
                        zorder=1)
        # Fit lines
        ax.plot(x_line, un_fit, color=COLOR_UNEXPOSED, lw=1.8, zorder=2)
        ax.plot(x_line, ex_fit, color=COLOR_EXPOSED,   lw=1.8, zorder=2)
        # Bin-mean dots
        ax.scatter(r["un_x"], r["un_y"], color=COLOR_UNEXPOSED, s=40, alpha=0.85,
                   edgecolor="white", linewidth=0.7, zorder=3,
                   label=f"{p['unexposed_lbl']} (n={r['n_un']:,})")
        ax.scatter(r["ex_x"], r["ex_y"], color=COLOR_EXPOSED, s=40, alpha=0.85,
                   edgecolor="white", linewidth=0.7, zorder=3,
                   label=f"{p['exposed_lbl']} (n={r['n_ex']:,})")
        # Reference lines
        ax.axhline(0, color="grey", lw=0.6, ls="--", alpha=0.6)
        ax.axvline(0, color="grey", lw=0.6, ls="--", alpha=0.4)

        ax.set_title(
            f"{p['label']}\n"
            + r"$\beta_{Heat \times Stress}$ = "
            + f"{r['beta_i']:+.3f}{stars(r['p_i'])} (p={r['p_i']:.3f})",
            fontsize=10,
        )
        ax.set_xlabel("Daily mean temperature, residualised (°C)")
        if ax is axes[0]:
            ax.set_ylabel("CES-D z, residualised (SD)")
        ax.legend(loc="upper left", frameon=False, fontsize=8.5)
        ax.tick_params(axis="both", length=3, width=0.6)
        ax.set_xlim(x_min, x_max)

    plt.tight_layout()

    pdf_path = FIG / "figure1_interaction.pdf"
    png_path = FIG / "figure1_interaction.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    plt.close(fig)

    print("\n=== Numerical summary ===")
    for r in results:
        p = r["panel"]
        m_un = pf.feols("cesd_resid ~ heat_resid", data=r["un_df"], vcov={"CRV1":"kab_code"})
        m_ex = pf.feols("cesd_resid ~ heat_resid", data=r["ex_df"], vcov={"CRV1":"kab_code"})
        b_un = m_un.coef().get("heat_resid", 0); se_un = m_un.se().get("heat_resid", 0)
        b_ex = m_ex.coef().get("heat_resid", 0); se_ex = m_ex.se().get("heat_resid", 0)
        print(f"\n{p['label']}  (n_un = {r['n_un']:,}, n_ex = {r['n_ex']:,})")
        print(f"  Unexposed slope: {b_un:+.4f}  (SE {se_un:.4f})")
        print(f"  Exposed   slope: {b_ex:+.4f}  (SE {se_ex:.4f})")
        print(f"  Slope difference (binscatter approx):     {b_ex - b_un:+.4f}")
        print(f"  Full-spec interaction coefficient β_HxS:  {r['beta_i']:+.4f}{stars(r['p_i'])}  (p={r['p_i']:.4f})")


if __name__ == "__main__":
    main()
