"""Table S: placebo tests for the palm x heat amplification.

Two placebos for the headline triple Heat x palm x IFLS5:

  Panel A -- Crop placebo (actual cultivation). IFLS records the crop a household
    grows (b2_ut1 ut07a/ut07b); there is no oil-palm code, so palm stays region x
    agricultural, but rice/maize/rubber/coffee/etc. are identified by true cultivation.
    Run the same wave-DiD triple per crop and read it against the crop's 2007-08 ->
    2014-15 price change. The major staples (rice, maize) -- the cleanest, largest
    grower groups -- should be null if the palm effect is crop-specific rather than
    "every farmer got more heat-sensitive in 2014."

  Panel B -- Fake-shock placebo. Randomly reassign palm-farmer status across
    households and re-estimate the triple; the true coefficient should sit far outside
    the permutation null (randomization-inference p).

Heat measure: tmean_7d_dev. Crop codes read from raw IFLS via the code/data/config.py path.
Cross-wave price changes are WB Commodity Markets Outlook (Pink Sheet), 2007-08 vs 2014-15.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pyfixest as pf

from _lettered_common import (
    CLUSTER,
    CONTROL_COLUMNS,
    CONTROLS,
    KECAMATAN_FE,
    PROJECT,
    cell,
    fit_model,
    load_analysis,
    restrict_panel,
    term_stats,
    write_outputs,
)

sys.path.insert(0, str(PROJECT / "code" / "data"))
from config import IFLS4_FOLDER, IFLS5_FOLDER  # noqa: E402

TABLE = "table_s_palm_placebo"
HEAT = "tmean_7d_dev"
FE_POOLED = f"month + year + wave + {KECAMATAN_FE}"
N_PERM = 200
PERM_SEED = 20240610

UT1 = {"IFLS4": IFLS4_FOLDER / "b2_ut1.dta", "IFLS5": IFLS5_FOLDER / "b2_ut1.dta"}
HHID = {"IFLS4": "hhid07", "IFLS5": "hhid14"}

# ut07 crop code -> (label, WB Pink Sheet 2007-08 vs 2014-15 price change). No oil-palm code in IFLS.
CROPS = [
    (90, "Rice", "$-24\\%$"),
    (6, "Maize", "$-17\\%$"),
    (1, "Cassava", "n/a"),
    (3, "Groundnuts", "$-17\\%$"),
    (5, "Soybean", "$-21\\%$"),
    (12, "Coffee", "$-11\\%$"),
    (15, "Rubber", "$-40\\%$"),
]
MIN_GROWERS = 60


def load_crop_growers() -> pd.DataFrame:
    """HH-level crop dummies per wave from ut07a (most valuable) | ut07b (next valuable)."""
    frames = []
    for wave, path in UT1.items():
        d = pd.read_stata(path, convert_categoricals=False)
        d["hhid_s"] = d[HHID[wave]].astype(str)
        a = pd.to_numeric(d.get("ut07a"), errors="coerce")
        b = pd.to_numeric(d.get("ut07b"), errors="coerce")
        out = pd.DataFrame({"hhid_s": d["hhid_s"]})
        for code, _, _ in CROPS:
            out[f"grow_{code}"] = ((a == code) | (b == code)).astype(int)
        out = out.groupby("hhid_s").max().reset_index()
        out["wave"] = wave
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def add_crop_growers(df: pd.DataFrame) -> pd.DataFrame:
    """Merge crop dummies and carry the IFLS4 baseline forward by pidlink."""
    df = df.copy()
    df["hhid_s"] = df["hhid"].astype(str)
    df = df.merge(load_crop_growers(), on=["hhid_s", "wave"], how="left")
    for code, _, _ in CROPS:
        df[f"grow_{code}"] = df[f"grow_{code}"].fillna(0).astype(int)
        base = (
            df.loc[df["wave"] == "IFLS4", ["pidlink", f"grow_{code}"]]
            .rename(columns={f"grow_{code}": f"g{code}_ifls4"})
            .drop_duplicates("pidlink")
        )
        df = df.merge(base, on="pidlink", how="left")
        df[f"g{code}_ifls4"] = df[f"g{code}_ifls4"].fillna(0).astype(int)
    return df


def fit_triple(df: pd.DataFrame, treat: str) -> dict[str, float]:
    formula = f"cesd_z ~ {HEAT}*ifls5*{treat} + {CONTROLS} | {FE_POOLED}"
    required = ["cesd_z", HEAT, "ifls5", treat, "month", "year", "wave", KECAMATAN_FE, *CONTROL_COLUMNS]
    model = fit_model(df, formula, required)
    stats = term_stats(model, f"{HEAT}:ifls5:{treat}")
    stats["n_treated"] = int(df.drop_duplicates("pidlink")[treat].sum())
    return stats


def permutation_placebo(df: pd.DataFrame, true_b: float) -> dict[str, float]:
    """Shuffle palm-farmer status across households; return the permutation null summary."""
    import gc

    # Slim frame with only the columns the placebo regression needs (keeps memory flat
    # across N_PERM refits of a fragmenting 150-column frame).
    cols = ["cesd_z", HEAT, "ifls5", "pidlink", "month", "year", "wave", KECAMATAN_FE, *CONTROL_COLUMNS]
    slim = df[cols].dropna().copy()
    pid = df.drop_duplicates("pidlink")[["pidlink", "palm_farmer_hh_ifls4"]].reset_index(drop=True)
    pid_keys = pid["pidlink"].to_numpy()
    base = pid["palm_farmer_hh_ifls4"].to_numpy()
    rng = np.random.default_rng(PERM_SEED)
    term = f"{HEAT}:ifls5:palm_perm"
    null = []
    for i in range(N_PERM):
        mapping = dict(zip(pid_keys, rng.permutation(base)))
        slim["palm_perm"] = slim["pidlink"].map(mapping).astype("int8")
        model = pf.feols(f"cesd_z ~ {HEAT}*ifls5*palm_perm + {CONTROLS} | {FE_POOLED}", slim, vcov="iid")
        null.append(float(model.coef().get(term, np.nan)))
        del model
        if i % 20 == 0:
            gc.collect()
    null = np.array([x for x in null if not np.isnan(x)])
    return {
        "true_b": true_b,
        "mean": float(null.mean()),
        "p05": float(np.percentile(null, 5)),
        "p95": float(np.percentile(null, 95)),
        "ri_p": float((np.abs(null) >= abs(true_b)).mean()),
        "draws": int(len(null)),
    }


def main() -> None:
    df = add_crop_growers(restrict_panel(load_analysis()))
    df["palm_farmer_hh_ifls4"] = df["palm_farmer_hh_ifls4"].fillna(0).astype(int)

    palm = fit_triple(df, "palm_farmer_hh_ifls4")
    crop_rows = []
    for code, label, price in CROPS:
        treat = f"g{code}_ifls4"
        n_treated = int(df.drop_duplicates("pidlink")[treat].sum())
        if n_treated < MIN_GROWERS:
            continue
        crop_rows.append((label, price, fit_triple(df, treat)))

    perm = permutation_placebo(df, palm["b"])

    rows = [{"crop": "Palm (region x ag)", "price": "~-30%", **palm}]
    rows += [{"crop": label, "price": price, **stats} for label, price, stats in crop_rows]
    rows.append({"crop": "_permutation", **perm})

    body = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Crop (treatment) & Price change & Heat $\times$ farmer $\times$ IFLS5 & $N_{\text{growers}}$ \\",
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{Panel A. Crop placebo (actual cultivation); DV: CES-D z-score}} \\",
        r"\addlinespace[2pt]",
    ]
    pc, ps = cell(palm["b"], palm["se"], palm["p"])
    body.append(rf"Palm (region $\times$ ag, reference) & $\sim-30\%$ & {pc} & {palm['n_treated']:,} \\")
    body.append(rf" & & {ps} & \\")
    body.append(r"\addlinespace[2pt]")
    for label, price, stats in crop_rows:
        coef_cell, se_cell = cell(stats["b"], stats["se"], stats["p"])
        body.append(rf"{label} & {price} & {coef_cell} & {stats['n_treated']:,} \\")
        body.append(rf" & & {se_cell} & \\")
        body.append(r"\addlinespace[2pt]")
    body.extend(
        [
            r"\midrule",
            r"\multicolumn{4}{l}{\textit{Panel B. Fake-shock placebo (palm status permuted across households)}} \\",
            r"\addlinespace[2pt]",
            rf"True Heat $\times$ palm $\times$ IFLS5 & & ${perm['true_b']:+.3f}$ & \\",
            rf"Permutation null (mean) & & ${perm['mean']:+.3f}$ & \\",
            rf"Permutation null [5th, 95th pct] & & $[{perm['p05']:+.3f}, {perm['p95']:+.3f}]$ & \\",
            rf"Randomization-inference $p$ & & ${perm['ri_p']:.3f}$ & {perm['draws']} draws \\",
            r"\midrule",
            r"Demographic controls + Kec/Month/Year/Wave FE & \multicolumn{3}{c}{Yes} \\",
            r"Cluster: kabupaten (Panel A) & \multicolumn{3}{c}{Yes} \\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    write_outputs(TABLE, body, rows)


if __name__ == "__main__":
    main()
