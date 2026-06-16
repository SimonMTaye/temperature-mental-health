"""Table T: are the crop placebos independent of palm, or just palm overlap?

Diagnostic for the crop placebo (Table S). Because "palm farmer" = agricultural HH x
palm province (IFLS has no oil-palm crop code), a household that grows maize/coffee/
rubber in a palm province is ALSO flagged a palm farmer. This table checks whether any
crop carries an effect independent of palm.

  Panel A -- crop-grower groups and their overlap with the palm flag.
  Panel B -- heat x grower x IFLS5, split by palm vs non-palm province.
  Panel C -- horse race: crop triple and palm triple in the same model. If the crop
             coefficients collapse while palm survives, the apparent crop effects are
             palm overlap, not independent crop effects.

Crop growers are actual cultivation (b2_ut1 ut07a/ut07b) read via code/data/config.py.
Heat = tmean_7d_dev; kabupaten-clustered SE; IFLS4-baseline panel.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from _lettered_common import (
    CONTROL_COLUMNS,
    CONTROLS,
    KECAMATAN_FE,
    PROJECT,
    load_analysis,
    restrict_panel,
    stars,
    term_stats,
    fit_model,
    write_outputs,
)

sys.path.insert(0, str(PROJECT / "code" / "data"))
from config import IFLS4_FOLDER, IFLS5_FOLDER  # noqa: E402

TABLE = "table_t_crop_palm_overlap"
HEAT = "tmean_7d_dev"
PALM = "palm_farmer_hh_ifls4"
FE_POOLED = f"month + year + wave + {KECAMATAN_FE}"
MIN_TREATED = 30

# 2-digit BPS palm-producing provinces (matches the pipeline's palm_region definition).
PALM_PROVS = {11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 61, 62, 63, 64}
UT1 = {"IFLS4": IFLS4_FOLDER / "b2_ut1.dta", "IFLS5": IFLS5_FOLDER / "b2_ut1.dta"}
HHID = {"IFLS4": "hhid07", "IFLS5": "hhid14"}
CROPS = [(90, "Rice"), (6, "Maize"), (5, "Soybean"), (12, "Coffee"), (15, "Rubber")]


def load_growers() -> pd.DataFrame:
    frames = []
    for wave, path in UT1.items():
        d = pd.read_stata(path, convert_categoricals=False)
        d["hhid_s"] = d[HHID[wave]].astype(str)
        a = pd.to_numeric(d.get("ut07a"), errors="coerce")
        b = pd.to_numeric(d.get("ut07b"), errors="coerce")
        out = pd.DataFrame({"hhid_s": d["hhid_s"]})
        for code, _ in CROPS:
            out[f"grow_{code}"] = ((a == code) | (b == code)).astype(int)
        out = out.groupby("hhid_s").max().reset_index()
        out["wave"] = wave
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hhid_s"] = df["hhid"].astype(str)
    df[PALM] = df[PALM].fillna(0).astype(int)
    df["palm_prov"] = df["province_code"].isin(PALM_PROVS).astype(int)
    df = df.merge(load_growers(), on=["hhid_s", "wave"], how="left")
    for code, _ in CROPS:
        df[f"grow_{code}"] = df[f"grow_{code}"].fillna(0).astype(int)
        base = (
            df.loc[df["wave"] == "IFLS4", ["pidlink", f"grow_{code}"]]
            .rename(columns={f"grow_{code}": f"g{code}"}).drop_duplicates("pidlink")
        )
        df = df.merge(base, on="pidlink", how="left")
        df[f"g{code}"] = df[f"g{code}"].fillna(0).astype(int)
    return df


def n_treated(df: pd.DataFrame, treat: str, mask: pd.Series | None = None) -> int:
    pids = df.drop_duplicates("pidlink")
    if mask is not None:
        pids = pids[mask.loc[pids.index]]
    return int(pids[treat].sum())


def triple(df: pd.DataFrame, treat: str) -> dict | None:
    if n_treated(df, treat) < MIN_TREATED:
        return None
    formula = f"cesd_z ~ {HEAT}*ifls5*{treat} + {CONTROLS} | {FE_POOLED}"
    required = ["cesd_z", HEAT, "ifls5", treat, "month", "year", "wave", KECAMATAN_FE, *CONTROL_COLUMNS]
    return term_stats(fit_model(df, formula, required), f"{HEAT}:ifls5:{treat}")


def horse_race(df: pd.DataFrame, treat: str) -> tuple[dict, dict]:
    formula = f"cesd_z ~ {HEAT}*ifls5*{treat} + {HEAT}*ifls5*{PALM} + {CONTROLS} | {FE_POOLED}"
    required = ["cesd_z", HEAT, "ifls5", treat, PALM, "month", "year", "wave", KECAMATAN_FE, *CONTROL_COLUMNS]
    model = fit_model(df, formula, required)
    return term_stats(model, f"{HEAT}:ifls5:{treat}"), term_stats(model, f"{HEAT}:ifls5:{PALM}")


def inline(stats: dict | None) -> str:
    if stats is None or pd.isna(stats.get("b")):
        return r"\textemdash"
    return f"${stats['b']:+.3f}{stars(stats['p'])}$ ({stats['se']:.3f})"


def main() -> None:
    df = prepare(restrict_panel(load_analysis()))
    rows: list[dict] = []

    body = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\multicolumn{4}{l}{\textit{Panel A. Crop-grower groups and palm overlap (IFLS4 baseline)}} \\",
        r"Crop & $N$ growers & \% in palm prov. & \% also palm farmer \\",
        r"\midrule",
    ]
    for code, name in CROPS:
        g = f"g{code}"
        pids = df.drop_duplicates("pidlink")
        growers = pids[pids[g] == 1]
        n = len(growers)
        pct_prov = 100 * growers["palm_prov"].mean() if n else np.nan
        pct_palm = 100 * growers[PALM].mean() if n else np.nan
        rows.append({"panel": "A", "crop": name, "n": n, "pct_palm_prov": pct_prov, "pct_palm_farmer": pct_palm})
        body.append(rf"{name} & {n:,} & {pct_prov:.0f}\% & {pct_palm:.0f}\% \\")

    body += [
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{Panel B. Heat $\times$ grower $\times$ IFLS5, by province type}} \\",
        r"Crop & All provinces & Palm provinces & Non-palm \\",
        r"\midrule",
    ]
    for code, name in CROPS:
        g = f"g{code}"
        all_s = triple(df, g)
        palm_s = triple(df[df["palm_prov"] == 1], g)
        nonp_s = triple(df[df["palm_prov"] == 0], g)
        rows.append({"panel": "B", "crop": name, "all": all_s, "palm_prov": palm_s, "non_palm": nonp_s})
        body.append(rf"{name} & {inline(all_s)} & {inline(palm_s)} & {inline(nonp_s)} \\")

    body += [
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{Panel C. Horse race: crop triple and palm triple jointly}} \\",
        r"Crop in model & Crop $\times$ IFLS5 & Palm $\times$ IFLS5 & \\",
        r"\midrule",
    ]
    for code, name in CROPS:
        crop_s, palm_s = horse_race(df, f"g{code}")
        rows.append({"panel": "C", "crop": name, "crop_coef": crop_s, "palm_coef": palm_s})
        body.append(rf"{name} & {inline(crop_s)} & {inline(palm_s)} & \\")

    body += [
        r"\midrule",
        r"\multicolumn{4}{l}{\footnotesize Heat $=$ 7-day mean dev.; Kec + Month + Year + Wave FE; kabupaten-clustered SE.} \\",
        r"\multicolumn{4}{l}{\footnotesize ``Palm farmer'' $=$ agricultural HH $\times$ palm province (IFLS has no oil-palm crop code).} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write_outputs(TABLE, body, rows)


if __name__ == "__main__":
    main()
