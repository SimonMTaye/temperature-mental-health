"""Change in the heat slope for the four agriculture-by-palm-province groups.

Run from the repo root:

    uv run python code/analysis/updates/palm_four_group_table.py

Responds to Reviewer 3, comment 2(a). Writes
output/tables/table_palm_four_groups.tex.

The palm proxy is agricultural households in palm provinces. The table splits
that comparison into its two parts: one individual-FE regression interacts
heat x IFLS5 with baseline agricultural household, baseline palm province and
their product, and each column reports the implied IFLS5-minus-IFLS4 change in
the heat slope for one group (a sum of coefficients, with its
district-clustered standard error). Column (4) is the palm proxy group.
Sample, controls and fixed effects are the same as in Table B column 2.
"""

import numpy as np
import pandas as pd
from library.caching import run_regression_with_caching
from library.config import TABLE_OUTPUT
from library.render import RegressionSpec
from library.specs import CONTROLS_PANEL, FE_NO_WAVE, MAIN_TEMP_MEASURE, analysis_df
from library.table_builder import coefficient_rows, make_row
from scipy import stats

AGRICULTURE = "farmer_hh_ifls4"
PALM_PROVINCE = "palm_region_ifls4"
# (agricultural household, palm province) for columns (1)-(4)
GROUPS = [(0, 0), (1, 0), (0, 1), (1, 1)]

TABLE_TEMPLATE = r"""
\begin{tabular}{@{}lcccc}
\toprule
 & \multicolumn{4}{c}{CES-D z-score} \\
\cmidrule(lr){2-5}
 & \multicolumn{2}{c}{Other Provinces} & \multicolumn{2}{c}{Palm Provinces} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
 & \shortstack{Nonagricultural\\Households} & \shortstack{Agricultural\\Households} & \shortstack{Nonagricultural\\Households} & \shortstack{Agricultural\\Households} \\
\cmidrule(lr){2-2} \cmidrule(lr){3-3} \cmidrule(lr){4-4} \cmidrule(lr){5-5}
 & (1) & (2) & (3) & (4) \\
\midrule\addlinespace[2.5pt]
\multicolumn{5}{l}{7-day mean Temperature} \\
\midrule
{slope_change_rows}
\midrule\addlinespace[2.5pt]
Kecamatan & x & x & x & x \\
Month-Year & x & x & x & x \\
Individual & x & x & x & x \\
\midrule\addlinespace[2.5pt]
{observations_row}
\bottomrule
\end{tabular}
"""


def four_group_spec() -> RegressionSpec:
    df = analysis_df.copy()
    baseline = df.loc[~df["ifls5"]].set_index("pidlink")
    df[AGRICULTURE] = df["pidlink"].map(baseline["farmer_hh"]).astype(float)
    df[PALM_PROVINCE] = df["pidlink"].map(baseline["palm_region"]).astype(float)
    df = df.dropna(subset=["palm_farmer_hh_ifls4", AGRICULTURE, PALM_PROVINCE])
    heat, a, r = MAIN_TEMP_MEASURE, AGRICULTURE, PALM_PROVINCE
    formula = (
        f"cesd_z ~ {heat} + ifls5:{heat} + {a}:{heat} + {r}:{heat} + {a}:{r}:{heat}"
        f" + {a}:ifls5 + {r}:ifls5 + {a}:{r}:ifls5"
        f" + {a}:ifls5:{heat} + {r}:ifls5:{heat} + {a}:{r}:ifls5:{heat}"
        f" + {CONTROLS_PANEL} | {FE_NO_WAVE}+pidlink+ifls5"
    )
    return RegressionSpec(
        title="Palm Four Groups",
        formula=formula,
        df=df,
        tags=frozenset(["palm-shock", "mean-daily-temp", "panel"]),
        show_terms=None,
    )


def slope_change(model, agriculture: int, palm_province: int) -> tuple:
    """IFLS5-minus-IFLS4 heat slope for one group: sum of the ifls5 x heat terms."""
    switched_on = {AGRICULTURE: agriculture, PALM_PROVINCE: palm_province}
    weights = pd.Series(0.0, index=model.coef_table.index)
    for term in weights.index:
        parts = set(term.split(":"))
        if not {"ifls5", MAIN_TEMP_MEASURE} <= parts:
            continue
        others = parts - {"ifls5", MAIN_TEMP_MEASURE}
        weights[term] = float(all(switched_on[other] for other in others))
    coefficient = float(weights @ model.coef_table["b"])
    standard_error = float(np.sqrt(weights @ model.vcov @ weights))
    p_value = 2 * stats.t.sf(
        abs(coefficient / standard_error), model.stats["n_clusters"] - 1
    )
    return coefficient, standard_error, p_value


def estimation_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Rows the regression uses: complete cases, then singleton fixed effects dropped."""
    controls = CONTROLS_PANEL.replace(" ", "").split("+")
    used = df.dropna(subset=["cesd_z", MAIN_TEMP_MEASURE, *controls])
    while True:
        keep = pd.Series(True, index=used.index)
        for fixed_effect in ["pidlink", "gadm_fullcode", ["month", "year"]]:
            keep &= used.groupby(fixed_effect)["wave"].transform("size").gt(1)
        if keep.all():
            return used
        used = used.loc[keep]


def build_table() -> str:
    spec = four_group_spec()
    model = run_regression_with_caching(spec)
    coefs, ses = coefficient_rows(
        "Change in Temperature Slope",
        [slope_change(model, *group) for group in GROUPS],
    )
    used = estimation_sample(spec.df)
    assert len(used) == model.stats["N"]
    observations_row = make_row(
        "Observations in Group",
        [
            int((used[AGRICULTURE].eq(a) & used[PALM_PROVINCE].eq(r)).sum())
            for a, r in GROUPS
        ],
    )
    return TABLE_TEMPLATE.replace("{slope_change_rows}", f"{coefs}\n{ses}").replace(
        "{observations_row}", observations_row
    )


def main() -> None:
    TABLE_OUTPUT.mkdir(parents=True, exist_ok=True)
    (TABLE_OUTPUT / "table_palm_four_groups.tex").write_text(
        build_table(), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
