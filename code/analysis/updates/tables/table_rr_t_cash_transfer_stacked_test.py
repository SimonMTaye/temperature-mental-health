"""Test whether Table B columns 5 and 6 differ, using a stacked regression.

Run from the repo root:

    uv run python code/analysis/updates/tables/table_rr_t_cash_transfer_stacked_test.py

Writes output/tables/table_rr_t_cash_transfer_stacked_test.tex.

Table B columns 5 and 6 are two separate IFLS5 regressions,

    cesd_z ~ G * post_subsidy * heat + controls | month*year + kecamatan,

with G = baseline urban vehicle owner who does NOT receive a cash transfer
(column 5) or who DOES (column 6). The question is whether the two
G:post_subsidy:heat coefficients are equal. A naive z-test that adds the two
squared standard errors is invalid: both regressions use the same respondents,
so their estimates are correlated.

Stacked regression (the fixed-effects version of Stata's `suest`):
    1. Make two copies of the IFLS5 sample. Copy 0 carries the column 5 group
       dummy, copy 1 the column 6 group dummy.
    2. Give each copy its own version of every regressor (x_eq0 = x in copy 0
       and 0 in copy 1; x_eq1 the reverse) and interact every fixed effect with
       the copy indicator. Nothing is shared across copies, so OLS on the
       stacked data reproduces columns 5 and 6 exactly.
    3. Cluster by kabupaten. Each respondent appears in both copies within the
       same cluster, so the cluster-robust covariance matrix also estimates
       Cov(b5, b6).
    4. Wald test of b6 = b5 using
       Var(b6 - b5) = Var(b5) + Var(b6) - 2 Cov(b5, b6),
       with a t distribution on G - 1 degrees of freedom (pyfixest's CRV1
       default, as for every other coefficient in the table).

Identification is Table B's: kecamatan and month-by-year fixed effects, so heat
effects come from week-to-week weather deviations around local and calendar
norms, assumed as good as random conditional on those fixed effects and the
demographic controls. Panels repeat the test for 7-day mean temperature and
7-day mean wet-bulb.
"""

import numpy as np
import pandas as pd
import pyfixest as pf
from analysis.tables_v2.table_b_temperature_and_shock_effects import (
    KABUPATEN_CLUSTER,
    WETBULB_MEASURE,
)
from scipy import stats

from library.config import TABLE_OUTPUT
from library.specs import CONTROLS, MAIN_TEMP_MEASURE, wave5_df
from library.table_builder import make_row, stars

TABLE_FILE = TABLE_OUTPUT / "table_rr_t_cash_transfer_stacked_test.tex"
POST = "post_subsidy"
GROUPS_BY_EQUATION = [
    "urban_vehicle_transfer_nonrecipient_ifls4",  # Table B column 5
    "urban_vehicle_transfer_recipient_ifls4",  # Table B column 6
]
CONTROL_COLUMNS = [term.strip() for term in CONTROLS.split("+")]
TRIPLE_DIFF_TERMS = [
    "group",
    "post",
    "heat",
    "group_post",
    "group_heat",
    "post_heat",
    "group_post_heat",
]

TABLE_TEMPLATE = r"""
\begin{tabular}{@{}lccc}
\toprule
 & \multicolumn{3}{c}{CES-D z-score} \\
\cmidrule(lr){2-4}
 & \shortstack{Urban Vehicle Owners\\No Cash Transfer} & \shortstack{Urban Vehicle Owners\\Cash Transfer} & \shortstack{Difference\\(2) $-$ (1)} \\
\cmidrule(lr){2-2} \cmidrule(lr){3-3} \cmidrule(lr){4-4}
 & (1) & (2) & (3) \\
\midrule\addlinespace[2.5pt]
{temperature_panel}
\midrule\addlinespace[2.5pt]
{wetbulb_panel}
\midrule\addlinespace[2.5pt]
Kecamatan & x & x & \\
Month-Year & x & x & \\
\midrule\addlinespace[2.5pt]
{observations_row}
\bottomrule
\end{tabular}
"""


def main() -> None:
    temperature_model = fit_stacked_regression(MAIN_TEMP_MEASURE)
    wetbulb_model = fit_stacked_regression(WETBULB_MEASURE)
    table = (
        TABLE_TEMPLATE.replace(
            "{temperature_panel}",
            panel_rows("7-day mean Temperature", temperature_model),
        )
        .replace("{wetbulb_panel}", panel_rows("7-day mean Wet-bulb", wetbulb_model))
        .replace("{observations_row}", observations_row(temperature_model))
    )
    TABLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    TABLE_FILE.write_text(table, encoding="utf-8")
    print_summary({"tmean_7d": temperature_model, "wetbulb_7d": wetbulb_model})


# ---------------------------------------------------------------------------
# Stacked regression
# ---------------------------------------------------------------------------


def numeric_controls() -> list[str]:
    return [c for c in CONTROL_COLUMNS if pd.api.types.is_numeric_dtype(wave5_df[c])]


def categorical_controls() -> list[str]:
    """Categorical controls enter as copy-specific fixed effects (same fit as dummies)."""
    return [c for c in CONTROL_COLUMNS if c not in numeric_controls()]


def stacked_regressors() -> list[str]:
    return TRIPLE_DIFF_TERMS + numeric_controls()


def stack_columns_5_and_6(heat: str) -> pd.DataFrame:
    """Two copies of the IFLS5 sample, one per Table B column, with copy-specific regressors.

    Missing values stay missing in their own copy, so each copy drops exactly
    the rows its original regression drops.
    """
    copies = []
    for equation, group in enumerate(GROUPS_BY_EQUATION):
        values = pd.DataFrame(
            {
                "group": wave5_df[group],
                "post": wave5_df[POST],
                "heat": wave5_df[heat],
                **{control: wave5_df[control] for control in numeric_controls()},
            }
        ).astype("float64")
        values["group_post"] = values["group"] * values["post"]
        values["group_heat"] = values["group"] * values["heat"]
        values["post_heat"] = values["post"] * values["heat"]
        values["group_post_heat"] = values["group"] * values["post"] * values["heat"]

        copy = wave5_df[
            ["cesd_z", "month", "year", "gadm_fullcode", KABUPATEN_CLUSTER]
            + categorical_controls()
        ].copy()
        copy["equation"] = equation
        for name in stacked_regressors():
            copy[f"{name}_eq{equation}"] = values[name]
            copy[f"{name}_eq{1 - equation}"] = 0.0
        copies.append(copy)
    return pd.concat(copies, ignore_index=True)


def fit_stacked_regression(heat: str):
    regressors = [f"{name}_eq{k}" for k in (0, 1) for name in stacked_regressors()]
    # Table B's fixed effects (kecamatan, month-by-year), each interacted with
    # the copy indicator so the two equations share nothing.
    fixed_effects = ["month^year^equation", "gadm_fullcode^equation"] + [
        f"{control}^equation" for control in categorical_controls()
    ]
    formula = f"cesd_z ~ {' + '.join(regressors)} | {' + '.join(fixed_effects)}"
    return pf.feols(
        formula,
        data=stack_columns_5_and_6(heat),
        vcov={"CRV1": KABUPATEN_CLUSTER},
    )


# ---------------------------------------------------------------------------
# Test and table rows
# ---------------------------------------------------------------------------


def equality_test(model) -> dict[str, float]:
    """Coefficients, SEs and p-values for each column, plus the Wald test of b6 = b5."""
    coefficients = model.coef()
    vcov = pd.DataFrame(model._vcov, index=coefficients.index, columns=coefficients.index)
    b5_term, b6_term = "group_post_heat_eq0", "group_post_heat_eq1"
    covariance = vcov.loc[b5_term, b6_term]
    difference = coefficients[b6_term] - coefficients[b5_term]
    se_difference = np.sqrt(
        vcov.loc[b5_term, b5_term] + vcov.loc[b6_term, b6_term] - 2 * covariance
    )
    n_clusters = model._data[KABUPATEN_CLUSTER].nunique()
    p_difference = 2 * stats.t.sf(abs(difference / se_difference), df=n_clusters - 1)
    return {
        "b5": coefficients[b5_term],
        "se5": model.se()[b5_term],
        "p5": model.pvalue()[b5_term],
        "b6": coefficients[b6_term],
        "se6": model.se()[b6_term],
        "p6": model.pvalue()[b6_term],
        "difference": difference,
        "se_difference": se_difference,
        "p_difference": p_difference,
        "correlation": covariance
        / np.sqrt(vcov.loc[b5_term, b5_term] * vcov.loc[b6_term, b6_term]),
    }


def panel_rows(label: str, model) -> str:
    test = equality_test(model)
    return "\n".join(
        [
            rf"\multicolumn{{4}}{{l}}{{\text{{{label}}}}} \\",
            r"\midrule",
            make_row(
                "Differential Impact of Heat",
                [
                    f"{test['b5']:.3f}{stars(test['p5'])}",
                    f"{test['b6']:.3f}{stars(test['p6'])}",
                    f"{test['difference']:.3f}{stars(test['p_difference'])}",
                ],
            ),
            make_row(
                "",
                [
                    f"({test['se5']:.3f})",
                    f"({test['se6']:.3f})",
                    f"({test['se_difference']:.3f})",
                ],
            ),
            make_row(r"p-value, $H_0$: (1) = (2)", ["", "", f"[{test['p_difference']:.3f}]"]),
            make_row("Correlation of (1) and (2)", ["", "", f"{test['correlation']:.3f}"]),
        ]
    )


def observations_row(model) -> str:
    """Observations per equation, i.e. per original Table B column."""
    per_equation = model._data["equation"].value_counts().sort_index()
    return make_row("Observations", [f"{n:,}" for n in per_equation] + [""])


def print_summary(models: dict) -> None:
    rows = [{"heat": heat, **equality_test(model)} for heat, model in models.items()]
    with pd.option_context("display.width", 200, "display.precision", 4):
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
