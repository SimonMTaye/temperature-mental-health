"""Formal test that Table B columns 5 and 6 have equal heat coefficients.

Run from the repo root:

    uv run python code/analysis/updates/one-off/table_b_cash_transfer_equality_test.py

Columns 5 and 6 of Table B are two separate IFLS5 regressions:

    cesd_z ~ G * post_subsidy * heat + controls | month-year + kecamatan

with G = urban vehicle owner WITHOUT a cash transfer (col 5) or WITH one
(col 6). The question is whether the two G:post_subsidy:heat coefficients
differ. A naive z-test, (b5 - b6) / sqrt(se5^2 + se6^2), is invalid: both
regressions use the same respondents, so the two estimates are correlated.
This script reports three valid versions of the test.

1. Stacked regression (the fixed-effects analogue of Stata's `suest`)
   Copy the estimation sample twice. Copy 0 carries the column 5 group dummy,
   copy 1 the column 6 dummy. Every regressor and every fixed effect is
   interacted with the copy indicator, so the stacked OLS is exactly the two
   original regressions side by side and reproduces their coefficients. Because
   each respondent appears in both copies and stays in the same kabupaten
   cluster, the cluster-robust covariance matrix picks up the covariance
   between the two estimates. A Wald test of b5 = b6 then uses
   Var(b5 - b6) = Var(b5) + Var(b6) - 2 Cov(b5, b6).

2. Pooled regression
   One regression with both mutually exclusive dummies, each fully interacted
   with post_subsidy and heat. Both groups are now compared with the same
   reference group (non-owners), unlike columns 5 and 6, where each group's
   comparison set contains the other group. Wald test on the two triple terms.

3. Four-way interaction
   Column 4 (urban vehicle owner x post x heat) fully interacted with
   cash_transfer_recipient. The four-way coefficient is the difference between
   recipients and non-recipients among owners, net of any recipient vs
   non-recipient difference among non-owners. This is a different contrast from
   (2): it also lets transfer recipients without vehicles respond differently.

Identification (all models): kecamatan and month-by-year fixed effects; the
heat coefficients come from week-to-week weather deviations around the
interview, within kecamatan and calendar month. Standard errors are clustered by
kabupaten, as in Table B; p-values use a t distribution with G - 1 degrees of
freedom, matching pyfixest's CRV1 default.
"""

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats

from library.specs import CONTROLS, FE_NO_WAVE, MAIN_TEMP_MEASURE, wave5_df

CLUSTER = "kabupaten_full_code"
POST = "post_subsidy"
NO_TRANSFER = "urban_vehicle_transfer_nonrecipient_ifls4"
TRANSFER = "urban_vehicle_transfer_recipient_ifls4"
URBAN_VEHICLE = "urban_vehicle_hh_ifls4"
RECIPIENT = "cash_transfer_recipient"
HEAT_MEASURES = {"7-day mean temperature": MAIN_TEMP_MEASURE, "7-day mean wet-bulb": "wetbulb_7d"}
CONTROL_COLUMNS = [term.strip() for term in CONTROLS.split("+")]


def main() -> None:
    for heat_label, heat in HEAT_MEASURES.items():
        separate = separate_regressions(heat)
        results = [
            naive_test(separate),
            stacked_test(heat),
            pooled_test(heat),
            four_way_test(heat),
        ]
        print(f"\n=== {heat_label} ({heat}) ===")
        print(f"Column 5 (no transfer): b = {separate[0][0]:.4f}  se = {separate[0][1]:.4f}")
        print(f"Column 6 (transfer):    b = {separate[1][0]:.4f}  se = {separate[1][1]:.4f}")
        print(pd.DataFrame(results).to_string(index=False, float_format="{:.4f}".format))


def triple_diff_formula(group: str, heat: str) -> str:
    return f"cesd_z ~ {group} * {POST} * {heat} + {CONTROLS} | {FE_NO_WAVE}"


def fit(formula: str, data: pd.DataFrame):
    return pf.feols(formula, data=data, vcov={"CRV1": CLUSTER})


def separate_regressions(heat: str) -> list[tuple[float, float]]:
    """Reproduce Table B columns 5 and 6: (coefficient, se) of G:post:heat."""
    out = []
    for group in [NO_TRANSFER, TRANSFER]:
        model = fit(triple_diff_formula(group, heat), wave5_df)
        term = f"{group}:{POST}:{heat}"
        out.append((model.coef()[term], model.se()[term]))
    return out


def naive_test(separate: list[tuple[float, float]]) -> dict:
    """Shown only for contrast: ignores the covariance between the two estimates."""
    (b5, se5), (b6, se6) = separate
    se_diff = np.sqrt(se5**2 + se6**2)
    z = (b6 - b5) / se_diff
    return result_row("Naive z (invalid)", b6 - b5, se_diff, 2 * stats.norm.sf(abs(z)), np.nan)


def stacked_test(heat: str) -> dict:
    stacked = stack_columns_5_and_6(heat)
    regressors = [f"{name}_eq{k}" for k in (0, 1) for name in stacked_regressor_names()]
    formula = (
        f"cesd_z ~ {' + '.join(regressors)}"
        " | month^year^equation + gadm_fullcode^equation"
    )
    model = fit(formula, stacked)
    b5, b6 = model.coef()["group_post_heat_eq0"], model.coef()["group_post_heat_eq1"]
    vcov = pd.DataFrame(model._vcov, index=model.coef().index, columns=model.coef().index)
    covariance = vcov.loc["group_post_heat_eq0", "group_post_heat_eq1"]
    se_diff = np.sqrt(
        vcov.loc["group_post_heat_eq0", "group_post_heat_eq0"]
        + vcov.loc["group_post_heat_eq1", "group_post_heat_eq1"]
        - 2 * covariance
    )
    correlation = covariance / np.sqrt(
        vcov.loc["group_post_heat_eq0", "group_post_heat_eq0"]
        * vcov.loc["group_post_heat_eq1", "group_post_heat_eq1"]
    )
    row = cluster_t_row("Stacked (suest)", b6 - b5, se_diff, model)
    row["corr(b5, b6)"] = correlation
    print(
        f"[stacked check] eq0 b = {b5:.4f} se = {model.se()['group_post_heat_eq0']:.4f}; "
        f"eq1 b = {b6:.4f} se = {model.se()['group_post_heat_eq1']:.4f}"
    )
    return row


def stacked_regressor_names() -> list[str]:
    return ["group", "post", "heat", "group_post", "group_heat", "post_heat", "group_post_heat"] + CONTROL_COLUMNS


def stack_columns_5_and_6(heat: str) -> pd.DataFrame:
    """Two copies of the IFLS5 sample, one per column, with copy-specific regressors.

    Each regressor x becomes x_eq0 (= x in copy 0, 0 in copy 1) and x_eq1
    (= 0 in copy 0, x in copy 1). Missing values stay missing in their own copy
    so each copy drops exactly the rows its original regression drops.
    """
    copies = []
    for equation, group in enumerate([NO_TRANSFER, TRANSFER]):
        base = wave5_df.copy()
        values = pd.DataFrame(
            {
                "group": base[group],
                "post": base[POST],
                "heat": base[heat],
                **{control: base[control] for control in CONTROL_COLUMNS},
            }
        ).astype("float64")
        values["group_post"] = values["group"] * values["post"]
        values["group_heat"] = values["group"] * values["heat"]
        values["post_heat"] = values["post"] * values["heat"]
        values["group_post_heat"] = values["group"] * values["post"] * values["heat"]

        copy = base[["cesd_z", "month", "year", "gadm_fullcode", CLUSTER]].copy()
        copy["equation"] = equation
        for name in stacked_regressor_names():
            copy[f"{name}_eq{equation}"] = values[name]
            copy[f"{name}_eq{1 - equation}"] = 0.0
        copies.append(copy)
    return pd.concat(copies, ignore_index=True)


def pooled_test(heat: str) -> dict:
    formula = (
        f"cesd_z ~ {NO_TRANSFER} * {POST} * {heat} + {TRANSFER} * {POST} * {heat}"
        f" + {CONTROLS} | {FE_NO_WAVE}"
    )
    model = fit(formula, wave5_df)
    no_transfer_term = f"{NO_TRANSFER}:{POST}:{heat}"
    transfer_term = f"{TRANSFER}:{POST}:{heat}"
    coefficients = model.coef()
    vcov = pd.DataFrame(model._vcov, index=coefficients.index, columns=coefficients.index)
    difference = coefficients[transfer_term] - coefficients[no_transfer_term]
    se_diff = np.sqrt(
        vcov.loc[transfer_term, transfer_term]
        + vcov.loc[no_transfer_term, no_transfer_term]
        - 2 * vcov.loc[transfer_term, no_transfer_term]
    )
    print(
        f"[pooled] no transfer b = {coefficients[no_transfer_term]:.4f}, "
        f"transfer b = {coefficients[transfer_term]:.4f}"
    )
    return cluster_t_row("Pooled", difference, se_diff, model)


def four_way_test(heat: str) -> dict:
    formula = f"cesd_z ~ {URBAN_VEHICLE} * {RECIPIENT} * {POST} * {heat} + {CONTROLS} | {FE_NO_WAVE}"
    model = fit(formula, wave5_df)
    term = f"{URBAN_VEHICLE}:{RECIPIENT}:{POST}:{heat}"
    return cluster_t_row("Four-way interaction", model.coef()[term], model.se()[term], model)


def cluster_t_row(label: str, difference: float, se_diff: float, model) -> dict:
    n_clusters = model._data[CLUSTER].nunique()
    t_stat = difference / se_diff
    p_value = 2 * stats.t.sf(abs(t_stat), df=n_clusters - 1)
    return result_row(label, difference, se_diff, p_value, model._N)


def result_row(label: str, difference: float, se_diff: float, p_value: float, n_obs) -> dict:
    return {
        "test": label,
        "b6 - b5": difference,
        "se": se_diff,
        "p": p_value,
        "N": n_obs,
        "corr(b5, b6)": np.nan,
    }


if __name__ == "__main__":
    main()
