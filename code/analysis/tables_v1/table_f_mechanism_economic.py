"""Table F: economic mechanism specifications."""

from __future__ import annotations

from _lettered_common import CONTROLS, FE_IFLS5, FE_POOLED, cell, fit_model, load_analysis, multi_stats, write_outputs

TABLE = "table_f_mechanism_economic"


def fit_terms(data, outcome: str, rhs: str, terms: dict[str, str], fe: str, required: list[str]) -> dict[str, object]:
    formula = f"{outcome} ~ {rhs} + {CONTROLS} | {fe}"
    model = fit_model(data, formula, [outcome, *required, "month", "year", "kecamatan_fe_code", "age", "female", "edu_yrs", "married", "widowed"])
    return {"dv": outcome, **multi_stats(model, terms)}


def main() -> None:
    df = load_analysis()
    df["palm_shock"] = df["palm_farmer_hh"] * df["ifls5"]
    df["coal_shock"] = df["coal_worker_hh"] * df["ifls5"]
    df["fuel_shock"] = df["post_subsidy"] * df["urban_vehicle_hh_ifls4"]
    sub5 = df[df["wave"] == "IFLS5"].copy()

    rows = [
        {"panel": "A", **fit_terms(df, "labor_real_w", "palm_shock + palm_farmer_hh", {"shock": "palm_shock"}, FE_POOLED, ["palm_shock", "palm_farmer_hh", "wave"])},
        {"panel": "A", **fit_terms(df, "nonlabor_real_w", "palm_shock + palm_farmer_hh", {"shock": "palm_shock"}, FE_POOLED, ["palm_shock", "palm_farmer_hh", "wave"])},
        {"panel": "A", **fit_terms(df, "job_loss_1_yr", "palm_shock + palm_farmer_hh", {"shock": "palm_shock"}, FE_POOLED, ["palm_shock", "palm_farmer_hh", "wave"])},
        {"panel": "B", **fit_terms(df, "labor_real_w", "coal_shock + coal_worker_hh", {"shock": "coal_shock"}, FE_POOLED, ["coal_shock", "coal_worker_hh", "wave"])},
        {"panel": "B", **fit_terms(df, "nonlabor_real_w", "coal_shock + coal_worker_hh", {"shock": "coal_shock"}, FE_POOLED, ["coal_shock", "coal_worker_hh", "wave"])},
        {"panel": "B", **fit_terms(df, "job_loss_1_yr", "coal_shock + coal_worker_hh", {"shock": "coal_shock"}, FE_POOLED, ["coal_shock", "coal_worker_hh", "wave"])},
        {"panel": "C", **fit_terms(df, "labor_real_w", "job_loss_1_yr", {"shock": "job_loss_1_yr"}, FE_POOLED, ["job_loss_1_yr", "wave"])},
        {"panel": "C", **fit_terms(df, "nonlabor_real_w", "job_loss_1_yr", {"shock": "job_loss_1_yr"}, FE_POOLED, ["job_loss_1_yr", "wave"])},
        {"panel": "D", **fit_terms(sub5, "transport_real_w", "fuel_shock + urban_vehicle_hh_ifls4", {"shock": "fuel_shock"}, FE_IFLS5, ["fuel_shock", "urban_vehicle_hh_ifls4"])},
        {"panel": "D", **fit_terms(sub5, "transport_share_ihs", "fuel_shock + urban_vehicle_hh_ifls4", {"shock": "fuel_shock"}, FE_IFLS5, ["fuel_shock", "urban_vehicle_hh_ifls4"])},
    ]

    body = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Panel / outcome & Key coefficient & SE & N \\",
        r"\midrule",
    ]
    for row in rows:
        c, s = cell(row["shock_b"], row["shock_se"], row["shock_p"])
        body.append(rf"{row['panel']}. {row['dv']} & {c} & {s} & {int(row['n']):,} \\")
    body.extend(
        [
            r"\midrule",
            r"Demographic controls & \multicolumn{3}{c}{Yes} \\",
            r"Kecamatan + Month + Year FE & \multicolumn{3}{c}{Yes} \\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    write_outputs(TABLE, body, rows)


if __name__ == "__main__":
    main()
