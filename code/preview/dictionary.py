"""Human-readable labels for variables used by preview and table output.

The labels here are centered on variables used in ``code/analysis/tables``.
They are intentionally plain text so they work in HTML preview output and can
also be escaped later for LaTeX if needed.
"""

from collections.abc import Iterable

VARIABLE_LABELS: dict[str, str] = {
    # Mental-health outcomes
    "cesd_z": "CES-D z-score",
    "cesd_raw": "CES-D raw score",
    "depressed": "Depressed",
    "somatic_z": "Somatic / activity-related CES-D factor",
    "depraffect_z": "Depressed-affect CES-D factor",
    "posaffect_z": "Positive-affect CES-D factor",
    # Temperature measures
    "heat_c_dev": "Mean temperature deviation",
    "tmean_c": "Daily Mean Temp",
    "tmean_c_dev": "Daily Mean Temp(deviation from avg)",
    "tmax_c": "Maximum temperature (deg C)",
    "tmax_c_dev": "Maximum temperature deviation",
    "tmin_c": "Minimum temperature (deg C)",
    "tmin_c_dev": "Minimum temperature deviation",
    "cdd_tmax30": "CDD: maximum temperature above 30 deg C",
    "cdd_tmax32": "CDD: maximum temperature above 32 deg C",
    "cdd_tmin23": "CDD: minimum temperature above 23 deg C",
    "cdd_tmin24": "CDD: minimum temperature above 24 deg C",
    "tmean_7d_dev": "7-day mean temperature deviation",
    "hot30_7d_dev": "7-day hot-day count deviation",
    "heatwave_7d_dev": "7-day heatwave-day count deviation",
    "wetbulb_c_dev": "Wet-bulb temperature deviation",
    "wetbulb_7d_dev": "7-day wet-bulb temperature deviation",
    "tmean_c_hour": "Survey-hour temperature (deg C)",
    "heat_hr_dev": "Survey-hour temperature deviation",
    # Stressors and baseline groups
    "job_loss_1_yr": "Job loss (1 yr)",
    "job_loss_90d": "Job loss within 3 months",
    "job_loss_180d": "Job loss within 6 months",
    "job_loss_270d": "Job loss within 9 months",
    "job_loss_365d": "Job loss within 12 months",
    "job_loss_540d": "Job loss within 18 months",
    "job_loss_730d": "Job loss within 24 months",
    "job_loss_1095d": "Job loss within 36 months",
    "job_loss_1825d": "Job loss within 60 months",
    "involuntary_loss_1_yr": "Involuntary job loss within 12 months",
    "family_loss_1_yr": "Family-related job loss within 12 months",
    "palm_farmer_individual": "Palm-farmer individual",
    "palm_farmer_individual_ifls4": "IFLS4 palm-farmer individual",
    "palm_farmer_hh": "Palm-farmer household",
    "palm_farmer_hh_ifls4": "Palm Farmer",
    "coal_worker_individual": "Coal-worker individual",
    "coal_worker_individual_ifls4": "IFLS4 coal-worker individual",
    "coal_worker_hh": "Coal-worker household",
    "coal_worker_hh_ifls4": "IFLS4 coal-worker household",
    "urban_vehicle_hh": "Urban vehicle-owning household",
    "urban_vehicle_hh_ifls4": "IFLS4 urban vehicle-owning household",
    "ifls5": "Wave: IFLS5",
    "post_subsidy": "Post-subsidy-cut interview",
    "palm_shock": "Palm-price shock exposure",
    "coal_shock": "Coal-price shock exposure",
    "fuel_shock": "Fuel-subsidy-cut exposure",
    # Economic and sleep mechanisms
    "labor_real_w": "Real labor income",
    "nonlabor_real_w": "Real nonlabor income",
    "transport_real_w": "Real transport spending",
    "transport_spending_mo": "Monthly transport spending",
    "transport_share": "Transport budget share",
    "transport_share_ihs": "IHS transport budget share",
    "transport_share_q5": "Transport budget-share quintile",
    "high_transport_share": "Top transport budget-share quintile",
    "sleep_dur_h": "Sleep duration (hours)",
    # Demographic controls
    "age": "Age",
    "female": "Female",
    "edu_yrs": "Years of education",
    "married": "Married",
    "widowed": "Widowed",
    # Fixed effects, clusters, and identifiers that may appear in previews
    "month": "Month",
    "year": "Year ",
    "wave": "Wave",
    "day_id": "Calendar-day",
    "gadm_fullcode": "Kecamatan fixed effects",
    "pidlink": "Individual",
    "hhid": "Household identifier",
}

HEAT_TERMS = [
    "heat_c_dev",
    "tmean_c_dev",
    "tmax_c_dev",
    "tmin_c_dev",
    "cdd_tmax30",
    "cdd_tmax32",
    "cdd_tmin23",
    "cdd_tmin24",
    "tmean_7d_dev",
    "wetbulb_c_dev",
    "wetbulb_7d_dev",
    "hot30_7d_dev",
    "heatwave_7d_dev",
    "heat_hr_dev",
]

STRESSOR_TERMS = [
    "job_loss_1_yr",
    "job_loss_90d",
    "job_loss_180d",
    "job_loss_270d",
    "job_loss_365d",
    "job_loss_540d",
    "job_loss_730d",
    "job_loss_1095d",
    "job_loss_1825d",
    "involuntary_loss_1_yr",
    "family_loss_1_yr",
    "palm_farmer_hh_ifls4",
    "coal_worker_hh_ifls4",
]


def label_term(term: str) -> str:
    """Return a human-readable label for a variable or interaction term."""
    if term in VARIABLE_LABELS:
        return VARIABLE_LABELS[term]
    if ":" not in term:
        return term
    return " x ".join(VARIABLE_LABELS.get(part, part) for part in term.split(":"))


def labels_for(terms: Iterable[str]) -> dict[str, str]:
    """Build a maketables-compatible labels dictionary for selected terms."""
    return {term: label_term(term) for term in terms}


INTERACTION_LABELS: dict[str, str] = {
    **{
        f"{heat}:{stressor}": label_term(f"{heat}:{stressor}")
        for heat in HEAT_TERMS
        for stressor in STRESSOR_TERMS
    },
    **{
        f"{heat}:ifls5:palm_farmer_hh_ifls4": label_term(
            f"{heat}:ifls5:palm_farmer_hh_ifls4"
        )
        for heat in HEAT_TERMS
    },
    **{
        f"{heat}:ifls5:coal_worker_hh_ifls4": label_term(
            f"{heat}:ifls5:coal_worker_hh_ifls4"
        )
        for heat in HEAT_TERMS
    },
    **{
        f"{heat}:post_subsidy:urban_vehicle_hh_ifls4": label_term(
            f"{heat}:post_subsidy:urban_vehicle_hh_ifls4"
        )
        for heat in HEAT_TERMS
    },
}

TABLE_LABELS: dict[str, str] = {
    **VARIABLE_LABELS,
    **INTERACTION_LABELS,
}
