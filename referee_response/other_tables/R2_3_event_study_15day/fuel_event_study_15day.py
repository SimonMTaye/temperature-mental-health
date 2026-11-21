"""Event-time figure for the fuel subsidy cut, 15-day windows (response R2.3 / R3.3b).

One regression on the IFLS5 fuel sample of Table B column 4. The post indicator is
replaced by dummies for 15-day windows counted from November 18, 2014. The window
-15 to -1 is the reference. Plotted: vehicle x window x heat, i.e. the urban-vehicle
minus comparison heat slope in each window relative to the 15 days before the cut.

Interviews 61 or more days before the cut (13 days of fieldwork) are in the regression
but not shown in the figure. Interviews 150 or more days after the cut are one window.

Run from the repository root:
uv run python referee_response/other_tables/R2_3_event_study_15day/fuel_event_study_15day.py
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
INPUT = ROOT / "data" / "generated" / "30_analysis_table_input.parquet"

HEAT = "tmean_7d"
GROUP = "urban_vehicle_hh_ifls4"
CONTROLS = (
    "age + female + edu_yrs_nullable + married + widowed + divorced"
    " + ethnicity_nullable + religion_nullable"
)
CUT_DATE = pd.Timestamp("2014-11-18")
WINDOW_DAYS = 15
FIRST_WINDOW = -5  # -75 to -61 days: estimated, not plotted
LAST_WINDOW = 10  # 150 days or more
REFERENCE = -1  # -15 to -1 days

# Sample: IFLS5 respondents with complete outcome, heat, group and controls,
# in kecamatan with more than one respondent (Table B column 4).
df = pd.read_parquet(INPUT)
df = df.loc[df["wave"] == "IFLS5"].copy()
df = df.dropna(
    subset=[
        "cesd_z", HEAT, GROUP, "interview_datetime", "month", "year",
        "gadm_fullcode", "kabupaten_full_code",
        *[column.strip() for column in CONTROLS.split("+")],
    ]
)
df = df.loc[df.groupby("gadm_fullcode")["gadm_fullcode"].transform("size") > 1].copy()
df["V"] = df[GROUP].astype(int)

days = (pd.to_datetime(df["interview_datetime"]).dt.normalize() - CUT_DATE).dt.days
df["window"] = np.floor(days / WINDOW_DAYS).astype(int).clip(FIRST_WINDOW, LAST_WINDOW)
windows = sorted(df["window"].unique())


def name(window: int) -> str:
    return f"w_{'m' if window < 0 else 'p'}{abs(window)}"


for window in windows:
    df[name(window)] = (df["window"] == window).astype(int)

window_terms = " + ".join(
    f"{name(w)} + {name(w)}:{HEAT} + V:{name(w)} + V:{name(w)}:{HEAT}"
    for w in windows
    if w != REFERENCE
)
formula = (
    f"cesd_z ~ {HEAT} + V + V:{HEAT} + {window_terms} + {CONTROLS}"
    " | month^year + gadm_fullcode"
)
model = pf.feols(formula, data=df, vcov={"CRV1": "kabupaten_full_code"})
tidy = model.tidy()

rows = []
for window in windows:
    row = {
        "window": window,
        "first_day": window * WINDOW_DAYS,
        "respondents": int((df["window"] == window).sum()),
        "vehicle_respondents": int(((df["window"] == window) & (df["V"] == 1)).sum()),
    }
    if window == REFERENCE:
        row.update(estimate=0.0, se=0.0, lower=0.0, upper=0.0, p=np.nan)
    else:
        term = [t for t in tidy.index if set(t.split(":")) == {"V", name(window), HEAT}][0]
        coefficient = tidy.loc[term]
        row.update(
            estimate=coefficient["Estimate"], se=coefficient["Std. Error"],
            lower=coefficient["2.5%"], upper=coefficient["97.5%"], p=coefficient["Pr(>|t|)"],
        )
    rows.append(row)
results = pd.DataFrame(rows)
results.to_csv(HERE / "fuel_event_study_15day.csv", index=False)
print(f"Observations: {int(model._N):,}; districts: {df['kabupaten_full_code'].nunique()}")
print(results.round(3).to_string(index=False))

shown = results.loc[results["window"] > FIRST_WINDOW].reset_index(drop=True)
x = np.arange(len(shown))
labels = [
    "150+" if w == LAST_WINDOW
    else f"{w * WINDOW_DAYS}\nto {w * WINDOW_DAYS + WINDOW_DAYS - 1}".replace("-", "−")
    for w in shown["window"]
]
cut_x = int(np.where(shown["window"] == 0)[0][0]) - 0.5
reference_x = int(np.where(shown["window"] == REFERENCE)[0][0])
estimated = shown["window"] != REFERENCE

fig, ax = plt.subplots(figsize=(9.2, 4.6))
ax.axvspan(cut_x, len(shown) - 0.5, color="firebrick", alpha=0.05, lw=0)
ax.axhline(0, color="0.5", lw=0.8)
ax.axvline(cut_x, color="firebrick", ls="--", lw=1)
ax.errorbar(
    x[estimated], shown.loc[estimated, "estimate"],
    yerr=[
        shown.loc[estimated, "estimate"] - shown.loc[estimated, "lower"],
        shown.loc[estimated, "upper"] - shown.loc[estimated, "estimate"],
    ],
    fmt="o", color="navy", capsize=3, ms=5, lw=1.2,
)
ax.plot(reference_x, 0, "o", mfc="white", mec="navy", ms=6)
ax.text(reference_x, -0.03, "reference", ha="center", fontsize=8, color="0.3")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_xlim(-0.5, len(shown) - 0.5)
ax.set_ylim(-0.15, 0.40)
ax.set_ylabel(
    "Heat effect on CES-D (SD per °C):\nvehicle owners minus others,\n"
    "relative to the 15 days before the cut",
    fontsize=9,
)
ax.set_xlabel("Days from the fuel subsidy cut (15-day windows)", fontsize=9)
ax.text((cut_x - 0.5) / 2, 0.375, "Before the cut", ha="center", fontsize=9, color="0.3")
ax.text(
    (cut_x + len(shown) - 0.5) / 2, 0.375, "After the fuel subsidy cut (18 Nov 2014)",
    ha="center", fontsize=9, color="firebrick",
)
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)
fig.tight_layout()
for extension in ["png", "pdf"]:
    fig.savefig(HERE / f"fuel_event_study_15day.{extension}", dpi=200)
