"""R2.3 / R3.3a: daily candidate-cutoff sweep, clean presentation. 12 Nov 2026.
Same exercise as Simon's fuel_cutoff_placebo.py (GitHub main e7d07d5): the fuel specification (Table B col 4) is
re-estimated with post defined at each candidate date. Fixed sample: saved 20,038 IFLS5 respondents.
Dates 15 Sep 2014 - 31 Dec 2014. Dates where any group x post cell has < 100 observations are flagged.
"""
import pathlib, warnings, numpy as np, pandas as pd, pyfixest as pf
warnings.filterwarnings("ignore")
H = pathlib.Path(__file__).resolve().parent; ROOT = H.parents[1]
s = pd.read_parquet(H.parent / "r2_comment4_current/equality/results/sample_dry_fuel.parquet")
a = pd.read_parquet(ROOT / "analysis/data/generated/30_analysis_table_input.parquet", columns=["pidlink", "wave", "interview_datetime"])
s = s.merge(a[a.wave == "IFLS5"][["pidlink", "interview_datetime"]], on="pidlink", how="left", validate="1:1")
s["d"] = pd.to_datetime(s.interview_datetime).dt.normalize(); s["V"] = s.urban_vehicle_hh_ifls4.astype(int)
ctrl = "age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable"
f = f"cesd_z ~ tmean_7d + V + post + V:tmean_7d + post:tmean_7d + V:post + V:post:tmean_7d + {ctrl} | month^year + gadm_fullcode"
rows = []
for c in pd.date_range("2014-09-15", "2014-12-31"):
    s["post"] = (s.d >= c).astype(int)
    cells = s.groupby(["V", "post"]).size()
    m = pf.feols(f, data=s, vcov={"CRV1": "kabupaten_full_code"}); r = m.tidy().loc["V:post:tmean_7d"]
    rows.append(dict(cutoff=c.date(), est=r["Estimate"], se=r["Std. Error"], lo=r["2.5%"], hi=r["97.5%"], p=r["Pr(>|t|)"], min_cell=int(cells.min()) if len(cells) == 4 else 0))
pd.DataFrame(rows).to_csv(H / "sweep.csv", index=False)
