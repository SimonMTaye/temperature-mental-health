"""R2.1 placebo-outcome table: fits. 12 Nov 2026.
Sample: saved grip/physical common sample (r2_comment1_current/grip_strength/results/primary_sample.parquet),
6,470 people aged 40+ at baseline. Formula = palm panel formula (Table B col 2). New here: CES-D on the same
sample and wet-bulb versions. The two saved temperature placebo estimates must be reproduced.
"""
import pathlib, pandas as pd, pyfixest as pf
HERE = pathlib.Path(__file__).resolve().parent
d = pd.read_parquet(HERE.parents[1] / "r2_comment1_current/grip_strength/results/primary_sample.parquet")
d["P"] = d["palm_farmer_hh_ifls4"].astype(int); d["ifls5"] = d["ifls5"].astype(int)
ctrl = "age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable"
rows = []
for heat in ["tmean_7d", "wetbulb_7d"]:
    for y in ["cesd_z", "function9_count_baseline_z", "grip_mean_kg_baseline_z"]:
        f = f"{y} ~ {heat} + P:ifls5:{heat} + P:{heat} + ifls5:{heat} + P:ifls5 + {ctrl} | month^year + gadm_fullcode + pidlink"
        m = pf.feols(f, data=d, vcov={"CRV1": "kabupaten_full_code"})
        r = m.tidy().loc[f"P:ifls5:{heat}"]
        rows.append(dict(heat=heat, outcome=y, estimate=r["Estimate"], se=r["Std. Error"], p=r["Pr(>|t|)"], ci_low=r["2.5%"], ci_high=r["97.5%"],
                         n=int(m._N), people=int(d.pidlink.nunique()), palm_people=int(d[d.P == 1].pidlink.nunique()),
                         clusters=int(d.kabupaten_full_code.nunique()), pyfixest=pf.__version__))
out = pd.DataFrame(rows); out.to_csv(HERE / "results.csv", index=False); print(out.round(4).to_string())
