"""R3.3(c): fuel specification on kecamatan interviewed both before and after the 18 Nov 2014 cut. 20 Nov 2026.
Re-implements code/analysis/updates/fuel_cut_both_sides_table.py (GitHub main e7d07d5) outside the repo's caching
layer: same formula, sample rule (kecamatan with post share strictly between 0 and 1 among all IFLS5 rows),
controls, FE and district CRV1. Adds the wet-bulb panel. Full-sample columns must reproduce Table B cols 4-6.
Also writes CES-D threshold shares by wave (12 Nov TODO).
"""
import sys, warnings, pandas as pd, pyfixest as pf
warnings.filterwarnings("ignore")
INP, OUT = sys.argv[1], sys.argv[2]
a = pd.read_parquet(INP)
# CES-D shares
rows = []
for w, d in a.groupby("wave"):
    r = d.cesd_raw.dropna()
    rows.append(dict(wave=w, n=len(r), mean=r.mean(), **{f"share_ge{k}": (r >= k).mean() for k in (8, 10, 12)}))
pd.DataFrame(rows).to_csv(f"{OUT}/cesd_threshold_shares.csv", index=False); print(pd.DataFrame(rows).round(4).to_string())
w5 = a[a.wave == "IFLS5"].copy()
for c in ["post_subsidy", "urban_vehicle_hh_ifls4", "urban_vehicle_transfer_recipient_ifls4", "urban_vehicle_transfer_nonrecipient_ifls4"]: w5[c] = w5[c].astype(float)
share = w5.groupby("gadm_fullcode").post_subsidy.transform("mean")
both = w5[(share > 0) & (share < 1)].copy()
kec = w5.groupby("gadm_fullcode").post_subsidy.mean()
print("kecamatan: all", len(kec), "both", int(((kec > 0) & (kec < 1)).sum()), "pre only", int((kec == 0).sum()), "post only", int((kec == 1).sum()))
print("rows: IFLS5", len(w5), "both", len(both), "pre rows", int((w5.post_subsidy == 0).sum()), "pre rows in both", int((both.post_subsidy == 0).sum()))
X = "age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable"
res = []
for heat in ["tmean_7d", "wetbulb_7d"]:
    for g in ["urban_vehicle_hh_ifls4", "urban_vehicle_transfer_nonrecipient_ifls4", "urban_vehicle_transfer_recipient_ifls4"]:
        for name, d in [("full", w5), ("both_sides", both)]:
            m = pf.feols(f"cesd_z ~ {g}*post_subsidy*{heat} + {X} | month^year + gadm_fullcode", data=d, vcov={"CRV1": "kabupaten_full_code"}); t = m.tidy()
            term = [i for i in t.index if set(i.split(":")) == {g, "post_subsidy", heat}][0]; r = t.loc[term]
            used = d.dropna(subset=["cesd_z", heat, g, "post_subsidy"] + [x.strip() for x in X.split("+")])
            res.append(dict(heat=heat, group=g, sample=name, estimate=r["Estimate"], se=r["Std. Error"], p=r["Pr(>|t|)"], n=int(m._N),
                            kecamatan=used.gadm_fullcode.nunique(), districts=used.kabupaten_full_code.nunique(), group_share=used[g].mean()))
out = pd.DataFrame(res); out.to_csv(f"{OUT}/fuel_both_sides_results.csv", index=False); pd.set_option("display.width", 220); print(out.round(4).to_string())
