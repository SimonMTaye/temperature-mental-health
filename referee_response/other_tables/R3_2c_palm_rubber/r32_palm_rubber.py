"""R3.2c / R3.2d restricted to palm and rubber, without and with the R3.2b crop controls. 20 Nov 2026.
Controls (identical to r3_2b_simple_controls_2026-11-05): baseline reported-crop indicators x IFLS5 and
crop-specific log international price changes x IFLS5 for rice, maize, rubber and coffee, plus unknown-crop x IFLS5.
Samples and formulas are the saved R3.2c / R3.2d ones (40,012 obs; individual, kecamatan, month-year FE; district CRV1).
No-control columns must reproduce the saved estimates.
"""
import sys, warnings, pandas as pd, pyfixest as pf
warnings.filterwarnings("ignore")
B, OUT = sys.argv[1], sys.argv[2]
b = pd.read_parquet(f"{B}/r3_2b_simple_controls_2026-11-05/sample.parquet")
ctrl_cols = ["unknown_post"] + [f"{k}_{c}_post" for c in ["rice", "maize", "rubber", "coffee"] for k in ["crop", "price"]]
X = "age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable"
FE = " | month^year + gadm_fullcode + pidlink"; T = "tmean_7d"
def tri(g): return f"cesd_z ~ {T} + {g}:ifls5:{T} + {g}:{T} + ifls5:{T} + {g}:ifls5 + {X}"
rows = []
for table, folder, groups in [("R3.2c", "r3_2c_reported_crops_full_sample_2026-11-05", {"palm": ("palm_farmer_hh_ifls4", None), "rubber": ("crop_rubber", "unknown_rubber")}),
                              ("R3.2d", "r3_2d_palm_price_other_crops_2026-11-05", {"palm": ("exposure_palm", None), "rubber": ("exposure_rubber", "unknown_exposure_rubber")})]:
    d = pd.read_parquet(f"{B}/{folder}/sample.parquet").merge(b[["pidlink", "wave"] + ctrl_cols], on=["pidlink", "wave"], how="left", validate="1:1")
    assert d[ctrl_cols].notna().all().all() and len(d) == 40012
    d["ifls5"] = d.ifls5.astype(int)
    saved = pd.read_csv(f"{B}/{folder}/results.csv").set_index("spec")
    for name, (g, unk) in groups.items():
        f0 = tri(g) + (f" + {unk}:{T} + {unk}:ifls5 + {unk}:ifls5:{T}" if unk else "")
        for spec, f in [("no_controls", f0), ("crop_and_price_controls", f0 + " + " + " + ".join(ctrl_cols))]:
            m = pf.feols(f + FE, data=d, vcov={"CRV1": "kabupaten_full_code"}); t = m.tidy()
            term = [i for i in t.index if set(i.split(":")) == {g, "ifls5", T}][0]; r = t.loc[term]
            if spec == "no_controls": assert abs(r["Estimate"] - saved.loc[name, "estimate"]) < 1e-6, (table, name, r["Estimate"], saved.loc[name, "estimate"])
            ex = d[(d.ifls5 == 0) & (d[g] != 0)] if table == "R3.2d" else d[(d.ifls5 == 0) & (d[g] == 1)]
            rows.append(dict(table=table, group=name, spec=spec, estimate=r["Estimate"], se=r["Std. Error"], p=r["Pr(>|t|)"], n=int(m._N), people=d.pidlink.nunique(),
                             districts=d.kabupaten_full_code.nunique(), exposed_people=int(saved.loc[name, "exposed_people"]), exposed_households=int(saved.loc[name, "exposed_households"]),
                             dropped=";".join(getattr(m, "_collin_vars", []) or [])))
out = pd.DataFrame(rows); out.to_csv(f"{OUT}/results.csv", index=False); pd.set_option("display.width", 220); print(out.round(4).to_string())
