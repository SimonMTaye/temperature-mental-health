"""Table R3.1d, full list of housing and asset indicators. 20 Nov 2026.
List fixed before computing (all IFLS housing/asset items observed in both waves that describe the dwelling,
utilities or durable assets). Sample, groups and inference identical to housing_significance_2026-11-05:
continuing non-split-off baseline farming household pairs; change Y5-Y4 on intercept + baseline palm proxy;
CRV1 by baseline district, t(G-1). The six original rows use the saved sidecar columns unchanged.
"""
import sys, json, numpy as np, pandas as pd
from scipy.stats import t
U, OUT = sys.argv[1], sys.argv[2]
R = f"{U}/analysis/data/raw/IFLS/extracted"
s = pd.read_parquet(f"{U}/meeting_robustness_2026-10-09/palm_identification/adaptation/household_linkage_and_outcomes.parquet")
a = pd.read_parquet(f"{U}/analysis/data/generated/30_analysis_table_input.parquet", columns=["hhid", "wave", "kabupaten_full_code"])
b = a.loc[a.wave.eq("IFLS4"), ["hhid", "kabupaten_full_code"]].drop_duplicates(); assert b.hhid.is_unique
s = s.loc[s.eligible_pair & s.farmer_baseline.eq(1)].merge(b, left_on="hhid07", right_on="hhid", validate="1:1")
assert s.hhid07.is_unique and s.hhid14.is_unique
def yn(x, yes, valid):
    x = pd.to_numeric(x, errors="coerce"); o = pd.Series(np.nan, index=x.index); o[x.isin(valid)] = 0.0; o[x.isin(yes)] = 1.0; return o
def build(folder, key):
    kr = pd.read_stata(f"{R}/{folder}/b2_kr.dta", convert_categoricals=False); k = pd.read_stata(f"{R}/{folder}/bk_krk.dta", convert_categoricals=False)
    hr = pd.read_stata(f"{R}/{folder}/b2_hr1.dta", convert_categoricals=False, columns=[key, "hrtype", "hr01"])
    assert kr[key].is_unique and k[key].is_unique
    o = pd.DataFrame({key: kr[key]})
    o["own_house"] = yn(kr.kr03, [1], [1, 2, 5, 95]).values
    o["improved_water"] = yn(kr.kr13, [1, 2, 10], [1, 2, 3, 4, 5, 6, 7, 8, 10, 95]).values
    o["own_toilet"] = yn(kr.kr20, [1, 2], [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 95]).values
    o["clean_stove"] = yn(kr.kr24, [1, 2], [1, 2, 3, 4, 5, 95]).values
    q = pd.DataFrame({key: k[key]})
    q["tile_concrete_roof"] = yn(k.krk10, [1, 4], [1, 2, 3, 4, 5, 6, 95]).values
    q["kitchen_outside"] = yn(k.krk02h, [1], [1, 3]).values
    q["cook_sleep_same_room"] = yn(k.krk02i, [1], [1, 3]).values
    rooms = pd.to_numeric(k.krk06, errors="coerce"); q["rooms"] = rooms.where(rooms < 98).values
    area = pd.to_numeric(k.krk05a.astype(str).str.replace("_", "", regex=False).str.replace(".", "", regex=False) if k.krk05a.dtype != float else k.krk05a, errors="coerce"); q["floor_area"] = area.where((area > 0) & (area < 9998)).values
    o = o.merge(q, on=key, how="outer")
    for code, name in [("E", "vehicle"), ("F", "appliances"), ("G", "savings"), ("J", "jewellery")]:
        h = hr[hr.hrtype == code]; assert h[key].is_unique
        o = o.merge(pd.DataFrame({key: h[key], name: yn(h.hr01, [1], [1, 3]).values}), on=key, how="left")
    return o
w4 = build("IFLS4/hh07", "hhid07"); w5 = build("IFLS5/hh14", "hhid14")
new = [c for c in w4.columns if c != "hhid07"]
s = s.merge(w4.rename(columns={c: c + "_4" for c in new}), on="hhid07", how="left").merge(w5.rename(columns={c: c + "_5" for c in new}), on="hhid14", how="left")
cap = np.nanpercentile(pd.concat([s.floor_area_4, s.floor_area_5]), 99); s["floor_area_4"] = s.floor_area_4.clip(upper=cap); s["floor_area_5"] = s.floor_area_5.clip(upper=cap)
spec = [("A", "adequate_ventilation", "Adequate ventilation", 100), ("A", "finished_floor", "Finished floor", 100), ("A", "masonry_walls", "Masonry walls", 100),
        ("A", "tile_concrete_roof", "Tile or concrete roof", 100), ("A", "kitchen_outside", "Kitchen outside the house", 100),
        ("A", "cook_sleep_same_room", "Cooking and sleeping in same room", 100), ("A", "rooms", "Number of rooms", 1), ("A", "floor_area", "Floor area (m$^2$)", 1),
        ("B", "electricity", "Electricity", 100), ("B", "improved_water", "Piped, pumped or bottled drinking water", 100), ("B", "own_toilet", "Own toilet", 100),
        ("B", "clean_stove", "Cooks with gas or electricity", 100),
        ("C", "own_house", "Owns the house", 100), ("C", "fridge_food_use", "Refrigerator", 100), ("C", "television", "Television", 100),
        ("C", "appliances", "Other household appliances", 100), ("C", "vehicle", "Vehicle", 100), ("C", "savings", "Savings", 100), ("C", "jewellery", "Jewellery", 100)]
rows = []
for panel, v, label, sc in spec:
    d = s.dropna(subset=[v + "_4", v + "_5"]).copy(); d["ch"] = d[v + "_5"] - d[v + "_4"]
    X = np.column_stack([np.ones(len(d)), d.palm_baseline.to_numpy(float)]); y = d.ch.to_numpy(float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]; res = y - X @ beta
    cc, cl = pd.factorize(d.kabupaten_full_code); G = len(cl); N = len(d)
    sc_ = np.zeros((G, 2)); np.add.at(sc_, cc, X * res[:, None]); br = np.linalg.inv(X.T @ X)
    V = br @ (sc_.T @ sc_) @ br * (G / (G - 1)) * ((N - 1) / (N - 2)); se = np.sqrt(V[1, 1])
    p1 = d[d.palm_baseline == 1]; p0 = d[d.palm_baseline == 0]
    rows.append(dict(panel=panel, var=v, label=label, scale=sc, palm_4=sc * p1[v + "_4"].mean(), palm_5=sc * p1[v + "_5"].mean(), palm_ch=sc * (beta[0] + beta[1]),
                     oth_4=sc * p0[v + "_4"].mean(), oth_5=sc * p0[v + "_5"].mean(), oth_ch=sc * beta[0], diff=sc * beta[1], se=sc * se,
                     p=2 * t.sf(abs(beta[1] / se), G - 1), n_palm=len(p1), n_oth=len(p0), districts=G))
r = pd.DataFrame(rows); o = np.argsort(r.p.values); adj = np.empty(len(r)); adj[o] = np.minimum(1, np.maximum.accumulate(r.p.values[o] * np.arange(len(r), 0, -1))); r["holm_p"] = adj
r.to_csv(f"{OUT}/results.csv", index=False)
json.dump({"floor_area_cap_p99": float(cap), "palm_households": int(s.palm_baseline.sum()), "other_households": int((s.palm_baseline == 0).sum())}, open(f"{OUT}/info.json", "w"))
pd.set_option("display.width", 250); print(r.drop(columns=["var", "scale"]).round(2).to_string())
