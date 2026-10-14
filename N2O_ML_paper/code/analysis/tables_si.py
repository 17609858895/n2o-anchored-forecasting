"""Supporting tables S1-S6 -> tables/*.csv and tables/Supporting_Tables.xlsx"""
import json, pickle, os, glob, re, numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, f1_score
from scipy import stats
from prep import SHARED, STREET, BASE
import bench2 as B
from stats_joint import dm_joint
os.makedirs("tables", exist_ok=True)
L = pd.read_pickle("long.pkl")
meta = json.load(open(glob.glob("../../data/*/meta_data/columns_translator_lane22.json")[0]))
SUB = str.maketrans("0123456789+-", "₀₁₂₃₄₅₆₇₈₉₊₋"); SUP = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")
def LABEL(s):
    s = re.sub(r"\$_\{([^}]*)\}\^\{([^}]*)\}\$", lambda m: m.group(1).translate(SUB) + m.group(2).translate(SUP), s)
    s = re.sub(r"\$_\{([^}]*)\}\$", lambda m: m.group(1).translate(SUB), s)
    s = re.sub(r"\$\^\{([^}]*)\}\$", lambda m: m.group(1).translate(SUP), s)
    return s.replace("$", "")

# ---------------- Table S1 sensors
UNIT = {"T_in": "°C", "pH_in": "–", "Q_out": "not stated in metadata", "Supern": "% valve opening (inferred from 0–100 range)", "RS": "% pump output (inferred from 0–100 range)", "DO1": "mg O₂ L⁻¹", "DO2": "mg O₂ L⁻¹",
        "DO3": "mg O₂ L⁻¹", "Air1": "not stated in metadata (mass flow)", "Air2": "not stated in metadata (mass flow)", "Air3": "not stated in metadata (mass flow)",
        "aOTE1": "%", "aOTE2": "%", "aOTE3": "%", "NH4": "mg N L⁻¹", "NO3": "mg N L⁻¹", "Cl": "mg L⁻¹", "N2O1": "g N h⁻¹", "N2O2": "g N h⁻¹", "N2O3": "g N h⁻¹"}
FILT = {"pH_in": "set missing outside 6–9.5", "DO1": "set missing > 8", "DO2": "set missing > 8", "DO3": "set missing > 8",
        "aOTE1": "set missing outside −5–30", "aOTE2": "set missing outside −5–30", "aOTE3": "set missing outside −5–30",
        "Air1": "negative → 0", "Air2": "negative → 0", "Air3": "negative → 0", "N2O1": "negative → 0; reported in kg N h⁻¹, converted to g N h⁻¹", "N2O2": "negative → 0; reported in kg N h⁻¹, converted to g N h⁻¹",
        "N2O3": "negative → 0; reported in kg N h⁻¹, converted to g N h⁻¹", "Cl": "street-1 ISE, gaps filled with street-2 ISE (shared influent)"}
inv = {}
for s, m in STREET.items():
    for tag, v in m.items(): inv.setdefault(v, {})[s] = tag
for tag, v in SHARED.items(): inv[v] = {1: tag, 2: tag}
inv["Cl"] = {1: "_f_cl_s21_z1_cal", 2: "_f_cl_s22_z1_cal"}
rows = []
for v in BASE + ["Cl", "N2O1", "N2O2", "N2O3"]:
    t1, t2 = inv[v][1], inv[v][2]
    rows.append({"Variable": v, "Description (dataset metadata)": LABEL(meta.get(t2, meta.get(t1, ""))).replace("zone", "zone"), "Unit": UNIT.get(v, ""),
                 "Tag street 2.1": t1, "Tag street 2.2": t2,
                 "Missing street 2.1 (%)": round(100 * L.loc[L.street == 1, v].isna().mean(), 1),
                 "Missing street 2.2 (%)": round(100 * L.loc[L.street == 2, v].isna().mean(), 1),
                 "Cleaning rule": FILT.get(v, "none")})
S1 = pd.DataFrame(rows)

# ---------------- Table S2 features
feats = pickle.load(open("feats.pkl", "rb"))
GROUPS = [("N2O history", lambda c: c.startswith(("yl", "y_", "N2O"))), ("Side-stream", lambda c: c.startswith("Supern")),
          ("Aeration / DO", lambda c: c.startswith(("DO", "Air", "aOTE", "swing"))), ("N loading", lambda c: c.startswith(("NH4", "NO3", "Nload"))),
          ("Hydraulics", lambda c: c.startswith(("Q_out", "RS"))), ("Temperature / snowmelt", lambda c: c.startswith(("T_in", "Cl", "T_drop", "pH"))),
          ("Calendar / street", lambda c: c.startswith(("hour", "doy", "weekend", "street")))]
def describe(c):
    if c in BASE: return f"Sensor value at hour t ({LABEL(meta.get(inv[c][2], c))})"
    if c == "Cl": return "Influent chloride at hour t (shared ISE)"
    if c == "swing_on": return "Swing zone aerated (Air1 > 50) at hour t"
    if c == "street": return "Street indicator"
    if c == "Nload": return "NH4+ × Q_out (ammonium load proxy)"
    if c == "swing_frac24": return "Fraction of last 24 h with swing zone aerated"
    if c == "T_drop72": return "T(t) − T(t − 72 h)"
    if c == "Cl_d24": return "Cl(t) − Cl(t − 24 h)"
    if c in ("hour_sin", "hour_cos"): return "Hour of day, sine/cosine"
    if c in ("doy_sin", "doy_cos"): return "Day of year, sine/cosine"
    if c == "weekend": return "Saturday or Sunday"
    if c.endswith("_l0"): return f"log(1 + N2O) of zone {c[3]} at hour t"
    if c.startswith("yl"): return f"log(1 + street N2O) at t − {c[2:]} h"
    m = {"y_m72": "72 h mean of log(1 + N2O)", "y_m168": "168 h mean of log(1 + N2O)", "y_d1": "1 h change of log(1 + N2O)",
         "y_d3": "3 h change of log(1 + N2O)", "y_max3": "3 h max of log(1 + N2O)", "y_m6": "6 h mean of log(1 + N2O)",
         "y_m24": "24 h mean of log(1 + N2O)", "y_max24": "24 h max of log(1 + N2O)", "y_sd24": "24 h SD of log(1 + N2O)"}
    if c in m: return m[c]
    base, suf = c.rsplit("_", 1)
    return {"d1": f"1 h change of {base}", "m6": f"6 h mean of {base}", "m24": f"24 h mean of {base}", "m72": f"72 h mean of {base}",
            "m168": f"168 h mean of {base}", "sd24": f"24 h SD of {base}"}.get(suf, c)
S2 = pd.DataFrame([{"Feature": c, "Group": next(g for g, f in GROUPS if f(c)), "Definition": describe(c),
                    "Type": "raw sensor" if c in BASE + ["Cl"] else ("N2O history" if next(g for g, f in GROUPS if f(c)) == "N2O history" else "engineered")} for c in feats])

# ---------------- Table S4 full metrics + DM p
def dm(e1, e2, h):
    d = e1 ** 2 - e2 ** 2; n = len(d); db = d.mean(); Lg = 4 * h
    g = [np.mean((d[k:] - db) * (d[:n - k] - db)) for k in range(Lg + 1)]
    v = g[0] + 2 * sum((1 - k / (Lg + 1)) * g[k] for k in range(1, Lg + 1)); s = db / np.sqrt(v / n)
    return 2 * (1 - stats.norm.cdf(abs(s)))
NAME = {"SeasonalNaive": "Seasonal naive", "ClimMean": "Climatology", "DT": "Decision tree", "RF": "Random forest", "ExtraTrees": "Extra trees",
        "[expert] resid": "Residual expert", "[expert] raw": "Raw-scale expert", "[expert] log": "Log-scale expert", "[expert] tweedie": "Tweedie expert"}
NAME.update({"DT": "Decision tree"})
rows = []
for key, scen in [("H1_chrono_h1_2022", "V1"), ("H3_chrono_h1_2022", "V1"), ("H6_chrono_hist1", "V1"), ("H12_chrono_h1_2022", "V1"),
                  ("H24_chrono_h1_2022", "V1"), ("H6_chrono_h1_2021", "V2"), ("H6_xstreet_h1_2022", "V3")]:
    d = pickle.load(open(f"pred_{key}.pkl", "rb"))
    if key == "H6_chrono_hist1":
        b6 = pickle.load(open("pred_H6_chrono_h1_2022.pkl", "rb")); d = dict(d); d["preds"] = dict(d["preds"]); d["preds"]["EG-MoE"] = b6["preds"]["EG-MoE"]
        for k_ in ["time", "street", "tau"]: d[k_] = b6[k_]
    tm = pd.to_datetime(d["time"]); o = np.lexsort((tm.values, d["street"]))
    y = d["y"][o]; h = int(key.split("_")[0][1:]); tau = d["tau"]; eg = np.clip(d["preds"]["EG-MoE"], 0, None)[o]
    pers = np.clip(d["preds"]["Persistence"], 0, None)[o] if "Persistence" in d["preds"] else None
    for m, p in d["preds"].items():
        p = np.clip(p, 0, None)[o]
        r = dict(Scenario=scen, H=h, Test=("2021" if key.endswith("2021") else "2022") + (" (street 2.2)" if "xstreet" in key else ""), Model=NAME.get(m, m),
                 R2=r2_score(y, p), RMSE=mean_squared_error(y, p) ** .5, MAE=mean_absolute_error(y, p), R2_log=r2_score(np.log1p(y), np.log1p(p)),
                 F1=f1_score(y > tau, p > tau), CumErr_pct=100 * (p.sum() - y.sum()) / y.sum(),
                 Skill_vs_persistence=1 - mean_squared_error(y, p) / mean_squared_error(y, pers) if pers is not None else np.nan,
                 DM_p_vs_EGMoE=np.nan if m == "EG-MoE" else dm_joint(y - eg, y - p, tm[o], h)[1])
        rows.append(r)
S4 = pd.DataFrame(rows).round(4)

# ---------------- Table S5 DML
a = pd.read_csv("dml_pilot_v2.csv"); b = pd.read_csv("figdata/dml_ate_block.csv")
a["Effect (%)"] = a["pct"].round(1); a["95% CI (forest/analytic, %)"] = [f"[{100 * (np.exp(l) - 1):.1f}, {100 * (np.exp(h) - 1):.1f}]" if pd.notna(l) else "" for l, h in zip(a.lo, a.hi)]
a = a[~(a.estimator.isin(["LinearDML-RF", "CausalForestDML"]) & a.treatment.str.startswith(("Swing", "Supernatant valve open")))]  # binary-treatment estimates are reported once, in the main table
S5a = a[["treatment", "estimator", "n", "ATE_log", "Effect (%)", "95% CI (forest/analytic, %)"]].rename(columns={"treatment": "Treatment", "estimator": "Estimator / refutation", "ATE_log": "Effect on log(1 + N2O)"})
b["Effect (%)"] = (100 * (np.exp(b.ate) - 1)).round(1)
b["95% calendar-week block CI (%)"] = [f"[{100 * (np.exp(l) - 1):.1f}, {100 * (np.exp(h) - 1):.1f}]" if pd.notna(l) else "" for l, h in zip(b.lo_block, b.hi_block)]
b["95% analytic / forest CI (%)"] = [f"[{100 * (np.exp(l) - 1):.1f}, {100 * (np.exp(h) - 1):.1f}]" for l, h in zip(b.lo_naive, b.hi_naive)]
S5b = b[["treatment", "estimator", "n", "ate", "Effect (%)", "95% calendar-week block CI (%)", "95% analytic / forest CI (%)"]].rename(
    columns={"treatment": "Treatment", "estimator": "Estimator", "ate": "Effect on log(1 + N2O)"})
if os.path.exists("figdata/si/S12.pkl"):
    s12 = pickle.load(open("figdata/si/S12.pkl", "rb")); S5c = pd.DataFrame(s12["nuis"]).round(3); S5d = pd.DataFrame(s12["evalue"]).round(2)
else:
    S5c = S5d = pd.DataFrame()

# ---------------- Table S6 ML audit
S6 = pd.read_csv("tables/ml_audit_H6.csv").round(3) if os.path.exists("tables/ml_audit_H6.csv") else pd.DataFrame()
if len(S6): S6["model"] = S6["model"].map(lambda m: NAME.get(m, m))

# ---------------- Table S3 hyperparameters
hp = {"Persistence": "ŷ = y(t)", "Seasonal naive": "mean of the same H-hour window on the previous day", "Climatology": "training mean",
      "Ridge": "median imputation, standardisation, α = 10", "ElasticNet": "median imputation, standardisation, α = 0.05, l1 ratio 0.5",
      "KNN": "median imputation, standardisation, k = 25, distance weights", "SVR": "RBF kernel, C = 30, ε = 1, γ = scale; 8,000-hour training subsample",
      "Decision tree": "max depth 10, min 50 samples per leaf", "Random forest": "300 trees, min 10 samples per leaf, max features 0.4",
      "Extra trees": "400 trees, min 5 samples per leaf, max features 0.5", "AdaBoost": "60 depth-6 trees, learning rate 0.05",
      "HistGB": "500 iterations, learning rate 0.05, 31 leaves, min 40 samples per leaf, L2 1.0",
      "XGBoost": "600 trees, learning rate 0.03, depth 6, subsample 0.8, colsample 0.7, min child weight 10",
      "LightGBM": "600 trees, learning rate 0.03, 31 leaves, min 40 samples per leaf, subsample 0.8, colsample 0.7, L2 1.0",
      "CatBoost": "800 iterations, learning rate 0.05, depth 6, L2 3", "MLP": "median imputation, standardisation, 128–64 units, α = 1e-3, early stopping",
      "LSTM / GRU": "2 layers × 64 units, dropout 0.1; 48-h window of 21 inputs; AdamW (lr 1e-3, weight decay 1e-4), batch 256, 12 epochs",
      "TCN": "5 dilated causal convolution blocks (dilation 1–16), 64 channels; training as LSTM",
      "Transformer": "2 encoder layers, d_model 64, 4 heads, feed-forward 128, learned positional embedding; training as LSTM",
      "LSTM-A / GRU-A / TCN-A / Transformer-A": "as the unanchored networks, trained on y(t+1:t+H) − y(t); anchor added back",
      "EG-MoE": "5 experts with LightGBM settings as above (Tweedie power 1.5); LightGBM episode classifier; GMM k = 6 (full covariance, 3 initialisations); bins p < 0.15, 0.15–0.5, ≥ 0.5; λ = 500; 4 forward-chained out-of-fold blocks"}
S3 = pd.DataFrame([{"Model": k, "Setting": v} for k, v in hp.items()])
if os.path.exists("figdata/si/S4.pkl"):
    s4 = pickle.load(open("figdata/si/S4.pkl", "rb"))
    S3b = pd.DataFrame([{"Model": k, "Trials": len(v["values"]), "Best CV RMSE": round(min(v["values"]), 2), "Default CV RMSE": round(v["cv_default"], 2),
                         "Best parameters": json.dumps({a: (round(b, 4) if isinstance(b, float) else b) for a, b in v["best_params"].items()}),
                         "Train R2 default": round(v.get("train_default", float("nan")), 3), "Train R2 tuned": round(v.get("train_tuned", float("nan")), 3), "Test R2 default": round(v["test_default"]["R2"], 3), "Test R2 tuned": round(v["test_tuned"]["R2"], 3),
                         "Test RMSE default": round(v["test_default"]["RMSE"], 2), "Test RMSE tuned": round(v["test_tuned"]["RMSE"], 2)} for k, v in s4.items()])
else:
    S3b = pd.DataFrame()

with pd.ExcelWriter("tables/Supporting_Tables.xlsx") as xw:
    for name, df in [("Table S1 sensors", S1), ("Table S2 features", S2), ("Table S3 default settings", S3), ("Table S3 Optuna", S3b), ("Table S4 ML audit", S6), ("Table S5 metrics", S4)] + [(n, pd.read_csv(f)) for n, f in [("Table S6 GRU seeds", "tables/Table_S6_GRU_seeds.csv"), ("Table S7 DM split", "tables/Table_S7_DM_split.csv")] if os.path.exists(f)] + [
                     ("Table S8 DML", S5b), ("Table S8 refutations", S5a), ("Table S8 nuisance", S5c), ("Table S8 E-values", S5d)]:
        if len(df): df.to_excel(xw, sheet_name=name[:31], index=False); df.to_csv(f"tables/{name.replace(' ', '_')}.csv", index=False, encoding="utf-8-sig")
print(S1.head(3).to_string()); print(S2.Group.value_counts()); print(S4.shape); print(S5b.to_string())
