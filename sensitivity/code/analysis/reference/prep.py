"""Data preparation: wide -> street-long panel, cleaning, feature engineering."""
import pandas as pd, numpy as np

RAW = "../../data/bd08c9ec-e189-482f-91cb-4269c21f7a44-n2oemissionsoperatingdatahofen/raw_data/qgl_hof_raw.csv"
SHARED = {"_210_1301": "T_in", "_210_3201": "pH_in", "_322_1101": "Q_out", "_551_5134": "Supern"}
STREET = {
    1: {"_312_3111": "DO1", "_312_3112": "DO2", "_312_3113": "DO3",
        "_f_t_mass_s21_z1": "Air1", "_f_t_mass_s21_z2": "Air2", "_f_t_mass_s21_z3": "Air3",
        "_f_aote_h1": "aOTE1", "_f_aote_h2": "aOTE2", "_f_aote_h3": "aOTE3",
        "_f_nh4_s21_z1_cal": "NH4", "_f_no3_s21_z1_cal": "NO3", "_312_4901": "RS",
        "_f_n2o_emissions_h1": "N2O1", "_f_n2o_emissions_h2": "N2O2", "_f_n2o_emissions_h3": "N2O3"},
    2: {"_312_3121": "DO1", "_312_3122": "DO2", "_312_3123": "DO3",
        "_f_t_mass_s22_z1": "Air1", "_f_t_mass_s22_z2": "Air2", "_f_t_mass_s22_z3": "Air3",
        "_f_aote_h4": "aOTE1", "_f_aote_h5": "aOTE2", "_f_aote_h6": "aOTE3",
        "_f_nh4_s22_z1_cal": "NH4", "_f_no3_s22_z1_cal": "NO3", "_312_4902": "RS",
        "_f_n2o_emissions_h4": "N2O1", "_f_n2o_emissions_h5": "N2O2", "_f_n2o_emissions_h6": "N2O3"},
}
BASE = ["T_in", "pH_in", "Q_out", "Supern", "RS", "DO1", "DO2", "DO3", "Air1", "Air2", "Air3",
        "aOTE1", "aOTE2", "aOTE3", "NH4", "NO3"]
# NB: zone-wise N2O mapping for street 2 follows the meta file (H4=Z2, H5=Z1, H6=Z3)
STREET[2].update({"_f_n2o_emissions_h5": "N2O1", "_f_n2o_emissions_h4": "N2O2", "_f_n2o_emissions_h6": "N2O3",
                  "_f_aote_h5": "aOTE1", "_f_aote_h4": "aOTE2", "_f_aote_h6": "aOTE3"})


def load_long():
    df = pd.read_csv(RAW)
    df["time"] = pd.to_datetime(df["_time"].str[:19]).dt.round("h")
    out = []
    for s, m in STREET.items():
        d = df[["time"] + list(SHARED) + list(m)].rename(columns={**SHARED, **m})
        d["street"] = s
        out.append(d)
    L = pd.concat(out, ignore_index=True)
    # physical-range cleaning (inputs)
    for c in ["aOTE1", "aOTE2", "aOTE3"]:
        L.loc[(L[c] > 30) | (L[c] < -5), c] = np.nan
    for c in ["Air1", "Air2", "Air3"]:
        L[c] = L[c].clip(lower=0)
    L.loc[(L.pH_in < 6) | (L.pH_in > 9.5), "pH_in"] = np.nan
    for c in ["DO1", "DO2", "DO3"]:
        L.loc[L[c] > 8, c] = np.nan
    # target: street-level N2O (sum of three hoods), negatives = analyzer noise -> 0, unit gN/h
    for c in ["N2O1", "N2O2", "N2O3"]:
        L[c] = L[c].clip(lower=0) * 1000
    L["y"] = L[["N2O1", "N2O2", "N2O3"]].sum(axis=1, min_count=3)
    # shared influent chloride (road-salt / snowmelt tracer): street-1 ISE, gaps filled by street-2 ISE
    cl = df.set_index("time")[["_f_cl_s21_z1_cal", "_f_cl_s22_z1_cal"]]
    cl = cl["_f_cl_s21_z1_cal"].fillna(cl["_f_cl_s22_z1_cal"]).groupby(level=0).mean()
    L["Cl"] = L["time"].map(cl)
    # sister-street N2O at the same hour (both hood analysers report in real time)
    sis = L.pivot_table(index="time", columns="street", values="y")
    L["y_sister"] = np.where(L.street == 1, L.time.map(sis.get(2)), L.time.map(sis.get(1)))
    L["swing_on"] = (L["Air1"] > 50).astype(float)
    return L.sort_values(["street", "time"]).reset_index(drop=True)


def engineer(L, h, use_n2o_hist=True):
    """Features known at time t, target y(t+h). Lags/rolls computed within street."""
    g = L.groupby("street", group_keys=False)
    F = L[["time", "street"] + BASE + ["swing_on"]].copy()
    key = ["T_in", "Q_out", "Supern", "DO2", "DO3", "Air1", "Air2", "Air3", "aOTE2", "aOTE3", "NH4", "NO3"]
    for c in key:
        F[c + "_d1"] = g[c].diff(1)
        F[c + "_m6"] = g[c].transform(lambda x: x.rolling(6, min_periods=3).mean())
        F[c + "_m24"] = g[c].transform(lambda x: x.rolling(24, min_periods=12).mean())
    for c in ["NH4", "Air2", "DO2", "Supern"]:
        F[c + "_sd24"] = g[c].transform(lambda x: x.rolling(24, min_periods=12).std())
    F["swing_frac24"] = g["swing_on"].transform(lambda x: x.rolling(24, min_periods=12).mean())
    F["Nload"] = L["NH4"] * L["Q_out"]
    F["hour_sin"] = np.sin(2 * np.pi * L.time.dt.hour / 24); F["hour_cos"] = np.cos(2 * np.pi * L.time.dt.hour / 24)
    F["doy_sin"] = np.sin(2 * np.pi * L.time.dt.dayofyear / 365.25); F["doy_cos"] = np.cos(2 * np.pi * L.time.dt.dayofyear / 365.25)
    F["weekend"] = (L.time.dt.dayofweek >= 5).astype(float)
    F["Cl"] = L["Cl"]
    for c, w in [("T_in", 168), ("Cl", 72), ("Cl", 168), ("NH4", 168), ("Q_out", 72)]:
        F[f"{c}_m{w}"] = g[c].transform(lambda x, w=w: x.rolling(w, min_periods=w // 2).mean())
    F["T_drop72"] = F["T_in"] - g["T_in"].shift(72)
    F["Cl_d24"] = F["Cl"] - g["Cl"].shift(24)
    if use_n2o_hist:
        for z in ["N2O1", "N2O2", "N2O3"]:
            F[z + "_l0"] = np.log1p(L[z])
        F["y_sister"] = np.log1p(L["y_sister"])
        ly = np.log1p(L["y"])
        lg = ly.groupby(L["street"])
        F["y_m72"] = lg.transform(lambda x: x.rolling(72, min_periods=36).mean())
        F["y_m168"] = lg.transform(lambda x: x.rolling(168, min_periods=84).mean())
        F["y_d1"] = lg.diff(1); F["y_d3"] = lg.diff(3)
        F["y_max3"] = lg.transform(lambda x: x.rolling(3, min_periods=1).max())
        for k in [0, 1, 2, 3, 6, 12, 24]:
            F[f"yl{k}"] = lg.shift(k)
        F["y_m6"] = lg.transform(lambda x: x.rolling(6, min_periods=3).mean())
        F["y_m24"] = lg.transform(lambda x: x.rolling(24, min_periods=12).mean())
        F["y_max24"] = lg.transform(lambda x: x.rolling(24, min_periods=12).max())
        F["y_sd24"] = lg.transform(lambda x: x.rolling(24, min_periods=12).std())
    F["target"] = L.groupby("street")["y"].shift(-h)
    F["y_now"] = L["y"]
    return F


if __name__ == "__main__":
    L = load_long()
    print(L.shape, L.y.notna().sum(), L.groupby("street").y.apply(lambda s: s.notna().sum()).to_dict())
    print(L.y.describe(percentiles=[.5, .8, .9, .95, .99]))
    print("zero frac", (L.y < 0.5).mean())
    print(L.groupby(L.time.dt.year).y.agg(["mean", "median", "count"]))
    print(L.groupby("street").y.agg(["mean", "median"]))
    print(L[BASE].isna().mean().round(3).to_dict())
    print(L[BASE + ["y"]].corr(method="spearman")["y"].round(2).sort_values())
    L.to_pickle("long.pkl")
