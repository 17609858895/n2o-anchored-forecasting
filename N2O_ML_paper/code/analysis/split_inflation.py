"""Validation-design audit: how much do random splits inflate apparent skill at Hofen?

The same feature set, target (6-h mean emission) and model settings as the main benchmark
are evaluated under three validation designs:
  chrono   train 2020-2021, test 2022 (main design, V1)
  week     random 30% of calendar weeks held out (blocked random split), 3 repeats
  random   random 30% of street-hours held out (as in many published N2O models), 3 repeats
Random forest and extra trees (slow) are refitted only in the first repeat of each random design.
For every design the skill relative to persistence is computed on the same held-out hours.
Writes ../../results/split_inflation.csv and ../../results/split_inflation_summary.csv.
Run from code/analysis after prep.py:  python split_inflation.py
"""
import warnings
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from bench2 import build, LG, RS

warnings.filterwarnings("ignore")
H = 6
REPEATS = 3


def models():
    imp = lambda: SimpleImputer(strategy="median")
    return {
        "Ridge": lambda: make_pipeline(imp(), StandardScaler(), Ridge(alpha=10)),
        "Random forest": lambda: make_pipeline(imp(), RandomForestRegressor(n_estimators=300, min_samples_leaf=10, max_features=0.4, n_jobs=-1, random_state=RS)),
        "Extra trees": lambda: make_pipeline(imp(), ExtraTreesRegressor(n_estimators=400, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=RS)),
        "LightGBM": lambda: lgb.LGBMRegressor(**LG),
    }


def evaluate(tr, te, feats, design, rep, slow=True):
    med = np.nanmedian(tr.target)
    ytr, yte = tr.target.values, te.target.values
    atr, ate = tr.pers.fillna(med).values, te.pers.fillna(med).values
    mse_p = mean_squared_error(yte, ate)
    rows = [dict(design=design, rep=rep, model="Persistence", R2=r2_score(yte, ate), Skill=0.0, n_test=len(te))]
    for name, mk in models().items():
        if not slow and name in ("Random forest", "Extra trees"):
            continue                       # bagged forests only in the first repeat (run time)
        p = np.clip(mk().fit(tr[feats], ytr).predict(te[feats]), 0, None)
        rows.append(dict(design=design, rep=rep, model=name, R2=r2_score(yte, p),
                         Skill=1 - mean_squared_error(yte, p) / mse_p, n_test=len(te)))
    # anchored LightGBM (residual expert): predicts the deviation from the current emission
    p = np.clip(ate + lgb.LGBMRegressor(**LG).fit(tr[feats], ytr - atr).predict(te[feats]), 0, None)
    rows.append(dict(design=design, rep=rep, model="Anchored LightGBM", R2=r2_score(yte, p),
                     Skill=1 - mean_squared_error(yte, p) / mse_p, n_test=len(te)))
    for r in rows:
        print(f"{design:7s} {rep} {r['model']:18s} R2={r['R2']:.3f} skill={r['Skill']:+.3f}", flush=True)
    return rows


def main():
    F, feats = build(H)
    F = F[F.time < pd.Timestamp("2023-01-01")].reset_index(drop=True)
    rows = []
    # chronological (main design)
    cut = pd.Timestamp("2022-01-01")
    tr = F[F.time < cut - pd.Timedelta(hours=H)]; te = F[F.time >= cut]
    rows += evaluate(tr, te, feats, "chrono", 0)
    week = F.time.dt.to_period("W").astype(str)
    weeks = np.array(sorted(week.unique()))
    for rep in range(REPEATS):
        rng = np.random.RandomState(RS + rep)
        # blocked random split: whole calendar weeks (both streets) held out
        test_weeks = set(rng.choice(weeks, int(round(0.3 * len(weeks))), replace=False))
        m = week.isin(test_weeks).values
        rows += evaluate(F[~m], F[m], feats, "week", rep, slow=rep == 0)
        # hour-level random split
        m = rng.rand(len(F)) < 0.3
        rows += evaluate(F[~m], F[m], feats, "random", rep, slow=rep == 0)
    R = pd.DataFrame(rows)
    R.to_csv("../../results/split_inflation.csv", index=False)
    S = R.groupby(["design", "model"]).agg(R2_mean=("R2", "mean"), R2_min=("R2", "min"), R2_max=("R2", "max"),
                                           Skill_mean=("Skill", "mean"), Skill_min=("Skill", "min"),
                                           Skill_max=("Skill", "max"), n_test=("n_test", "mean")).reset_index()
    S.to_csv("../../results/split_inflation_summary.csv", index=False)
    print(S.round(3).to_string())


if __name__ == "__main__":
    main()
