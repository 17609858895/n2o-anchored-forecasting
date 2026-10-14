"""Inference helpers that treat the two streets as one system.
dm_joint: Diebold-Mariano test on the squared-error differential averaged over streets at each time stamp (HAC, Bartlett, lag 4H).
week_id: calendar week shared by both streets, used as the bootstrap block."""
import numpy as np, pandas as pd
from scipy import stats


def dm_joint(e1, e2, time, h):
    d = pd.Series(np.asarray(e1) ** 2 - np.asarray(e2) ** 2).groupby(pd.to_datetime(np.asarray(time))).mean().sort_index().values
    n = len(d); db = d.mean(); L = 4 * h
    g = [np.mean((d[k:] - db) * (d[:n - k] - db)) for k in range(L + 1)]
    v = g[0] + 2 * sum((1 - k / (L + 1)) * g[k] for k in range(1, L + 1)); s = db / np.sqrt(v / n)
    return s, 2 * (1 - stats.norm.cdf(abs(s)))


def week_id(t, s=None):
    t = pd.to_datetime(t); iso = t.isocalendar()
    return (iso.year.values.astype(int) * 100 + iso.week.values.astype(int))
