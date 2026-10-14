"""Sequence baselines (LSTM, GRU, TCN, Transformer) on 48-h windows; CPU-friendly."""
import numpy as np, pandas as pd, torch, torch.nn as nn, time
torch.set_num_threads(2); torch.manual_seed(42); np.random.seed(42)
SEQ = 48
COLS = ["T_in", "pH_in", "Q_out", "Supern", "RS", "DO1", "DO2", "DO3", "Air1", "Air2", "Air3", "aOTE1", "aOTE2", "aOTE3", "NH4", "NO3", "Cl", "swing_on", "hour_sin", "hour_cos"]

class RNN(nn.Module):
    def __init__(s, d, cell):
        super().__init__(); s.r = cell(d, 64, num_layers=2, batch_first=True, dropout=0.1); s.h = nn.Linear(64, 1)
    def forward(s, x): o, _ = s.r(x); return s.h(o[:, -1]).squeeze(-1)

class TCN(nn.Module):
    def __init__(s, d):
        super().__init__(); L = []; c = d
        for dil in [1, 2, 4, 8, 16]:
            L += [nn.Conv1d(c, 64, 3, padding=2 * dil, dilation=dil), nn.GELU()]; c = 64
        s.net = nn.Sequential(*L); s.h = nn.Linear(64, 1)
    def forward(s, x):
        z = s.net(x.transpose(1, 2))[:, :, :x.shape[1]]; return s.h(z[:, :, -1]).squeeze(-1)

class TF(nn.Module):
    def __init__(s, d):
        super().__init__(); s.inp = nn.Linear(d, 64); s.pos = nn.Parameter(torch.randn(1, SEQ, 64) * .02)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(64, 4, 128, 0.1, batch_first=True), 2); s.h = nn.Linear(64, 1)
    def forward(s, x): return s.h(s.enc(s.inp(x) + s.pos)[:, -1]).squeeze(-1)

def run_deep(F, tr, te, H, hist, epochs=12):
    cols = COLS + (["yl0"] if hist else [])
    M = F.reindex(range(F.index.max() + 1))  # not used
    L = pd.read_pickle("long.pkl"); from prep import engineer
    E = engineer(L, 1, use_n2o_hist=hist)[cols + ["street"]]
    E[cols] = E.groupby("street")[cols].ffill(limit=6)
    trmask = E.index.isin(tr.index)
    mu, sd = E.loc[trmask, cols].mean(), E.loc[trmask, cols].std() + 1e-6
    A = ((E[cols] - mu) / sd).fillna(0).values.astype(np.float32)
    start = E.groupby("street").apply(lambda d: d.index.min()).to_dict(); st = E.street.values
    def seqs(idx):
        idx = np.asarray(idx); ok = idx - np.vectorize(start.get)(st[idx]) >= SEQ - 1; idx = idx[ok]
        return idx, np.stack([A[i - SEQ + 1:i + 1] for i in idx])
    itr, Xtr = seqs(tr.index); ite, Xte = seqs(te.index)
    ys = tr.target.std(); ytr = (tr.target.loc[itr].values / ys).astype(np.float32)
    out = {}
    for name, mk in [("LSTM", lambda d: RNN(d, nn.LSTM)), ("GRU", lambda d: RNN(d, nn.GRU)), ("TCN", TCN), ("Transformer", TF)]:
        torch.manual_seed(42); m = mk(len(cols)); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
        Xt, yt = torch.tensor(Xtr), torch.tensor(ytr); t0 = time.time()
        for ep in range(epochs):
            m.train(); perm = torch.randperm(len(Xt))
            for b in range(0, len(Xt), 256):
                j = perm[b:b + 256]; opt.zero_grad(); loss = nn.functional.mse_loss(m(Xt[j]), yt[j]); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): p = np.concatenate([m(torch.tensor(Xte[b:b + 1024])).numpy() for b in range(0, len(Xte), 1024)]) * ys
        full = pd.Series(np.nan, index=te.index); full.loc[ite] = p
        out[name] = full.fillna(tr.target.median()).values
        print(f"   {name} {time.time()-t0:.0f}s", flush=True)
    return out
