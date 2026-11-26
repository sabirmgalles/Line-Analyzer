"""Deep learning (PyTorch, optionnel) : CNN 1D sur la courbe TDR brute, LSTM sur l'historique SNR/atténuation/CRC.

- TDRNet : convolutions 1D + canal de position (CoordConv) -> type de défaut ET distance (multi-tâche).
  Contrairement au Random Forest, il lit la courbe brute sans features écrites à la main.
- HistNet : LSTM sur 30 jours -> classe de tendance + log(jours avant coupure).

Sans PyTorch l'application fonctionne normalement avec les Random Forest : `AVAILABLE` vaut False.
"""
from __future__ import annotations

import numpy as np

from config import COPPER_CLASSES, HISTORY_CLASSES, MODELS_DIR, SEED, TDR_MAX_RANGE_M

try:
    import torch
    from torch import nn
    AVAILABLE = True
except Exception:          # torch absent : repli silencieux
    AVAILABLE = False

TDR_FILE, HIST_FILE = MODELS_DIR / "tdr_cnn.pt", MODELS_DIR / "history_lstm.pt"

if AVAILABLE:
    class TDRNet(nn.Module):
        def __init__(self, n_classes: int = len(COPPER_CLASSES)):
            super().__init__()
            def block(i, o, k):
                return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2))
            self.conv = nn.Sequential(block(2, 24, 9), block(24, 48, 7), block(48, 64, 5), nn.AdaptiveAvgPool1d(16))
            self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64 * 16, 128), nn.ReLU(), nn.Dropout(0.3))
            self.cls, self.dist = nn.Linear(128, n_classes), nn.Linear(128, 1)

        def forward(self, x):                       # x : (B, N) -> canaux [signal, position]
            pos = torch.linspace(0, 1, x.shape[1], device=x.device).expand_as(x)
            h = self.fc(self.conv(torch.stack([x, pos], dim=1)))
            return self.cls(h), self.dist(h).squeeze(-1)

    class HistNet(nn.Module):
        def __init__(self, n_classes: int = len(HISTORY_CLASSES)):
            super().__init__()
            self.lstm = nn.LSTM(3, 48, num_layers=2, batch_first=True, dropout=0.1)
            self.head = nn.Sequential(nn.Linear(48, 32), nn.ReLU())
            self.cls, self.ttf = nn.Linear(32, n_classes), nn.Linear(32, 1)

        def forward(self, x):                       # x : (B, T, 3)
            h = self.head(self.lstm(x)[0][:, -1])
            return self.cls(h), self.ttf(h).squeeze(-1)


def _seed():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def _batches(n, bs, rng):
    idx = rng.permutation(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


# ----------------------------------------------------------------------------- TDR (CNN 1D)
def train_tdr(tdr, labels, dist_m, tr, te, epochs: int = 25) -> dict:
    """tdr (N,550), labels (N,) str, dist_m (N,) ; tr/te : indices. Retourne métriques et sauvegarde le modèle."""
    _seed()
    mu, sd = float(tdr[tr].mean()), float(tdr[tr].std() + 1e-9)
    X = torch.tensor((tdr - mu) / sd, dtype=torch.float32)
    y = torch.tensor([COPPER_CLASSES.index(l) for l in labels])
    d = torch.tensor(dist_m / TDR_MAX_RANGE_M, dtype=torch.float32)
    fault = torch.tensor(labels != "healthy")
    net, rng = TDRNet(), np.random.default_rng(SEED)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=epochs * int(np.ceil(len(tr) / 64)))
    for ep in range(epochs):
        net.train()
        for b in _batches(len(tr), 64, rng):
            i = torch.tensor(tr[b])
            xb = X[i] + 0.03 * torch.randn_like(X[i])            # augmentation : bruit de mesure
            lc, ld = net(xb)
            m = fault[i]
            loss = nn.functional.cross_entropy(lc, y[i]) + (20 * nn.functional.smooth_l1_loss(ld[m], d[i][m], beta=0.02) if m.any() else 0)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    with torch.no_grad():
        lc, ld = net(X[torch.tensor(te)])
    pred, yt = lc.argmax(1).numpy(), y[torch.tensor(te)].numpy()
    m = fault[torch.tensor(te)].numpy()
    err = np.abs(ld.numpy() * TDR_MAX_RANGE_M - dist_m[te])[m]
    acc = float((pred == yt).mean())
    torch.save({"state": net.state_dict(), "mu": mu, "sd": sd, "classes": COPPER_CLASSES}, TDR_FILE)
    return {"acc": acc, "dist_mae_m": float(err.mean()), "dist_p90_m": float(np.percentile(err, 90))}


def predict_tdr(trace) -> dict | None:
    if not AVAILABLE or not TDR_FILE.exists():
        return None
    b = _load(TDR_FILE, TDRNet)
    x = torch.tensor(((np.asarray(trace, float) - b["mu"]) / b["sd"])[None], dtype=torch.float32)
    with torch.no_grad():
        lc, ld = b["net"](x)
    p = torch.softmax(lc, 1)[0].numpy()
    return {"proba": {c: float(v) for c, v in zip(b["classes"], p)}, "distance_m": float(ld.item() * TDR_MAX_RANGE_M)}


# ----------------------------------------------------------------------------- historique (LSTM)
def _hist_tensor(snr, att, crc, mu=None, sd=None):
    a = np.stack([np.asarray(snr, float), np.asarray(att, float), np.log1p(np.asarray(crc, float))], axis=-1)
    if mu is None:
        mu, sd = a.reshape(-1, 3).mean(0), a.reshape(-1, 3).std(0) + 1e-9
    return torch.tensor((a - mu) / sd, dtype=torch.float32), mu, sd


def train_history(snr, att, crc, labels, ttf_days, tr, te, epochs: int = 40) -> dict:
    _seed()
    X, mu, sd = _hist_tensor(snr[tr], att[tr], crc[tr])
    Xall, _, _ = _hist_tensor(snr, att, crc, mu, sd)
    y = torch.tensor([HISTORY_CLASSES.index(l) for l in labels])
    t = torch.tensor(np.log1p(ttf_days), dtype=torch.float32)
    net, rng = HistNet(), np.random.default_rng(SEED)
    opt = torch.optim.AdamW(net.parameters(), 3e-3, weight_decay=1e-3)
    for ep in range(epochs):
        net.train()
        for b in _batches(len(tr), 64, rng):
            i = torch.tensor(tr[b])
            lc, lt = net(Xall[i] + 0.02 * torch.randn_like(Xall[i]))
            loss = nn.functional.cross_entropy(lc, y[i]) + 0.5 * nn.functional.smooth_l1_loss(lt, t[i])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    ti = torch.tensor(te)
    with torch.no_grad():
        lc, lt = net(Xall[ti])
    acc = float((lc.argmax(1) == y[ti]).float().mean())
    real, pred = ttf_days[te], np.expm1(lt.numpy())
    near = real <= 90
    torch.save({"state": net.state_dict(), "mu": mu, "sd": sd, "classes": HISTORY_CLASSES}, HIST_FILE)
    return {"acc": acc, "ttf_mae_days_under90": float(np.abs(pred[near] - real[near]).mean())}


def predict_history(snr, att, crc) -> dict | None:
    if not AVAILABLE or not HIST_FILE.exists():
        return None
    b = _load(HIST_FILE, HistNet)
    x, _, _ = _hist_tensor(np.asarray(snr)[None], np.asarray(att)[None], np.asarray(crc)[None], b["mu"], b["sd"])
    with torch.no_grad():
        lc, lt = b["net"](x)
    p = torch.softmax(lc, 1)[0].numpy()
    return {"proba": {c: float(v) for c, v in zip(b["classes"], p)}, "ttf_days": float(min(np.expm1(lt.item()), 365.0))}


_NETS: dict = {}


def _load(path, cls) -> dict:
    if path not in _NETS:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        net = cls()
        net.load_state_dict(ck["state"])
        net.eval()
        _NETS[path] = {**ck, "net": net}
    return _NETS[path]


def deep_ready() -> bool:
    return AVAILABLE and TDR_FILE.exists() and HIST_FILE.exists()
