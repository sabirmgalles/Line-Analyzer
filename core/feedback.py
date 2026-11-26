"""Auto-apprentissage : retours du technicien -> suivi de performance -> ré-entraînement contrôlé.

Boucle :
1. Le technicien confirme (ou corrige) le défaut réel après intervention (et la distance réelle pour le cuivre).
2. `live_performance` mesure la précision RÉELLE du modèle sur ces retours (dérive éventuelle par rapport aux scores de test).
3. `retrain` entraîne un modèle candidat sur les données d'origine + les retours (pondérés), le compare au modèle actuel sur
   un jeu de retours mis de côté et sur le jeu de test d'origine, et ne le déploie que s'il est meilleur sans régression.
   L'ancien modèle est archivé : `rollback` le restaure.

Les retours sont stockés en local (SQLite, data/feedback.db) sous forme de features, sans les courbes brutes.
"""
from __future__ import annotations

import json
import contextlib
import shutil
import sqlite3
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix

from config import COPPER_CLASSES, DATA_DIR, FIBER_CLASSES, MODELS_DIR, RADIO_CLASSES, SEED
from core.features import COPPER_FEATURES, FIBER_FEATURES, RADIO_FEATURES

DB_PATH = DATA_DIR / "feedback.db"
ARCHIVE = MODELS_DIR / "archive"
REGISTRY = MODELS_DIR / "registry.json"
CFG = {
    "copper": {"model": "copper_fault", "features": COPPER_FEATURES, "classes": COPPER_CLASSES, "label": "Cuivre"},
    "fiber": {"model": "fiber_fault", "features": FIBER_FEATURES, "classes": FIBER_CLASSES, "label": "Fibre"},
    "radio": {"model": "radio_fault", "features": RADIO_FEATURES, "classes": RADIO_CLASSES, "label": "Radio"},
}


# ----------------------------------------------------------------------------- stockage
@contextlib.contextmanager
def _db():
    DB_PATH.parent.mkdir(exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    try:
        yield _init(c)
        c.commit()
    finally:
        c.close()


def _init(c: sqlite3.Connection) -> sqlite3.Connection:
    c.execute("""CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, domain TEXT, features TEXT,
                 predicted TEXT, confidence REAL, confirmed TEXT, distance_pred REAL, distance_true REAL, tech TEXT, note TEXT,
                 simulated INTEGER DEFAULT 0)""")
    return c


def add_feedback(domain: str, features: dict, predicted: str, confidence: float, confirmed: str,
                 distance_pred: float | None = None, distance_true: float | None = None, tech: str = "", note: str = "",
                 simulated: bool = False) -> int:
    if domain not in CFG:
        raise ValueError(f"Domaine inconnu : {domain}")
    if confirmed not in CFG[domain]["classes"]:
        raise ValueError(f"Classe inconnue pour {domain} : {confirmed}")
    row = {k: float(features[k]) for k in CFG[domain]["features"]}
    with _db() as c:
        cur = c.execute("INSERT INTO feedback (ts,domain,features,predicted,confidence,confirmed,distance_pred,distance_true,tech,note,simulated)"
                        " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (time.time(), domain, json.dumps(row), predicted, float(confidence), confirmed, distance_pred, distance_true,
                         tech, note, int(simulated)))
        return int(cur.lastrowid)


def load_feedback(domain: str, include_simulated: bool = True) -> pd.DataFrame:
    with _db() as c:
        df = pd.read_sql_query("SELECT * FROM feedback WHERE domain=? ORDER BY ts", c, params=(domain,))
    if not include_simulated:
        df = df[df["simulated"] == 0]
    return df.reset_index(drop=True)


def clear_feedback(domain: str | None = None, only_simulated: bool = False):
    with _db() as c:
        q, p = "DELETE FROM feedback WHERE 1=1", []
        if domain:
            q += " AND domain=?"; p.append(domain)
        if only_simulated:
            q += " AND simulated=1"
        c.execute(q, p)


def _feature_matrix(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    return pd.DataFrame([json.loads(s) for s in df["features"]])[CFG[domain]["features"]]


# ----------------------------------------------------------------------------- suivi de performance
def live_performance(domain: str, window: int = 20) -> dict:
    df = load_feedback(domain)
    if df.empty:
        return {"n": 0}
    ok = (df["predicted"] == df["confirmed"]).values
    classes = CFG[domain]["classes"]
    cm = confusion_matrix(df["confirmed"], df["predicted"], labels=classes)
    rolling = pd.Series(ok.astype(float)).rolling(window, min_periods=min(5, len(ok))).mean().tolist()
    by_class = {c: float(ok[df["confirmed"].values == c].mean()) for c in classes if (df["confirmed"] == c).any()}
    conf = df["confidence"].values
    errs = df[~ok].groupby(["confirmed", "predicted"]).size().sort_values(ascending=False)
    dist = df.dropna(subset=["distance_true", "distance_pred"])
    return {"n": int(len(df)), "accuracy": float(ok.mean()), "rolling": rolling, "by_class": by_class,
            "confusion": cm, "classes": classes, "conf_correct": float(conf[ok].mean()) if ok.any() else None,
            "conf_wrong": float(conf[~ok].mean()) if (~ok).any() else None,
            "top_confusions": [(a, b, int(n)) for (a, b), n in errs.head(3).items()],
            "distance_mae_m": float((dist["distance_true"] - dist["distance_pred"]).abs().mean()) if len(dist) else None,
            "n_distance": int(len(dist)), "n_simulated": int(df["simulated"].sum())}


# ----------------------------------------------------------------------------- données d'origine
def save_base(domain: str, X: pd.DataFrame, y, tr, te, dist=None):
    joblib.dump({"X": X.reset_index(drop=True), "y": np.asarray(y), "tr": np.asarray(tr), "te": np.asarray(te), "dist": None if dist is None else np.asarray(dist)},
                DATA_DIR / f"base_{domain}.joblib", compress=3)


def _load_base(domain: str) -> dict:
    p = DATA_DIR / f"base_{domain}.joblib"
    if not p.exists():
        raise FileNotFoundError("Données d'origine absentes : lancez python train_model.py")
    return joblib.load(p)


# ----------------------------------------------------------------------------- registre / archive
def registry() -> list[dict]:
    return json.loads(REGISTRY.read_text()) if REGISTRY.exists() else []


def _log(event: dict):
    MODELS_DIR.mkdir(exist_ok=True)
    REGISTRY.write_text(json.dumps(registry() + [event], indent=2, default=float))


def _reload_models():
    from core import inference
    inference._CACHE.clear()


def _acc(model, X, y) -> float:
    return float((model.predict(X) == np.asarray(y)).mean()) if len(y) else float("nan")


# ----------------------------------------------------------------------------- ré-entraînement contrôlé
def retrain(domain: str, min_feedback: int = 20, feedback_weight: float = 5.0, holdout_frac: float = 0.3,
            tolerance: float = 0.01, seed: int = SEED, deploy: bool = True) -> dict:
    """Candidat = origine + retours (pondérés). Déployé seulement s'il bat le modèle actuel sur des retours mis de côté
    sans perdre plus de `tolerance` sur le jeu de test d'origine."""
    cfg = CFG[domain]
    fb = load_feedback(domain)
    if len(fb) < min_feedback:
        return {"status": "insuffisant", "n_feedback": len(fb), "needed": min_feedback,
                "message": f"Il faut au moins {min_feedback} retours ({len(fb)} enregistrés)."}
    base = _load_base(domain)
    bundle = joblib.load(MODELS_DIR / f"{cfg['model']}.joblib")
    cur = bundle["model"]
    Xb, yb, tr, te = base["X"][cfg["features"]], base["y"], base["tr"], base["te"]
    Xf, yf = _feature_matrix(fb, domain), fb["confirmed"].values

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(fb))
    n_hold = max(int(round(len(fb) * holdout_frac)), 5)
    hold, learn = order[:n_hold], order[n_hold:]

    def fit(Xl, yl):
        X = pd.concat([Xb.iloc[tr], Xl], ignore_index=True)
        y = np.concatenate([yb[tr], yl])
        w = np.concatenate([np.ones(len(tr)), np.full(len(yl), feedback_weight)])
        return RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced", n_jobs=-1,
                                      random_state=seed).fit(X, y, sample_weight=w)

    cand = fit(Xf.iloc[learn], yf[learn])
    r = {"domain": domain, "n_feedback": len(fb), "n_holdout": int(n_hold),
         "holdout_acc_before": _acc(cur, Xf.iloc[hold], yf[hold]), "holdout_acc_after": _acc(cand, Xf.iloc[hold], yf[hold]),
         "base_acc_before": _acc(cur, Xb.iloc[te], yb[te]), "base_acc_after": _acc(cand, Xb.iloc[te], yb[te])}
    better = r["holdout_acc_after"] > r["holdout_acc_before"]
    no_regression = r["base_acc_after"] >= r["base_acc_before"] - tolerance
    r["promoted"] = bool(better and no_regression)
    r["status"] = "déployé" if (r["promoted"] and deploy) else ("candidat meilleur (non déployé)" if r["promoted"] else "refusé")
    r["reason"] = ("amélioration sur les retours, sans régression" if r["promoted"] else
                   "pas d'amélioration sur les retours mis de côté" if not better else "régression sur le jeu de test d'origine")
    if r["promoted"] and deploy:
        final = fit(Xf, yf)                       # le modèle déployé apprend aussi des retours mis de côté
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        shutil.copy2(MODELS_DIR / f"{cfg['model']}.joblib", ARCHIVE / f"{cfg['model']}_{stamp}.joblib")
        joblib.dump({**bundle, "model": final, "classes": list(final.classes_), "version": stamp,
                     "trained_on_feedback": int(len(fb))}, MODELS_DIR / f"{cfg['model']}.joblib", compress=3)
        _reload_models()
        r["version"] = stamp
    if domain == "copper" and deploy:
        r["distance"] = _retrain_distance(fb, Xf, base, seed, feedback_weight, tolerance)
    _log({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **{k: v for k, v in r.items()}})
    return r


def _retrain_distance(fb, Xf, base, seed, weight, tolerance) -> dict:
    """Distance du défaut : même principe, sur les retours qui contiennent la distance réellement trouvée."""
    has = fb["distance_true"].notna().values
    if has.sum() < 12 or base["dist"] is None:
        return {"status": "insuffisant", "n": int(has.sum())}
    bundle = joblib.load(MODELS_DIR / "copper_distance.joblib")
    cur, feats = bundle["model"], bundle["features"]
    Xb, yb, te = base["X"][feats], base["dist"], base["te"]
    faulty = base["y"] != "healthy"
    tr = np.array([i for i in base["tr"] if faulty[i]])
    te_f = np.array([i for i in te if faulty[i]])
    Xd, yd = Xf[has][feats].reset_index(drop=True), fb["distance_true"].values[has]
    order = np.random.default_rng(seed).permutation(len(yd))
    n_hold = max(int(round(len(yd) * 0.3)), 4)
    hold, learn = order[:n_hold], order[n_hold:]

    def fit(Xl, yl):
        return RandomForestRegressor(n_estimators=250, min_samples_leaf=2, n_jobs=-1, random_state=seed).fit(
            pd.concat([Xb.iloc[tr], Xl], ignore_index=True), np.concatenate([yb[tr], yl]),
            sample_weight=np.concatenate([np.ones(len(tr)), np.full(len(yl), weight)]))

    cand = fit(Xd.iloc[learn], yd[learn])
    mae = lambda m, X, y: float(np.abs(m.predict(X) - y).mean())
    out = {"n": int(has.sum()), "holdout_mae_before": mae(cur, Xd.iloc[hold], yd[hold]), "holdout_mae_after": mae(cand, Xd.iloc[hold], yd[hold]),
           "base_mae_before": mae(cur, Xb.iloc[te_f], yb[te_f]), "base_mae_after": mae(cand, Xb.iloc[te_f], yb[te_f])}
    ok = out["holdout_mae_after"] < out["holdout_mae_before"] and out["base_mae_after"] <= out["base_mae_before"] * (1 + 5 * tolerance)
    out["promoted"] = bool(ok)
    if ok:
        final = fit(Xd, yd)
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        shutil.copy2(MODELS_DIR / "copper_distance.joblib", ARCHIVE / f"copper_distance_{time.strftime('%Y%m%d_%H%M%S')}.joblib")
        err = np.abs(final.predict(Xb.iloc[te_f]) - yb[te_f])
        p90c = {c: float(np.percentile(err[base["y"][te_f] == c], 90)) for c in COPPER_CLASSES[1:] if (base["y"][te_f] == c).any()}
        joblib.dump({**bundle, "model": final, "mae_m": float(err.mean()), "p90_m": float(np.percentile(err, 90)), "p90_by_class": p90c},
                    MODELS_DIR / "copper_distance.joblib", compress=3)
        _reload_models()
    return out


def rollback(domain: str) -> dict:
    """Restaure la dernière version archivée du modèle de défaut du domaine."""
    name = CFG[domain]["model"]
    files = sorted(ARCHIVE.glob(f"{name}_*.joblib")) if ARCHIVE.exists() else []
    if not files:
        return {"status": "rien à restaurer"}
    last = files[-1]
    shutil.copy2(last, MODELS_DIR / f"{name}.joblib")
    last.unlink()
    _reload_models()
    _log({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "domain": domain, "status": "retour arrière", "restored": last.name})
    return {"status": "restauré", "file": last.name}


# ----------------------------------------------------------------------------- retours terrain SIMULÉS (démonstration / tests)
def simulate_field_feedback(domain: str, n: int = 100, drift: float = 2.5, seed: int = 123) -> int:
    """Génère `n` retours avec une dérive de mesure (le terrain diffère de la simulation d'entraînement).
    Étiquetés « simulés » : pour démontrer et tester la boucle, jamais pour entraîner en production."""
    from core import inference
    from core.features import analyze_otdr, copper_feature_row, COPPER_BASE
    import data_generator as G
    rng, classes = np.random.default_rng(seed), CFG[domain]["classes"]
    for i in range(n):
        label = classes[i % len(classes)]
        if domain == "copper":
            rec, tr = G.simulate_copper_line(label, rng)
            m = {k: rec[k] for k in COPPER_BASE}
            m["impedance_ohm"] *= 1 + 0.10 * drift; m["noise_floor_dbm_hz"] += 3 * drift
            m["voltage_mean_v"] -= 0.8 * drift; m["voltage_min_v"] -= 0.8 * drift
            m["snr_db"] -= 2 * drift; m["crc_errors_per_min"] *= 1 + 0.5 * drift
            tr = tr * (1 - 0.3 * drift) + rng.normal(0, 0.012 * drift, len(tr))
            res = inference.analyze_copper({**m, "tdr": tr.tolist()})
            feats, dist_t = res["features"], (None if label == "healthy" else float(rec["fault_distance_m"]))
        elif domain == "fiber":
            rec, tr = G.simulate_fiber_link(label, rng)
            res = inference.analyze_fiber({"otdr": (tr + rng.normal(0, 0.03 * drift, len(tr))).tolist(), "expected_length_m": rec["expected_length_m"],
                                           "lambda_ratio": rec["lambda_ratio"], "rx_power_dbm": rec["rx_power_dbm"] - 1.0 * drift,
                                           "tx_power_dbm": rec["tx_power_dbm"], "fiber_att_db_km": rec["fiber_att_db_km"] * (1 + 0.08 * drift)})
            feats, dist_t = res["features"], None
        else:
            k = G.simulate_radio_kpi(label, 1, rng)
            k["rsrp_dbm"] -= 2.0 * drift; k["sinr_db"] -= 1.0 * drift; k["bler_pct"] = k["bler_pct"] * (1 + 0.25 * drift)
            res = inference.analyze_radio(k)
            feats, dist_t = res["features"], None
        add_feedback(domain, feats, res["fault"]["label"], res["fault"]["confidence"], label, res.get("distance_m"), dist_t,
                     tech="simulation", note=f"dérive {drift}", simulated=True)
    return n
