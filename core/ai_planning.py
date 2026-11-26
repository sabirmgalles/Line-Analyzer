"""IA de planification.

1. Modèle de coût génie civil (surrogate) : un Gradient Boosting appris sur des centaines de plans FTTH simulés
   (réseau de rues fictif + métré `core.civil`). Il prédit en une milliseconde le coût par foyer et la surface cassée
   à partir de quelques paramètres (densité, rayon, roche, aérien, trottoir…), avec un intervalle à 80 % (calibration conforme).
   Utile pour comparer des scénarios sans relancer le routage complet.
2. Optimisation des sites radio : recouvrement glouton + élagage sur une carte de demande réaliste (quartiers denses),
   comparé au maillage hexagonal régulier.

Les deux restent des modèles d'avant-projet : le surrogate ne vaut que dans le domaine simulé (voir FEATURE_RANGES).
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance

from config import MODELS_DIR
from core import civil, planning as P
from core.osm import RoadGraph

CIVIL_FEATURES = ["radius_m", "density_km2", "block_m", "rock_pct", "aerial_share", "sw_feeder", "sw_dist", "sw_drop",
                  "chamber_spacing_m"]
FEATURE_RANGES = {"radius_m": (500, 900), "density_km2": (200, 1500), "block_m": (90, 140), "rock_pct": (0, 60),
                  "aerial_share": (0, 0.6), "sw_feeder": (0, 1), "sw_dist": (0, 1), "sw_drop": (0, 1),
                  "chamber_spacing_m": (100, 250)}
MODEL_NAME = "civil_surrogate"


# ----------------------------------------------------------------------------- 1. surrogate coût génie civil
def simulate_civil_scenarios(n_plans: int = 40, draws: int = 8, seed: int = 0) -> pd.DataFrame:
    """Plans FTTH simulés (rues fictives) × tirages d'hypothèses de génie civil -> coût et casses."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_plans):
        R = float(rng.uniform(*FEATURE_RANGES["radius_m"]))
        dens = float(rng.uniform(*FEATURE_RANGES["density_km2"]))
        block = float(rng.uniform(*FEATURE_RANGES["block_m"]))
        g = RoadGraph.synthetic_grid(R * 1.1, block, seed=int(rng.integers(1_000_000)))
        homes = P.synthetic_homes(R, dens, seed=int(rng.integers(1_000_000)))
        plan = P.plan_ftth(homes, (0.0, 0.0), router=g)
        for _ in range(draws):
            f = {n: float(rng.uniform(*FEATURE_RANGES[n])) for n in
                 ("rock_pct", "aerial_share", "sw_feeder", "sw_dist", "sw_drop", "chamber_spacing_m")}
            r = civil.estimate_civil(plan, g, {
                "rock_pct": f["rock_pct"], "aerial_share_drop": f["aerial_share"], "chamber_spacing_m": f["chamber_spacing_m"],
                "sidewalk_share": {"feeder": f["sw_feeder"], "dist": f["sw_dist"], "drop": f["sw_drop"]}})
            rows.append({"plan": k, "radius_m": R, "density_km2": dens, "block_m": block, **f,
                         "n_homes": plan["n_homes"], "cost_per_home": r["cost_per_home_tnd"],
                         "broken_m2_per_home": r["broken_m2"] / plan["n_homes"],
                         "trench_m_per_home": r["trench_m"] / plan["n_homes"]})
    return pd.DataFrame(rows)


def train_civil_surrogate(n_plans: int = 40, draws: int = 8, seed: int = 0, save: bool = True) -> dict:
    df = simulate_civil_scenarios(n_plans, draws, seed)
    rng = np.random.default_rng(seed)
    plans = rng.permutation(df["plan"].unique())
    n20 = max(len(plans) // 5, 1)
    te_p, cal_p = set(plans[:n20]), set(plans[n20:2 * n20])      # test / calibration, découpés par plan (pas de fuite)
    te, cal = df["plan"].isin(te_p).values, df["plan"].isin(cal_p).values
    tr = ~(te | cal)
    X, y = df[CIVIL_FEATURES], np.log(df["cost_per_home"])
    kw = dict(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=seed)
    mid = GradientBoostingRegressor(**kw).fit(X[tr], y[tr])
    # intervalle 80 % par calibration conforme : quantile des erreurs (en log) sur des plans jamais vus à l'entraînement
    q80 = float(np.quantile(np.abs(mid.predict(X[cal]) - y[cal]), 0.8))
    pred = np.exp(mid.predict(X[te]))
    true = df["cost_per_home"].values[te]
    mape = float(np.mean(np.abs(pred - true) / true) * 100)
    r2 = float(1 - ((pred - true) ** 2).sum() / ((true - true.mean()) ** 2).sum())
    cover = float((np.abs(np.log(pred) - np.log(true)) <= q80).mean())
    imp = permutation_importance(mid, X[te], y[te], n_repeats=8, random_state=seed)
    importance = dict(sorted(zip(CIVIL_FEATURES, map(float, imp.importances_mean)), key=lambda kv: -kv[1]))
    # casses par foyer : second modèle
    yb = np.log(df["broken_m2_per_home"] + 1e-6)
    brk = GradientBoostingRegressor(**kw).fit(X[tr], yb[tr])
    pb, tb = np.exp(brk.predict(X[te])), df["broken_m2_per_home"].values[te]
    bundle = {"cost": mid, "q80_log": q80, "broken": brk, "features": CIVIL_FEATURES,
              "metrics": {"mape_pct": mape, "r2": r2, "interval_80_coverage": cover,
                          "broken_mape_pct": float(np.mean(np.abs(pb - tb) / tb) * 100),
                          "n_train": int(tr.sum()), "n_test": int(te.sum()), "importance": importance},
              "ranges": FEATURE_RANGES}
    if save:
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(bundle, MODELS_DIR / f"{MODEL_NAME}.joblib")
    return bundle["metrics"]


_CACHE: dict = {}


def surrogate_ready() -> bool:
    return (MODELS_DIR / f"{MODEL_NAME}.joblib").exists()


def predict_civil(radius_m: float, density_km2: float, rock_pct: float = 10, aerial_share: float = 0.0,
                  sw_feeder: float = 0.3, sw_dist: float = 0.5, sw_drop: float = 0.8, block_m: float = 110,
                  chamber_spacing_m: float = 150) -> dict:
    """Coût génie civil prédit (TND) + intervalle à 80 %, et hors-domaine signalé."""
    if MODEL_NAME not in _CACHE:
        _CACHE[MODEL_NAME] = joblib.load(MODELS_DIR / f"{MODEL_NAME}.joblib")
    b = _CACHE[MODEL_NAME]
    v = dict(radius_m=radius_m, density_km2=density_km2, block_m=block_m, rock_pct=rock_pct, aerial_share=aerial_share,
             sw_feeder=sw_feeder, sw_dist=sw_dist, sw_drop=sw_drop, chamber_spacing_m=chamber_spacing_m)
    X = pd.DataFrame([v])[b["features"]]
    homes = np.pi * (radius_m / 1000) ** 2 * density_km2
    c = float(np.exp(b["cost"].predict(X)[0]))
    lo, hi = c * float(np.exp(-b["q80_log"])), c * float(np.exp(b["q80_log"]))
    out_of = [k for k, (a, z) in b["ranges"].items() if not a <= v[k] <= z]
    return {"homes": homes, "cost_per_home_tnd": c, "interval_tnd_per_home": [min(lo, c), max(hi, c)],
            "total_tnd": c * homes, "total_interval_tnd": [min(lo, c) * homes, max(hi, c) * homes],
            "broken_m2": float(np.exp(b["broken"].predict(X)[0])) * homes, "out_of_domain": out_of,
            "metrics": b["metrics"]}


# ----------------------------------------------------------------------------- 2. optimisation des sites radio
def demand_map(radius_m: float, n_points: int = 4000, n_hot: int = 6, seed: int = 0):
    """Demande synthétique : quelques quartiers denses + fond diffus (poids = abonnés relatifs)."""
    rng = np.random.default_rng(seed)
    hot = rng.uniform(-0.7, 0.7, (n_hot, 2)) * radius_m
    pts, w = [], []
    while len(pts) < n_points:
        if rng.random() < 0.75:
            c = hot[rng.integers(n_hot)]
            p, wt = c + rng.normal(0, radius_m * 0.12, 2), 3.0
        else:
            p, wt = rng.uniform(-radius_m, radius_m, 2), 1.0
        if np.hypot(*p) <= radius_m:
            pts.append(p); w.append(wt)
    return np.array(pts), np.array(w)


def optimize_sites(pts: np.ndarray, w: np.ndarray, cell_r_m: float, radius_m: float, target: float = 0.98,
                   spacing_div: float = 3.0) -> np.ndarray:
    """Recouvrement glouton pondéré (candidats sur grille fine), puis élagage des sites redondants."""
    step = max(cell_r_m / spacing_div, radius_m / 45.0)      # borne le nombre de candidats sur les grandes zones
    g = np.arange(-radius_m - cell_r_m, radius_m + cell_r_m + step, step)
    cand = np.array([(x, y) for x in g for y in g if np.hypot(x, y) <= radius_m + cell_r_m * 0.5])
    tree = cKDTree(pts)
    cover = [np.array(tree.query_ball_point(c, cell_r_m), dtype=int) for c in cand]
    covered = np.zeros(len(pts), dtype=int)
    chosen, total = [], w.sum()
    while w[covered > 0].sum() < target * total:
        gain = np.array([w[ix][covered[ix] == 0].sum() if len(ix) else 0.0 for ix in cover])
        j = int(gain.argmax())
        if gain[j] <= 0:
            break
        chosen.append(j)
        covered[cover[j]] += 1
    for j in list(chosen):                      # élagage : retirer un site si la cible reste atteinte
        covered[cover[j]] -= 1
        if w[covered > 0].sum() >= target * total:
            chosen.remove(j)
        else:
            covered[cover[j]] += 1
    return cand[chosen]


def weighted_coverage(sites: np.ndarray, pts: np.ndarray, w: np.ndarray, cell_r_m: float) -> float:
    d, _ = cKDTree(sites).query(pts)
    return float(w[d <= cell_r_m].sum() / w.sum())


def compare_site_plans(radius_m: float, cell_r_m: float, n_hex: int, target: float = 0.98, seed: int = 0) -> dict:
    """Optimisation vs maillage hexagonal : nombre de sites et part de la demande couverte."""
    pts, w = demand_map(radius_m, seed=seed)
    opt = optimize_sites(pts, w, cell_r_m, radius_m, target)
    hexs = P.hex_layout(radius_m, n_hex)
    c_opt, c_hex = weighted_coverage(opt, pts, w, cell_r_m), weighted_coverage(hexs, pts, w, cell_r_m)
    return {"sites_opt": len(opt), "sites_hex": len(hexs), "cov_opt": c_opt, "cov_hex": c_hex,
            "saving_pct": 100.0 * (1 - len(opt) / len(hexs)), "opt_xy": opt, "hex_xy": hexs, "pts": pts, "w": w}
