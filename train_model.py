"""Entraînement des modèles IA d'AI Line Analyzer.

Modèles produits dans ./models (format joblib) :
  copper_fault      RandomForestClassifier  -> type de défaut cuivre (6 classes)
  copper_distance   RandomForestRegressor   -> distance du défaut (m)
  copper_anomaly    IsolationForest         -> anomalie / écoute (entraîné sur lignes SAINES)
  modulation        RandomForestClassifier  -> profil ADSL2+ / VDSL2 / G.fast / Aucun
  history_class     RandomForestClassifier  -> stable / dégradation / coupure imminente
  history_ttf       RandomForestRegressor   -> jours avant coupure
  fiber_fault       RandomForestClassifier  -> défaut fibre (6 classes)
  radio_fault       RandomForestClassifier  -> santé / sécurité radio 4G-5G-6G
  fiberhist_class   RandomForestClassifier  -> tendance de la puissance reçue ONU (fibre)
  fiberhist_ttf     RandomForestRegressor   -> jours avant d'atteindre la sensibilité ONU

Usage :  python train_model.py            (génère les données si absentes)
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (classification_report, confusion_matrix, mean_absolute_error)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import (COPPER_CLASSES, DATA_DIR, FIBER_CLASSES, FIBER_MAX_RANGE_M, MODELS_DIR,
                    RADIO_CLASSES, SEED)
from core.features import (COPPER_FEATURES, FIBER_FEATURES, FIBER_HISTORY_FEATURES, HISTORY_FEATURES,
                           MODULATION_FEATURES, RADIO_FEATURES, TSCM_FEATURES, analyze_otdr,
                           build_copper_features, history_features, rx_history_features)
from core import feedback
from core.physics import best_profile

METRICS: dict = {}


def _rf_clf(**kw):
    return RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced",
                                  n_jobs=-1, random_state=SEED, **kw)


def _report(name, y_true, y_pred, labels):
    acc = float((np.asarray(y_true) == np.asarray(y_pred)).mean())
    print(f"\n=== {name} === accuracy = {acc:.3f}")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0, digits=3))
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels)
    print("Matrice de confusion (lignes = réel) :\n", cm.to_string())
    return acc


def _save(name, bundle):
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, MODELS_DIR / f"{name}.joblib", compress=3)


def ensure_data():
    needed = ["copper_lines.csv", "copper_tdr.npy", "history.npz", "fiber_links.csv",
              "fiber_otdr.npy", "radio_kpi.csv", "fiber_history.npz", "fiber_history_labels.csv"]
    if not all((DATA_DIR / f).exists() for f in needed):
        print("Données absentes -> génération automatique...")
        import subprocess, sys
        subprocess.run([sys.executable, str(DATA_DIR.parent / "data_generator.py")], check=True)


# ----------------------------------------------------------------------------- cuivre
def train_copper():
    df = pd.read_csv(DATA_DIR / "copper_lines.csv")
    tdr = np.load(DATA_DIR / "copper_tdr.npy")
    X = build_copper_features(df, tdr)
    y = df["label"].values
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)

    # 1) classification du défaut
    feedback.save_base("copper", X, y, tr, te, df["fault_distance_m"].values)
    clf = _rf_clf().fit(X.iloc[tr], y[tr])
    METRICS["copper_fault_acc"] = _report("Défaut cuivre", y[te], clf.predict(X.iloc[te]), COPPER_CLASSES)
    imp = pd.Series(clf.feature_importances_, index=COPPER_FEATURES).sort_values(ascending=False)
    print("\nTop features :", ", ".join(f"{k}={v:.3f}" for k, v in imp.head(6).items()))
    _save("copper_fault", {"model": clf, "features": COPPER_FEATURES, "classes": list(clf.classes_)})

    # 2) distance du défaut (lignes en défaut uniquement)
    faulty = df["label"].ne("healthy").values
    trf = np.array([i for i in tr if faulty[i]])
    tef = np.array([i for i in te if faulty[i]])
    reg = RandomForestRegressor(n_estimators=250, min_samples_leaf=2, n_jobs=-1, random_state=SEED)
    reg.fit(X.iloc[trf], df["fault_distance_m"].values[trf])
    pred = reg.predict(X.iloc[tef])
    err = np.abs(pred - df["fault_distance_m"].values[tef])
    METRICS["distance_mae_m"] = float(err.mean())
    METRICS["distance_p90_m"] = float(np.percentile(err, 90))
    print(f"\n=== Distance du défaut === MAE = {err.mean():.1f} m | P90 = {np.percentile(err, 90):.1f} m")
    # incertitude (P90) par type de défaut : l'écoute discrète est moins précise qu'une coupure
    p90_by_class = {}
    for lab in COPPER_CLASSES[1:]:
        m = y[tef] == lab
        p90_by_class[lab] = float(np.percentile(err[m], 90))
        print(f"  {lab:<14} MAE = {err[m].mean():6.1f} m | P90 = {p90_by_class[lab]:6.1f} m")
    METRICS["distance_p90_by_class_m"] = p90_by_class
    _save("copper_distance", {"model": reg, "features": COPPER_FEATURES, "mae_m": METRICS["distance_mae_m"],
                              "p90_m": METRICS["distance_p90_m"], "p90_by_class": p90_by_class})

    # 3) détection d'anomalies (TSCM) : on n'apprend QUE le comportement normal
    healthy_tr = np.array([i for i in tr if y[i] == "healthy"])
    iso = make_pipeline(StandardScaler(),
                        IsolationForest(n_estimators=300, contamination=0.02, random_state=SEED, n_jobs=-1))
    iso.fit(X.iloc[healthy_tr][TSCM_FEATURES])
    sc_h = iso.score_samples(X.iloc[healthy_tr][TSCM_FEATURES])
    thr, scale = float(np.percentile(sc_h, 2)), float(np.std(sc_h))
    sc_te = iso.score_samples(X.iloc[te][TSCM_FEATURES])
    flagged = sc_te < thr
    for lab in COPPER_CLASSES:
        m = y[te] == lab
        print(f"  IsolationForest : {lab:<14} anomalies détectées = {flagged[m].mean():.1%}")
    METRICS["iforest_wiretap_recall"] = float(flagged[y[te] == "wiretap"].mean())
    METRICS["iforest_healthy_fpr"] = float(flagged[y[te] == "healthy"].mean())
    _save("copper_anomaly", {"model": iso, "features": TSCM_FEATURES, "threshold": thr, "scale": scale})

    # 4) recommandation de modulation (étiquettes issues du modèle physique)
    ok = df["label"].isin(["healthy", "oxidation", "bad_splice", "wiretap"]).values
    prof = np.array([best_profile(r.attenuation_db, r.snr_db, r.crc_errors_per_min)[0]
                     for r in df.itertuples()])
    Xm = df[MODULATION_FEATURES]
    mtr = np.array([i for i in tr if ok[i]]); mte = np.array([i for i in te if ok[i]])
    mod = _rf_clf().fit(Xm.iloc[mtr], prof[mtr])
    print("\nDistribution des profils :", dict(pd.Series(prof[ok]).value_counts()))
    METRICS["modulation_acc"] = _report("Profil de modulation", prof[mte], mod.predict(Xm.iloc[mte]),
                                        ["ADSL2+", "VDSL2", "G.fast", "Aucun"])
    _save("modulation", {"model": mod, "features": MODULATION_FEATURES, "classes": list(mod.classes_)})


# ----------------------------------------------------------------------------- historique
def train_history():
    z = np.load(DATA_DIR / "history.npz")
    meta = pd.read_csv(DATA_DIR / "history_labels.csv")
    X = pd.DataFrame([history_features(z["snr"][i], z["att"][i], z["crc"][i]) for i in range(len(meta))])[HISTORY_FEATURES]
    y = meta["label"].values
    idx = np.arange(len(meta))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    clf = _rf_clf().fit(X.iloc[tr], y[tr])
    METRICS["history_acc"] = _report("Maintenance prédictive", y[te], clf.predict(X.iloc[te]),
                                     ["stable", "degrading", "imminent"])
    _save("history_class", {"model": clf, "features": HISTORY_FEATURES, "classes": list(clf.classes_)})

    ttf = np.log1p(meta["ttf_days"].values)
    reg = RandomForestRegressor(n_estimators=250, min_samples_leaf=2, n_jobs=-1, random_state=SEED)
    reg.fit(X.iloc[tr], ttf[tr])
    pred = np.expm1(reg.predict(X.iloc[te]))
    real = meta["ttf_days"].values[te]
    near = real <= 90
    mae = float(np.abs(pred[near] - real[near]).mean())
    METRICS["ttf_mae_days_under90"] = mae
    print(f"\n=== Jours avant coupure (lignes à TTF <= 90 j) === MAE = {mae:.1f} jours")
    _save("history_ttf", {"model": reg, "features": HISTORY_FEATURES, "mae_days": mae})


# ----------------------------------------------------------------------------- fibre
def train_fiber():
    df = pd.read_csv(DATA_DIR / "fiber_links.csv")
    tr_ = np.load(DATA_DIR / "fiber_otdr.npy")
    res = [analyze_otdr(tr_[i], r.expected_length_m, r.fiber_att_db_km, r.lambda_ratio,
                        r.rx_power_dbm, r.tx_power_dbm) for i, r in enumerate(df.itertuples())]
    X = pd.DataFrame([r["features"] for r in res])[FIBER_FEATURES]
    y = df["label"].values
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    feedback.save_base("fiber", X, y, tr, te)
    clf = _rf_clf().fit(X.iloc[tr], y[tr])
    METRICS["fiber_acc"] = _report("Défaut fibre (OTDR)", y[te], clf.predict(X.iloc[te]), FIBER_CLASSES)
    # précision de localisation (heuristique : pire événement, ou fin de trace pour une rupture)
    errs = []
    for i in te:
        if y[i] == "healthy":
            continue
        pos = res[i]["end_dist_m"] if y[i] == "fiber_break" else res[i]["worst_event_pos_m"]
        if pos is not None:
            errs.append(abs(pos - df["fault_distance_m"].iloc[i]))
    METRICS["fiber_location_mae_m"] = float(np.mean(errs))
    print(f"\nLocalisation fibre : MAE = {np.mean(errs):.0f} m (médiane {np.median(errs):.0f} m)")
    _save("fiber_fault", {"model": clf, "features": FIBER_FEATURES, "classes": list(clf.classes_)})


# ----------------------------------------------------------------------------- radio
def train_radio():
    df = pd.read_csv(DATA_DIR / "radio_kpi.csv")
    X, y = df[RADIO_FEATURES], df["label"].values
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    feedback.save_base("radio", X, y, tr, te)
    clf = _rf_clf().fit(X.iloc[tr], y[tr])
    METRICS["radio_acc"] = _report("Diagnostic radio 4G/5G/6G", y[te], clf.predict(X.iloc[te]), RADIO_CLASSES)
    _save("radio_fault", {"model": clf, "features": RADIO_FEATURES, "classes": list(clf.classes_)})


def train_fiber_history():
    """Maintenance prédictive fibre : puissance reçue ONU sur 30 jours."""
    rx = np.load(DATA_DIR / "fiber_history.npz")["rx"]
    meta = pd.read_csv(DATA_DIR / "fiber_history_labels.csv")
    X = pd.DataFrame([rx_history_features(r) for r in rx])[FIBER_HISTORY_FEATURES]
    y = meta["label"].values
    idx = np.arange(len(meta))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    clf = _rf_clf().fit(X.iloc[tr], y[tr])
    METRICS["fiber_history_acc"] = _report("Maintenance prédictive fibre (Rx ONU)", y[te], clf.predict(X.iloc[te]),
                                           ["stable", "degrading", "imminent"])
    # le saut brutal suivi d'un palier ne doit PAS déclencher d'alerte
    step = (meta["kind"].values[te] == "step")
    METRICS["fiber_history_step_false_alarm"] = float((clf.predict(X.iloc[te][step]) != "stable").mean())
    print(f"\nSauts brutaux classés à tort en dégradation : {METRICS['fiber_history_step_false_alarm']:.1%}")
    _save("fiberhist_class", {"model": clf, "features": FIBER_HISTORY_FEATURES, "classes": list(clf.classes_)})

    reg = RandomForestRegressor(n_estimators=250, min_samples_leaf=2, n_jobs=-1, random_state=SEED)
    reg.fit(X.iloc[tr], np.log1p(meta["ttf_days"].values[tr]))
    pred, real = np.expm1(reg.predict(X.iloc[te])), meta["ttf_days"].values[te]
    near = real <= 90
    mae = float(np.abs(pred[near] - real[near]).mean())
    METRICS["fiber_ttf_mae_days_under90"] = mae
    print(f"=== Jours avant seuil Rx (TTF <= 90 j) === MAE = {mae:.1f} jours")
    _save("fiberhist_ttf", {"model": reg, "features": FIBER_HISTORY_FEATURES, "mae_days": mae})


def train_deep():
    """CNN 1D (TDR) et LSTM (historique), comparés aux Random Forest sur les mêmes jeux de test."""
    from core import deep
    if not deep.AVAILABLE:
        print("\nPyTorch absent : deep learning ignoré (pip install torch).")
        return
    df = pd.read_csv(DATA_DIR / "copper_lines.csv")
    tdr = np.load(DATA_DIR / "copper_tdr.npy")
    tr, te = train_test_split(np.arange(len(df)), test_size=0.2, stratify=df["label"].values, random_state=SEED)
    r = deep.train_tdr(tdr, df["label"].values, df["fault_distance_m"].values, tr, te)
    METRICS["cnn_tdr_acc"], METRICS["cnn_tdr_dist_mae_m"], METRICS["cnn_tdr_dist_p90_m"] = r["acc"], r["dist_mae_m"], r["dist_p90_m"]
    print(f"\n=== CNN 1D sur courbe TDR brute === précision {r['acc']:.1%} | distance MAE {r['dist_mae_m']:.1f} m "
          f"(Random Forest : {METRICS['copper_fault_acc']:.1%} | {METRICS['distance_mae_m']:.1f} m)")
    z = np.load(DATA_DIR / "history.npz")
    meta = pd.read_csv(DATA_DIR / "history_labels.csv")
    tr, te = train_test_split(np.arange(len(meta)), test_size=0.2, stratify=meta["label"].values, random_state=SEED)
    r = deep.train_history(z["snr"], z["att"], z["crc"], meta["label"].values, meta["ttf_days"].values, tr, te)
    METRICS["lstm_history_acc"], METRICS["lstm_ttf_mae_days_under90"] = r["acc"], r["ttf_mae_days_under90"]
    print(f"=== LSTM historique 30 jours === précision {r['acc']:.1%} | TTF MAE {r['ttf_mae_days_under90']:.1f} j "
          f"(Random Forest : {METRICS['history_acc']:.1%} | {METRICS['ttf_mae_days_under90']:.1f} j)")


def train_ai_planning():
    from core import ai_planning
    m = ai_planning.train_civil_surrogate(120, 8)
    METRICS["civil_surrogate"] = {k: v for k, v in m.items() if k != "importance"}
    print(f"\n=== Modèle de coût génie civil (Gradient Boosting) === MAPE {m['mape_pct']:.1f} % | R² {m['r2']:.2f} | "
          f"intervalle 80 % couvre {m['interval_80_coverage']:.0%} des plans de test")


def main():
    t0 = time.time()
    ensure_data()
    train_copper()
    train_history()
    train_fiber()
    train_radio()
    train_fiber_history()
    train_deep()
    train_ai_planning()
    MODELS_DIR.mkdir(exist_ok=True)
    (MODELS_DIR / "metrics.json").write_text(json.dumps(METRICS, indent=2))
    print(f"\nTerminé en {time.time() - t0:.0f}s -> modèles dans {MODELS_DIR}")
    print(json.dumps(METRICS, indent=2))


if __name__ == "__main__":
    main()
