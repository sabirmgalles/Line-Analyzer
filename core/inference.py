"""Couche d'inférence : transforme des mesures brutes en diagnostic exploitable.

Toutes les fonctions prennent/retournent des dict JSON-compatibles, pour être
appelées indifféremment par l'API FastAPI ou directement par Streamlit.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from config import (FIBER_MAX_RANGE_M, FIBER_RX_FAIL_DBM, LABELS_FR, MODELS_DIR, REF_FREQ_MHZ,
                    RADIO_TECHS, SNR_FAIL_DB, TDR_N_POINTS)
from core.features import (COPPER_BASE, COPPER_DEFAULTS, analyze_otdr, copper_feature_row,
                           history_features, linear_time_to_failure, rx_history_features,
                           rx_time_to_failure, tdr_axis)
from core import deep
from core.physics import NOMINAL_VOLTAGE_V, best_profile, snr_deficit_db

_CACHE: dict = {}


def load_model(name: str) -> dict:
    """Charge (et met en cache) un bundle joblib produit par train_model.py."""
    if name not in _CACHE:
        path = MODELS_DIR / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Modèle '{name}' introuvable. Lancez : python train_model.py")
        _CACHE[name] = joblib.load(path)
    return _CACHE[name]


def models_ready() -> bool:
    return all((MODELS_DIR / f"{n}.joblib").exists() for n in
               ["copper_fault", "copper_distance", "copper_anomaly", "modulation",
                "history_class", "history_ttf", "fiber_fault", "radio_fault",
                "fiberhist_class", "fiberhist_ttf"])


def _proba(bundle: dict, X: pd.DataFrame) -> dict:
    p = bundle["model"].predict_proba(X[bundle["features"]])[0]
    return {c: float(v) for c, v in zip(bundle["classes"], p)}


def _alert(level: str, title: str, detail: str = "") -> dict:
    return {"level": level, "title": title, "detail": detail}   # level: critical | warning | info


def _resample(trace, n: int = TDR_N_POINTS) -> np.ndarray:
    """Ré-échantillonne une courbe TDR sur l'axe standard (0 -> portée max)."""
    trace = np.asarray(trace, dtype=float)
    if len(trace) == n:
        return trace
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(trace)), trace)


# =============================================================================
# CUIVRE
# =============================================================================
def analyze_copper(payload: dict) -> dict:
    """Analyse complète d'une ligne cuivre : défaut, distance, TSCM, modulation, tendance."""
    trace = _resample(payload["tdr"])
    row = copper_feature_row(payload, trace)
    X = pd.DataFrame([row])
    alerts: list[dict] = []

    # --- Mission 1 : type de défaut + distance
    fb = load_model("copper_fault")
    proba = _proba(fb, X)
    label = max(proba, key=proba.get)
    conf = proba[label]
    distance, dist_err = None, None
    if label != "healthy":
        rb = load_model("copper_distance")
        distance = float(rb["model"].predict(X[rb["features"]])[0])
        dist_err = float(rb.get("p90_by_class", {}).get(label, rb["p90_m"]))

    # --- Mission 4 : sécurité / TSCM (Isolation Forest + probabilité "wiretap")
    ab = load_model("copper_anomaly")
    s = float(ab["model"].score_samples(X[ab["features"]])[0])
    if_risk = float(1.0 / (1.0 + np.exp(-3.0 * (ab["threshold"] - s) / max(ab["scale"], 1e-6))))
    p_tap = proba.get("wiretap", 0.0)
    risk = p_tap if label in ("open_cut", "short_circuit") else max(p_tap, 0.5 * p_tap + 0.5 * if_risk)
    risk_level = "ÉLEVÉ" if risk >= 0.65 else "MOYEN" if risk >= 0.35 else "FAIBLE"
    indicators = []
    if row["voltage_drop_v"] > 1.5 and label not in ("short_circuit",):
        indicators.append(f"Chute de tension de {row['voltage_drop_v']:.1f} V sous la nominale ({NOMINAL_VOLTAGE_V:.0f} V)")
    if row["voltage_max_step_v"] > 1.0:
        indicators.append(f"Saut de tension brutal de {row['voltage_max_step_v']:.1f} V")
    if row["noise_floor_dbm_hz"] > -128:
        indicators.append(f"Plancher de bruit anormal ({row['noise_floor_dbm_hz']:.0f} dBm/Hz)")
    if row["tdr_neg_amp"] < -0.07 and label not in ("short_circuit",):
        indicators.append(f"Écho TDR négatif (dérivation) à ~{row['tdr_neg_pos_m']:.0f} m")
    if row["impedance_ohm"] < 88:
        indicators.append(f"Impédance abaissée ({row['impedance_ohm']:.0f} Ω) : charge en parallèle")
    tscm = {"risk_score": float(risk), "risk_level": risk_level, "anomaly_score": if_risk,
            "wiretap_proba": float(p_tap), "indicators": indicators,
            "suspected_position_m": float(row["tdr_neg_pos_m"]) if p_tap > 0.35 else None}

    # --- Mission 3 : profil de modulation recommandé
    mb = load_model("modulation")
    prof, rates = best_profile(row["attenuation_db"], row["snr_db"], row["crc_errors_per_min"])
    service_down = label in ("open_cut", "short_circuit")
    if service_down:
        modulation = {"recommended": "Aucun", "ml_recommended": "Aucun", "confidence": 1.0,
                      "rates_mbps": {k: 0.0 for k in rates}, "note": "Ligne hors service : réparer d'abord."}
    else:
        pm = _proba(mb, X)
        ml = max(pm, key=pm.get)
        modulation = {"recommended": ml, "ml_recommended": ml, "physics_recommended": prof,
                      "confidence": float(pm[ml]), "rates_mbps": {k: round(v, 1) for k, v in rates.items()},
                      "note": ""}

    # --- Mission 2 : maintenance prédictive (si historique fourni)
    maintenance = None
    h = payload.get("history")
    if h and len(h.get("snr_db", [])) >= 14:
        hf = pd.DataFrame([history_features(h["snr_db"], h["attenuation_db"], h["crc_errors_per_min"])])
        hb, tb = load_model("history_class"), load_model("history_ttf")
        hp = _proba(hb, hf)
        hl = max(hp, key=hp.get)
        ttf_ml = float(np.expm1(tb["model"].predict(hf[tb["features"]])[0]))
        ttf_lin = linear_time_to_failure(h["snr_db"], SNR_FAIL_DB)
        maintenance = {"label": hl, "label_fr": LABELS_FR[hl], "proba": hp,
                       "ttf_days_ml": round(min(ttf_ml, 365.0), 1), "ttf_days_linear": round(ttf_lin, 1),
                       "snr_slope_db_per_day": hf["snr_slope"].iloc[0], "att_slope_db_per_day": hf["att_slope"].iloc[0]}

    # --- second avis : deep learning (CNN 1D sur la courbe brute, LSTM sur l'historique), si PyTorch est installé
    deep_view = None
    dt = deep.predict_tdr(trace)
    if dt:
        dl = max(dt["proba"], key=dt["proba"].get)
        deep_view = {"fault": {"label": dl, "label_fr": LABELS_FR[dl], "confidence": dt["proba"][dl], "proba": dt["proba"]},
                     "distance_m": dt["distance_m"] if dl != "healthy" else None, "agrees_with_rf": dl == label,
                     "history": None}
        if h and len(h.get("snr_db", [])) == 30:
            dh = deep.predict_history(h["snr_db"], h["attenuation_db"], h["crc_errors_per_min"])
            if dh:
                hl2 = max(dh["proba"], key=dh["proba"].get)
                deep_view["history"] = {"label": hl2, "label_fr": LABELS_FR[hl2], "ttf_days": round(dh["ttf_days"], 1),
                                        "agrees_with_rf": bool(maintenance and hl2 == maintenance["label"])}

    # --- score de santé 0-100
    crc_score = float(np.clip(1.0 - np.log10(1.0 + row["crc_errors_per_min"]) / 3.0, 0, 1))
    quality = float(np.clip(1.0 - row["snr_deficit_db"] / 20.0, 0, 1))
    health = 100.0 * (0.6 * proba.get("healthy", 0.0) + 0.25 * quality + 0.15 * crc_score)
    if service_down:
        health = min(health, 10.0)
    if risk_level == "ÉLEVÉ":
        health = min(health, 55.0)
    if maintenance and maintenance["label"] == "imminent":
        health = min(health, 40.0)
    elif maintenance and maintenance["label"] == "degrading":
        health = min(health, 75.0)

    # --- alertes
    loc = f" à ~{distance:.0f} m (± {dist_err:.0f} m)" if distance is not None else ""
    if label in ("open_cut", "short_circuit"):
        alerts.append(_alert("critical", f"{LABELS_FR[label]}{loc}", "Service interrompu : intervention requise."))
    elif label in ("oxidation", "bad_splice"):
        alerts.append(_alert("warning", f"{LABELS_FR[label]}{loc}", "Dégradation du SNR / hausse des CRC : planifier une intervention."))
    if risk_level != "FAIBLE":
        pos = f" vers {tscm['suspected_position_m']:.0f} m" if tscm["suspected_position_m"] else ""
        alerts.append(_alert("critical" if risk_level == "ÉLEVÉ" else "warning",
                             f"Risque TSCM {risk_level} : branchement/écoute suspecté{pos}",
                             " ; ".join(indicators) or "Anomalie statistique détectée"))
    if maintenance and maintenance["label"] == "imminent":
        alerts.append(_alert("critical", f"Coupure probable sous ~{maintenance['ttf_days_ml']:.0f} jours",
                             f"Pente SNR {maintenance['snr_slope_db_per_day']:+.2f} dB/jour"))
    elif maintenance and maintenance["label"] == "degrading":
        alerts.append(_alert("warning", f"Dégradation progressive (~{maintenance['ttf_days_ml']:.0f} jours avant seuil)", ""))
    if row["crc_errors_per_min"] > 100 and not service_down:
        alerts.append(_alert("warning", f"Taux de CRC élevé ({row['crc_errors_per_min']:.0f}/min)", "Risque de resynchronisations."))
    if not service_down and modulation["recommended"] != "Aucun":
        alerts.append(_alert("info", f"Profil recommandé : {modulation['recommended']}",
                             f"≈ {modulation['rates_mbps'].get(modulation['recommended'], 0):.0f} Mbit/s descendant estimés"))
    if deep_view and not deep_view["agrees_with_rf"]:
        alerts.append(_alert("warning", f"Avis divergents : Random Forest « {LABELS_FR[label]} », CNN « {deep_view['fault']['label_fr']} »",
                             "Refaire la mesure ou vérifier sur le terrain avant d'intervenir."))
    if not alerts:
        alerts.append(_alert("info", "Aucune anomalie détectée", ""))

    return {"health_score": round(float(health), 1),
            "fault": {"label": label, "label_fr": LABELS_FR[label], "confidence": float(conf), "proba": proba},
            "distance_m": distance, "distance_error_p90_m": dist_err,
            "tdr_peak_m": float(row["tdr_peak_pos_m"]),
            "tscm": tscm, "modulation": modulation, "maintenance": maintenance, "alerts": alerts,
            "deep": deep_view, "features": {k: float(v) for k, v in row.items()}}


# =============================================================================
# FIBRE
# =============================================================================
def analyze_fiber(payload: dict) -> dict:
    """Analyse d'une trace OTDR : défaut, localisation, budget optique ONU."""
    trace = np.asarray(payload["otdr"], dtype=float)
    res = analyze_otdr(trace, payload["expected_length_m"], payload.get("fiber_att_db_km", 0.22),
                       payload.get("lambda_ratio", 1.0), payload.get("rx_power_dbm"),
                       payload.get("tx_power_dbm", 3.0))
    X = pd.DataFrame([res["features"]])
    fb = load_model("fiber_fault")
    proba = _proba(fb, X)
    label = max(proba, key=proba.get)
    pos = None
    if label == "fiber_break":
        pos = res["end_dist_m"]
    elif label != "healthy":
        pos = res["worst_event_pos_m"]
    rx = res["features"]["rx_power_dbm"]
    margin = rx - (-27.0)                                  # sensibilité ONU GPON classe B+
    alerts = []
    if label == "fiber_break":
        alerts.append(_alert("critical", f"{LABELS_FR[label]} à ~{pos:.0f} m", "Perte de signal ONU : intervention immédiate."))
    elif label in ("macrobend", "bad_splice", "dirty_connector"):
        alerts.append(_alert("warning", f"{LABELS_FR[label]} à ~{pos:.0f} m" if pos else LABELS_FR[label],
                             f"Perte ponctuelle ≈ {res['features']['max_loss_db']:.1f} dB"))
    elif label == "fiber_tap":
        alerts.append(_alert("critical", f"Prélèvement optique suspecté à ~{pos:.0f} m" if pos else "Prélèvement optique suspecté",
                             "Perte non documentée, faible dépendance en longueur d'onde : vérifier l'intrusion physique."))
    if margin < 3.0 and label != "fiber_break":
        alerts.append(_alert("warning", f"Marge de budget optique faible ({margin:.1f} dB)", f"Rx ONU = {rx:.1f} dBm (seuil −27 dBm)"))
    if not alerts:
        alerts.append(_alert("info", "Liaison optique saine", f"Rx ONU = {rx:.1f} dBm, marge {margin:.1f} dB"))
    health = 100 * (0.7 * proba.get("healthy", 0) + 0.3 * float(np.clip(margin / 10, 0, 1)))
    if label == "fiber_break":
        health = min(health, 5.0)
    return {"health_score": round(float(health), 1),
            "fault": {"label": label, "label_fr": LABELS_FR[label], "confidence": float(proba[label]), "proba": proba},
            "distance_m": pos, "events": res["events"], "end_dist_m": res["end_dist_m"],
            "splitter_loss_db": res["splitter_loss_db"], "rx_power_dbm": rx, "margin_db": float(margin),
            "alerts": alerts, "features": res["features"]}


def analyze_fiber_history(payload: dict) -> dict:
    """Maintenance prédictive d'une liaison fibre à partir de la puissance reçue ONU (dBm/jour)."""
    rx = np.asarray(payload["rx_dbm"], dtype=float)
    sens = float(payload.get("sensitivity_dbm", FIBER_RX_FAIL_DBM))
    f = pd.DataFrame([rx_history_features(rx, sens)])
    hb, tb = load_model("fiberhist_class"), load_model("fiberhist_ttf")
    proba = _proba(hb, f)
    label = max(proba, key=proba.get)
    ttf_ml = float(min(365.0, np.expm1(tb["model"].predict(f[tb["features"]])[0])))
    ttf_lin = rx_time_to_failure(rx, sens)
    margin = float(f["rx_margin_last"].iloc[0])
    alerts = []
    if label == "imminent":
        alerts.append(_alert("critical", f"Perte de liaison probable sous ~{ttf_ml:.0f} jours",
                             f"Rx = {f['rx_last'].iloc[0]:.1f} dBm, pente {f['rx_slope_7d'].iloc[0]:+.2f} dB/jour : "
                             "inspecter connecteurs et chemin de câble."))
    elif label == "degrading":
        alerts.append(_alert("warning", f"Dégradation progressive (~{ttf_ml:.0f} jours avant la sensibilité)",
                             f"Pente {f['rx_slope'].iloc[0]:+.3f} dB/jour : planifier un nettoyage / contrôle OTDR."))
    elif f["rx_max_daily_drop"].iloc[0] > 1.0:
        alerts.append(_alert("info", f"Chute brutale de {f['rx_max_daily_drop'].iloc[0]:.1f} dB observée puis stabilité",
                             "Probable intervention ou déplacement de câble : pas de dégradation en cours."))
    if margin < 3.0:
        alerts.append(_alert("warning", f"Marge de puissance faible ({margin:.1f} dB)", f"Sensibilité ONU : {sens:.0f} dBm."))
    if not alerts:
        alerts.append(_alert("info", "Puissance reçue stable", f"Marge {margin:.1f} dB"))
    return {"label": label, "label_fr": LABELS_FR[label], "proba": proba, "confidence": float(proba[label]),
            "ttf_days_ml": round(ttf_ml, 1), "ttf_days_linear": round(ttf_lin, 1), "margin_db": margin,
            "rx_slope_db_per_day": float(f["rx_slope"].iloc[0]), "alerts": alerts}


# =============================================================================
# RADIO
# =============================================================================
def analyze_radio(payload: dict) -> dict:
    """Diagnostic d'un jeu de KPI radio (4G/5G/6G) : santé et menaces (jamming, IMSI-catcher)."""
    rb = load_model("radio_fault")
    X = pd.DataFrame([{k: float(payload.get(k, 0.0)) for k in rb["features"]}])
    proba = _proba(rb, X)
    label = max(proba, key=proba.get)
    tech = RADIO_TECHS.get(int(payload.get("tech_id", 1)), "5G")
    alerts = []
    msgs = {
        "interference": ("warning", "SINR dégradé et BLER élevé : rechercher une source d'interférence ou un problème de réutilisation de fréquences."),
        "coverage_hole": ("warning", "RSRP très bas : zone d'ombre, ajouter/optimiser un site ou une antenne."),
        "jamming": ("critical", "RSSI très élevé avec SINR effondré : brouillage intentionnel probable."),
        "rogue_cell": ("critical", "Cellule isolée très puissante, changements de TAC, repli RAT / chiffrement nul : IMSI-catcher probable."),
    }
    if label in msgs:
        alerts.append(_alert(msgs[label][0], f"{LABELS_FR[label]} ({tech})", msgs[label][1]))
    else:
        alerts.append(_alert("info", f"Cellule {tech} saine", ""))
    return {"fault": {"label": label, "label_fr": LABELS_FR[label], "confidence": float(proba[label]), "proba": proba},
            "tech": tech, "alerts": alerts, "features": {k: float(v) for k, v in X.iloc[0].items()}}
