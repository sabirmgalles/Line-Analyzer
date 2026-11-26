"""Extraction de caractéristiques (features) pour les modèles IA.

- TDR cuivre   : position/forme/signe des échos -> type et distance du défaut
- OTDR fibre   : détection d'événements (pertes, réflexions) et fin de fibre
- Historique   : pente, volatilité et accélération du SNR / de l'atténuation
- Radio        : KPI 4G/5G/6G déjà scalaires (voir RADIO_FEATURES)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import find_peaks

from config import FIBER_MAX_RANGE_M, HISTORY_DAYS, TDR_MAX_RANGE_M
from core.physics import NOMINAL_VOLTAGE_V, snr_deficit_db

# ----------------------------------------------------------------------------- cuivre
COPPER_BASE = [
    "voltage_mean_v", "voltage_min_v", "voltage_std_v", "voltage_max_step_v",
    "impedance_ohm", "snr_db", "attenuation_db", "crc_errors_per_min",
    "noise_floor_dbm_hz", "impulse_noise_ratio",
]
COPPER_DEFAULTS = {
    "voltage_mean_v": 48.0, "voltage_min_v": 47.7, "voltage_std_v": 0.08,
    "voltage_max_step_v": 0.25, "impedance_ohm": 100.0, "snr_db": 35.0,
    "attenuation_db": 15.0, "crc_errors_per_min": 2.0,
    "noise_floor_dbm_hz": -133.0, "impulse_noise_ratio": 0.005,
}
TDR_FEATURES = [
    "tdr_peak_amp", "tdr_peak_pos_m", "tdr_peak_abs", "tdr_peak_width_m",
    "tdr_n_peaks", "tdr_bipolar", "tdr_neg_amp", "tdr_pos_amp", "tdr_neg_pos_m",
    "tdr_post_ratio", "tdr_noise", "tdr_energy",
]
COPPER_DERIVED = ["snr_deficit_db", "log_crc", "voltage_drop_v"]
COPPER_FEATURES = COPPER_BASE + COPPER_DERIVED + TDR_FEATURES
# Sous-ensemble utilisé pour la détection d'anomalies (TSCM, Isolation Forest)
TSCM_FEATURES = [
    "voltage_min_v", "voltage_std_v", "voltage_max_step_v", "voltage_drop_v",
    "impedance_ohm", "noise_floor_dbm_hz", "impulse_noise_ratio",
    "snr_deficit_db", "tdr_neg_amp",
]
MODULATION_FEATURES = ["attenuation_db", "snr_db", "crc_errors_per_min", "impedance_ohm"]


def tdr_axis(n: int) -> np.ndarray:
    return np.linspace(0.0, TDR_MAX_RANGE_M, n)


def tdr_features(trace, blank_m: float = 90.0) -> dict:
    """Résume une courbe TDR (amplitude du coefficient de réflexion vs distance)."""
    trace = np.asarray(trace, dtype=float)
    x = tdr_axis(len(trace))
    dx = x[1] - x[0]
    mask = x > blank_m                      # on ignore la zone aveugle (impulsion émise)
    t, xm = trace[mask], x[mask]
    a = np.abs(t)
    noise = max(1.4826 * float(np.median(np.abs(t - np.median(t)))), 1e-4)
    i = int(np.argmax(a))

    # largeur à mi-hauteur du pic principal
    thr = 0.5 * a[i]
    lo = i
    while lo > 0 and a[lo - 1] >= thr:
        lo -= 1
    hi = i
    while hi < len(a) - 1 and a[hi + 1] >= thr:
        hi += 1

    peaks, _ = find_peaks(a, prominence=max(0.03, 6 * noise))
    win = t[max(0, i - 12): i + 13]
    pos, neg = max(float(win.max()), 0.0), -min(float(win.min()), 0.0)
    return {
        "tdr_peak_amp": float(t[i]),
        "tdr_peak_pos_m": float(xm[i]),
        "tdr_peak_abs": float(a[i]),
        "tdr_peak_width_m": float((hi - lo + 1) * dx),
        "tdr_n_peaks": int(len(peaks)),
        "tdr_bipolar": float(min(pos, neg) / (max(pos, neg) + 1e-9)),
        "tdr_neg_amp": float(t.min()),
        "tdr_pos_amp": float(t.max()),
        "tdr_neg_pos_m": float(xm[int(np.argmin(t))]),
        "tdr_post_ratio": float(a[i + 10:].mean() / noise) if i + 10 < len(a) else 0.0,
        "tdr_noise": float(noise),
        "tdr_energy": float(np.sum(t ** 2) * dx / 1000.0),
    }


def copper_feature_row(measure: dict, trace) -> dict:
    """Construit la ligne de features complète (mesures + dérivées + TDR)."""
    row = {k: float(measure.get(k, COPPER_DEFAULTS[k])) for k in COPPER_BASE}
    row["snr_deficit_db"] = float(snr_deficit_db(row["snr_db"], row["attenuation_db"]))
    row["log_crc"] = float(np.log1p(row["crc_errors_per_min"]))
    row["voltage_drop_v"] = float(NOMINAL_VOLTAGE_V - row["voltage_min_v"])
    row.update(tdr_features(trace))
    return row


def build_copper_features(df: pd.DataFrame, tdr: np.ndarray) -> pd.DataFrame:
    """Version vectorisée (boucle) pour tout un dataset."""
    rows = [copper_feature_row(r, tdr[k]) for k, r in enumerate(df[COPPER_BASE].to_dict("records"))]
    return pd.DataFrame(rows)[COPPER_FEATURES]


# ----------------------------------------------------------------------------- historique
HISTORY_FEATURES = [
    "snr_last", "snr_mean", "snr_slope", "snr_slope_7d", "snr_resid_std", "snr_min",
    "att_last", "att_slope", "att_slope_7d", "crc_last_log", "crc_slope_log", "snr_accel",
]


def _slope(y: np.ndarray) -> float:
    return float(np.polyfit(np.arange(len(y)), y, 1)[0])


def history_features(snr, att, crc) -> dict:
    """Pente, accélération et volatilité des séries SNR / atténuation / CRC."""
    snr, att, crc = (np.asarray(v, dtype=float) for v in (snr, att, crc))
    t = np.arange(len(snr))
    fit = np.polyval(np.polyfit(t, snr, 1), t)
    s_full, s7 = _slope(snr), _slope(snr[-7:])
    return {
        "snr_last": float(snr[-3:].mean()),
        "snr_mean": float(snr.mean()),
        "snr_slope": s_full,
        "snr_slope_7d": s7,
        "snr_resid_std": float(np.std(snr - fit)),
        "snr_min": float(snr.min()),
        "att_last": float(att[-3:].mean()),
        "att_slope": _slope(att),
        "att_slope_7d": _slope(att[-7:]),
        "crc_last_log": float(np.log1p(crc[-3:]).mean()),
        "crc_slope_log": _slope(np.log1p(crc)),
        "snr_accel": s7 - s_full,          # < 0 : la dégradation s'accélère
    }


def linear_time_to_failure(snr, fail_db: float, horizon: float = 365.0) -> float:
    """Extrapolation linéaire (jours) avant que le SNR ne passe sous `fail_db`."""
    snr = np.asarray(snr, dtype=float)
    s = _slope(snr[-14:]) if len(snr) >= 14 else _slope(snr)
    last = float(snr[-3:].mean())
    if s >= -1e-3:
        return horizon
    return float(min(horizon, max(0.0, (last - fail_db) / abs(s))))


# ----------------------------------------------------------------------------- fibre
FIBER_FEATURES = [
    "end_ratio", "n_events", "max_loss_db", "max_loss_pos_frac", "n_reflective",
    "max_spike_db", "excess_loss_db", "lambda_ratio", "rx_power_dbm", "slope_db_km",
]
SPLITTER_MIN_LOSS_DB = 7.0     # une chute > 7 dB = coupleur PON (1:8 ou plus), pas un défaut


def analyze_otdr(trace, expected_length_m: float, fiber_att_db_km: float = 0.22,
                 lambda_ratio: float = 1.0, rx_power_dbm: float | None = None,
                 tx_power_dbm: float = 3.0, max_range_m: float = FIBER_MAX_RANGE_M) -> dict:
    """Détecte les événements d'une trace OTDR et calcule les features fibre.

    Retourne {"features": {...}, "events": [...], "end_dist_m": float, "splitter_loss_db": float}.
    """
    trace = np.asarray(trace, dtype=float)
    n = len(trace)
    dx = max_range_m / n
    x = np.arange(n) * dx
    med = median_filter(trace, size=9, mode="nearest")        # supprime les pics de réflexion
    sm = uniform_filter1d(med, 5, mode="nearest")
    floor = float(np.median(trace[-max(30, n // 20):]))        # plancher de bruit
    dead = int(250 / dx)                                       # zone morte du connecteur OTDR

    above = sm > floor + 4.0
    bad = np.where(~above[dead:])[0]
    end_idx = dead + int(bad[0]) if len(bad) else n - 1        # fin de la fibre "visible"
    end_dist = end_idx * dx

    # pente d'atténuation : médiane de pentes locales (robuste aux marches)
    slopes = [np.polyfit(x[i:i + 40] / 1000.0, med[i:i + 40], 1)[0]
              for i in range(dead + 5, end_idx - 45, 20)]
    slope = float(np.clip(np.median(slopes), -0.6, -0.1)) if slopes else -fiber_att_db_km

    # détection des marches (pertes ponctuelles) sur le résidu détrendé
    resid = med - slope * (x / 1000.0)
    w, g = 10, 4                      # fenêtre de mesure et écart autour de l'événement (points)

    def step_profile(r: np.ndarray) -> np.ndarray:
        """Niveau moyen avant - niveau moyen après, avec un écart `g` de part et d'autre
        de l'événement (comme la méthode des moindres carrés des OTDR) : > 0 = chute."""
        cs = np.concatenate([[0.0], np.cumsum(r)])
        prof = np.zeros(n)
        i = np.arange(w + g, n - w - g)
        prof[i] = (cs[i - g + 1] - cs[i - g - w + 1]) / w - (cs[i + g + w] - cs[i + g]) / w
        prof[:dead + w] = 0.0
        prof[max(0, end_idx - (w + g + 3)):] = 0.0       # la chute finale vers le bruit n'est pas une perte
        return prof

    d = step_profile(resid)
    loss_pk, _ = find_peaks(d, height=0.3, distance=8)
    dval = {int(p): float(d[p]) for p in loss_pk}
    best_prof = d
    big = [int(p) for p in loss_pk if d[p] >= SPLITTER_MIN_LOSS_DB]
    if big:
        # 2e passe : on retire la marche du coupleur PON, qui masque les défauts voisins (< 100 m)
        resid2 = resid.copy()
        for p in big:                      # on "comble" la marche : plateau de part et d'autre, ±6 points aplatis
            lo, hi = max(7, p - 6), min(n - 8, p + 6)
            left, right = resid2[lo - 6:lo].mean(), resid2[hi + 1:hi + 7].mean()
            resid2[hi + 1:] += left - right
            resid2[lo:hi + 1] = left
        d2 = step_profile(resid2)
        small_pk, _ = find_peaks(d2, height=0.3, distance=8)
        dval = {p: float(d[p]) for p in big}
        dval.update({int(p): float(d2[p]) for p in small_pk})
        loss_pk = np.array(sorted(dval))
        best_prof = d2
    # meilleure estimation de position même si aucun événement n'atteint le seuil (aide à la localisation)
    bp = int(np.argmax(best_prof))
    best_guess = float(bp * dx) if best_prof[bp] > 0.12 else None

    # détection des pics de réflexion
    spike = trace - med
    sp_pk, sp_props = find_peaks(spike, height=1.5, distance=4)
    sp_pk = [(p, h) for p, h in zip(sp_pk, sp_props["peak_heights"])
             if p * dx > 250 and p < end_idx - 6]

    events, used = [], set()
    splitter_loss = 0.0
    for p in loss_pk:
        loss = dval[int(p)]
        if loss >= SPLITTER_MIN_LOSS_DB:
            splitter_loss += loss
            events.append({"pos_m": float(p * dx), "loss_db": loss, "spike_db": 0.0, "type": "splitter"})
            continue
        near = [(q, h) for q, h in sp_pk if abs(q - p) <= 5]
        sp = max((h for _, h in near), default=0.0)
        used.update(q for q, _ in near)
        events.append({"pos_m": float(p * dx), "loss_db": loss, "spike_db": float(sp),
                       "type": "reflective" if sp > 0 else "loss"})
    for q, h in sp_pk:                                           # réflexions sans perte associée
        if q not in used:
            events.append({"pos_m": float(q * dx), "loss_db": 0.0, "spike_db": float(h), "type": "reflective"})

    faults = [e for e in events if e["type"] != "splitter"]
    worst = max(faults, key=lambda e: e["loss_db"] + 0.1 * e["spike_db"], default=None)
    total_drop = float(sm[dead + 3] - sm[max(end_idx - 5, dead + 4)])
    expected_drop = abs(slope) * end_dist / 1000.0 + splitter_loss
    rx = rx_power_dbm if rx_power_dbm is not None else tx_power_dbm - max(total_drop, 0.0) - 0.5

    feats = {
        "end_ratio": float(end_dist / max(expected_length_m, 1.0)),
        "n_events": float(len(faults)),
        "max_loss_db": float(worst["loss_db"]) if worst else 0.0,
        "max_loss_pos_frac": float(worst["pos_m"] / max(end_dist, 1.0)) if worst else 0.0,
        "n_reflective": float(sum(1 for e in faults if e["spike_db"] > 0)),
        "max_spike_db": float(max((e["spike_db"] for e in faults), default=0.0)),
        "excess_loss_db": float(total_drop - expected_drop),
        "lambda_ratio": float(lambda_ratio),
        "rx_power_dbm": float(rx),
        "slope_db_km": float(abs(slope)),
    }
    return {"features": feats, "events": events, "end_dist_m": float(end_dist),
            "splitter_loss_db": float(splitter_loss),
            "worst_event_pos_m": float(worst["pos_m"]) if worst else best_guess}


# ----------------------------------------------------------------------------- historique fibre (Rx ONU)
FIBER_HISTORY_FEATURES = [
    "rx_last", "rx_mean", "rx_slope", "rx_slope_7d", "rx_resid_std", "rx_min",
    "rx_accel", "rx_total_drop", "rx_max_daily_drop", "rx_margin_last",
]


def rx_history_features(rx, sensitivity_dbm: float = -28.0) -> dict:
    """Résume la puissance reçue (dBm) d'un ONU sur N jours."""
    rx = np.asarray(rx, dtype=float)
    t = np.arange(len(rx))
    fit = np.polyval(np.polyfit(t, rx, 1), t)
    s_full, s7 = _slope(rx), _slope(rx[-7:])
    last = float(rx[-3:].mean())
    return {
        "rx_last": last, "rx_mean": float(rx.mean()), "rx_slope": s_full, "rx_slope_7d": s7,
        "rx_resid_std": float(np.std(rx - fit)), "rx_min": float(rx.min()),
        "rx_accel": s7 - s_full, "rx_total_drop": float(rx[:3].mean() - last),
        "rx_max_daily_drop": float(max(0.0, -np.diff(rx).min())),     # plus forte chute d'un jour à l'autre
        "rx_margin_last": last - sensitivity_dbm,
    }


def rx_time_to_failure(rx, sensitivity_dbm: float = -28.0, horizon: float = 365.0) -> float:
    """Extrapolation linéaire (jours) avant que la puissance reçue n'atteigne la sensibilité."""
    rx = np.asarray(rx, dtype=float)
    s = _slope(rx[-14:]) if len(rx) >= 14 else _slope(rx)
    last = float(rx[-3:].mean())
    if s >= -1e-3:
        return horizon
    return float(min(horizon, max(0.0, (last - sensitivity_dbm) / abs(s))))


# ----------------------------------------------------------------------------- radio
RADIO_FEATURES = [
    "tech_id", "rsrp_dbm", "rsrq_db", "sinr_db", "rssi_dbm", "bler_pct", "cqi",
    "throughput_mbps", "n_neighbors", "tac_changes_per_h", "rat_fallback", "cipher_null",
]
