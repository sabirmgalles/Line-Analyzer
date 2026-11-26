"""Générateur de données synthétiques réalistes pour AI Line Analyzer.

Quatre jeux de données sont produits dans ./data :
  1. copper_lines.csv + copper_tdr.npy : lignes cuivre (saines / défauts / écoute)
  2. history.npz + history_labels.csv   : séries SNR/atténuation/CRC sur 30 jours
  3. fiber_links.csv + fiber_otdr.npy   : liaisons FTTH/GPON avec traces OTDR
  4. radio_kpi.csv                      : KPI radio 4G/5G/6G (diagnostic + sécurité)
  5. fiber_history.npz + labels         : puissance reçue ONU sur 30 jours (maintenance prédictive fibre)

Les fonctions `simulate_*` sont aussi utilisées par l'application Streamlit
pour fabriquer des "mesures terrain" de démonstration.

Usage :  python data_generator.py --n-copper 6000 --seed 42
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.special import expit

from config import (COPPER_CLASSES, DATA_DIR, FIBER_CLASSES, FIBER_HISTORY_KINDS, FIBER_MAX_RANGE_M,
                    FIBER_N_POINTS, FIBER_RX_FAIL_DBM, HISTORY_DAYS, HISTORY_KINDS, RADIO_CLASSES,
                    REF_FREQ_MHZ, SEED, SNR_FAIL_DB, TDR_MAX_RANGE_M, TDR_N_POINTS)
from core.physics import NOMINAL_VOLTAGE_V, cable_loss_db, expected_snr_db


# =============================================================================
# 1. LIGNES CUIVRE
# =============================================================================
def _gauss(x, center, width, amp):
    """Écho TDR simple (réflexion de même signe)."""
    return amp * np.exp(-0.5 * ((x - center) / width) ** 2)


def _blip(x, center, width, amp):
    """Écho bipolaire (+ puis -) typique d'une épissure : dérivée de gaussienne."""
    u = (x - center) / width
    return amp * (-u) * np.exp(0.5 - 0.5 * u ** 2)


def simulate_copper_line(label: str, rng: np.random.Generator,
                         length_m: float | None = None, gauge_mm: float | None = None):
    """Simule UNE ligne cuivre. Retourne (enregistrement dict, courbe TDR ndarray).

    La simulation applique, pour chaque défaut, des signatures physiques :
      - TDR : coefficient de réflexion (+ ouvert, - court-circuit, - shunt = dérivation)
      - électrique : tension, impédance, SNR, atténuation, CRC, bruit
    Des cas "difficiles" (écoute discrète, oxydation avec chute de tension) créent
    un chevauchement réaliste entre classes.
    """
    gauge = float(gauge_mm) if gauge_mm else float(rng.choice([0.4, 0.5, 0.6], p=[0.5, 0.3, 0.2]))
    if length_m is None:
        r = rng.random()
        length_m = rng.uniform(50, 400) if r < 0.30 else rng.uniform(400, 2500) if r < 0.70 \
            else rng.uniform(2500, 5000)
    length_m = float(length_m)
    x = np.linspace(0.0, TDR_MAX_RANGE_M, TDR_N_POINTS)
    vop_err = 1.0 + rng.normal(0, 0.015)          # erreur sur la vitesse de propagation (axe distance)

    # ---- état sain de référence
    att = float(cable_loss_db(length_m, REF_FREQ_MHZ, gauge) + rng.normal(0, 0.7))
    snr = float(expected_snr_db(att) + rng.normal(0, 2.5))
    voltage = rng.normal(NOMINAL_VOLTAGE_V, 0.4)
    v_std = abs(rng.normal(0.08, 0.03))
    v_step = abs(rng.normal(0.25, 0.10))
    v_min = voltage - 3 * v_std - abs(rng.normal(0, 0.1))
    impedance = rng.normal(100, 4)
    noise_floor = rng.normal(-133, 2.5)
    crc = float(rng.gamma(1.2, 2.0))
    impulse = abs(rng.normal(0.005, 0.004))

    fault_d, eff_len, end_gamma, events = np.nan, length_m, rng.uniform(0.03, 0.14), []
    stealth = rng.random() < 0.12                  # défaut "discret" : signature atténuée
    k = 0.5 if stealth else 1.0

    if label in ("open_cut", "short_circuit"):
        fault_d = max(60.0, rng.uniform(0.08, 0.95) * length_m)
        eff_len = fault_d
        sign = 1.0 if label == "open_cut" else -1.0
        end_gamma = sign * rng.uniform(0.7, 0.95)
        snr, att = rng.uniform(0, 4), rng.uniform(60, 90)           # plus de synchronisation
        crc = rng.uniform(500, 5000)
        if label == "open_cut":
            impedance = rng.uniform(2000, 20000)
        else:
            impedance = rng.uniform(1, 25)
            voltage = rng.uniform(0, 6)
            v_min = voltage - abs(rng.normal(0, 0.4))
    elif label == "oxidation":
        fault_d = rng.uniform(0.1, 0.9) * length_m
        events.append((_gauss, fault_d, rng.uniform(40, 120), rng.uniform(0.10, 0.30) * k))
        att += rng.uniform(4, 16) * k
        snr -= rng.uniform(3, 12) * k
        crc = float(rng.gamma(3, 40))
        impedance *= rng.uniform(1.05, 1.4)
        drop = rng.uniform(0.3, 3.0) * (1.6 if rng.random() < 0.15 else 1.0)  # cas piégeux
        voltage -= drop
        v_min -= drop
        v_std += rng.uniform(0.0, 0.2)
        impulse += rng.uniform(0.01, 0.05)
    elif label == "bad_splice":
        fault_d = rng.uniform(0.1, 0.9) * length_m
        events.append((_blip, fault_d, rng.uniform(25, 40), rng.choice([-1, 1]) * rng.uniform(0.10, 0.30) * k))
        att += rng.uniform(1, 5) * k
        snr -= rng.uniform(1, 7) * k
        crc = float(rng.gamma(2, 40))
        impedance *= rng.uniform(0.95, 1.15)
        impulse += rng.uniform(0.02, 0.08)           # contact intermittent
        v_step += rng.uniform(0.0, 0.8)
    elif label == "wiretap":
        fault_d = rng.uniform(0.05, 0.95) * length_m
        amp = rng.uniform(0.08, 0.35) * k
        events.append((_gauss, fault_d, rng.uniform(25, 50), -amp))                 # shunt -> écho négatif
        events.append((_gauss, fault_d + rng.uniform(20, 150), 30, 0.3 * amp))      # écho de la dérivation
        drop = rng.uniform(0.8, 6.0) * k
        voltage -= drop
        v_min -= drop + rng.uniform(0, 1.5)
        v_step = rng.uniform(1.0, 5.0) * k + v_step
        v_std += rng.uniform(0.05, 0.4) * k
        impedance *= 1.0 - rng.uniform(0.03, 0.22) * k
        snr -= rng.uniform(1, 6) * k
        noise_floor += rng.uniform(2, 10) * k        # bruit suspect (micro / émetteur)
        impulse += rng.uniform(0.01, 0.06)
        att += rng.uniform(0.3, 2.5) * k
        crc = float(rng.gamma(2, 15))

    # ---- courbe TDR (amplitude = coefficient de réflexion, gain compensé)
    trace = 1.2 * np.exp(-x / 25.0)                                   # impulsion émise (zone aveugle)
    two_way = lambda d: 10 ** (-2 * float(cable_loss_db(d, 0.01, gauge)) / 20)
    trace += _gauss(x, eff_len * vop_err, 30, end_gamma * two_way(eff_len))   # fin de ligne / défaut franc
    for fn, pos, width, amp in events:
        trace += fn(x, pos * vop_err, width, amp * two_way(pos))
    trace += rng.normal(0, rng.uniform(0.004, 0.015), size=x.shape)  # bruit de mesure
    trace += _gauss(x, rng.uniform(300, 4500), 40, rng.uniform(0, 0.02))  # diaphonie parasite

    rec = {
        "label": label, "line_length_m": length_m, "gauge_mm": gauge,
        "fault_distance_m": float(fault_d),
        "voltage_mean_v": float(voltage), "voltage_min_v": float(v_min),
        "voltage_std_v": float(v_std), "voltage_max_step_v": float(v_step),
        "impedance_ohm": float(impedance), "snr_db": float(max(snr, 0.0)),
        "attenuation_db": float(att), "crc_errors_per_min": float(crc),
        "noise_floor_dbm_hz": float(noise_floor), "impulse_noise_ratio": float(impulse),
    }
    return rec, trace.astype(np.float32)


def generate_copper_dataset(n: int, rng: np.random.Generator):
    labels = rng.choice(COPPER_CLASSES, size=n, p=[0.28, 0.13, 0.12, 0.15, 0.14, 0.18])
    rows, traces = [], np.zeros((n, TDR_N_POINTS), dtype=np.float32)
    for k, lab in enumerate(labels):
        rec, tr = simulate_copper_line(str(lab), rng)
        rec["line_id"] = f"CU-{k:05d}"
        rows.append(rec)
        traces[k] = tr
    return pd.DataFrame(rows), traces


# =============================================================================
# 2. HISTORIQUE (maintenance prédictive)
# =============================================================================
def simulate_history(kind: str, rng: np.random.Generator, days: int = HISTORY_DAYS) -> dict:
    """Série journalière SNR / atténuation / CRC et jours réels avant coupure.

    kind : "stable" | "degrading" (pente linéaire) | "accelerating" (dégradation exponentielle)
    """
    t = np.arange(days)
    att0 = rng.uniform(6, 45)
    snr_last = rng.uniform(14, 42)
    sigma = rng.uniform(0.3, 1.1)
    if kind == "stable":
        slope = rng.normal(0, 0.012)
        clean = snr_last + slope * (t - (days - 1))
        ttf = 365.0 if slope > -1e-3 else min(365.0, (snr_last - SNR_FAIL_DB) / abs(slope))
    elif kind == "degrading":
        slope = -rng.uniform(0.05, 0.40)
        clean = snr_last + slope * (t - (days - 1))
        ttf = min(365.0, (snr_last - SNR_FAIL_DB) / abs(slope))
    else:  # accelerating : chute exponentielle, coupure dans K jours
        K, tau, snr0 = rng.uniform(1, 12), rng.uniform(4, 10), rng.uniform(30, 46)
        A = (snr0 - SNR_FAIL_DB) / (np.exp((days - 1 + K) / tau) - 1)
        clean = snr0 - A * (np.exp(t / tau) - 1)
        ttf = float(K)
    noise = rng.normal(0, sigma, days)
    snr = clean + noise
    worsening = np.maximum(clean[0] - clean, 0.0)
    att = att0 + 0.35 * worsening + rng.normal(0, 0.3, days)
    crc = rng.gamma(1.5, 2.0 * np.exp(0.12 * worsening))
    label = "imminent" if ttf <= 7 else "degrading" if ttf <= 90 else "stable"
    return {"snr_db": snr, "attenuation_db": att, "crc_errors_per_min": crc,
            "ttf_days": float(ttf), "label": label}


def generate_history_dataset(n: int, rng: np.random.Generator):
    kinds = rng.choice(HISTORY_KINDS, size=n, p=[0.40, 0.30, 0.30])
    S, A, C = (np.zeros((n, HISTORY_DAYS), dtype=np.float32) for _ in range(3))
    meta = []
    for k, kind in enumerate(kinds):
        h = simulate_history(str(kind), rng)
        S[k], A[k], C[k] = h["snr_db"], h["attenuation_db"], h["crc_errors_per_min"]
        meta.append({"series_id": f"HI-{k:05d}", "kind": kind, "label": h["label"], "ttf_days": h["ttf_days"]})
    return S, A, C, pd.DataFrame(meta)


# =============================================================================
# 3. FIBRE OPTIQUE (FTTH / GPON, trace OTDR)
# =============================================================================
def simulate_fiber_link(label: str, rng: np.random.Generator, length_m: float | None = None):
    """Simule une liaison OLT -> splitter -> ONU et sa trace OTDR (dB relatifs)."""
    length = float(length_m) if length_m else float(rng.uniform(800, 15000))
    att_km = float(rng.normal(0.22, 0.01))                    # 1550 nm
    n = FIBER_N_POINTS
    dx = FIBER_MAX_RANGE_M / n
    x = np.arange(n) * dx
    floor = -float(rng.uniform(33, 42))                        # dynamique de l'OTDR
    split_pos = float(rng.uniform(0.2, 0.8) * length)
    split_loss = float(rng.choice([10.5, 13.7, 17.5]))         # 1:8 / 1:16 / 1:32

    # événements : (position, perte dB, hauteur du pic de réflexion dB)
    events = [(split_pos, split_loss, rng.uniform(1, 4))]
    for _ in range(int(length // 2500) + 1):                   # soudures
        events.append((rng.uniform(300, length - 100), rng.uniform(0.02, 0.15), 0.0))
    end_spike = rng.uniform(4, 9)
    eff_end, lam, fault_d, extra_loss = length, 1.0 + abs(rng.normal(0, 0.1)), np.nan, 0.0
    if label == "fiber_break":
        fault_d = float(rng.uniform(0.1, 0.95) * length)
        eff_end = fault_d
        end_spike = rng.uniform(5, 12) if rng.random() < 0.7 else 0.0
    elif label == "macrobend":
        fault_d = float(rng.uniform(0.1, 0.95) * length); extra_loss = rng.uniform(0.6, 3.5)
        events.append((fault_d, extra_loss, 0.0)); lam = rng.uniform(1.8, 3.5)
    elif label == "dirty_connector":
        fault_d = float(split_pos if rng.random() < 0.5 else rng.uniform(0.05, 0.5) * length)
        extra_loss = rng.uniform(0.8, 3.0)
        events.append((fault_d, extra_loss, rng.uniform(8, 14))); lam = 1.0 + abs(rng.normal(0, 0.1))
    elif label == "bad_splice":
        fault_d = float(rng.uniform(0.1, 0.95) * length); extra_loss = rng.uniform(0.4, 1.6)
        events.append((fault_d, extra_loss, 0.0)); lam = 1.0 + rng.normal(0, 0.15)
    elif label == "fiber_tap":
        fault_d = float(rng.uniform(0.1, 0.95) * length); extra_loss = rng.uniform(0.25, 1.2)
        events.append((fault_d, extra_loss, 0.0)); lam = rng.uniform(1.1, 1.9)

    clean = -att_km * x / 1000.0
    total_loss = att_km * eff_end / 1000.0 + 0.7
    for pos, loss, spike in events:
        if pos >= eff_end:
            continue
        clean -= loss * expit((x - pos) / (0.6 * dx))          # marche (largeur ~ durée d'impulsion OTDR)
        if spike > 0:
            clean[int(pos / dx)] += spike
            clean[min(n - 1, int(pos / dx) + 1)] += 0.5 * spike
        total_loss += loss
    if end_spike > 0 and eff_end <= length:
        clean[min(n - 1, int(eff_end / dx))] += end_spike
    clean[x > eff_end + dx] = floor                           # plus de rétrodiffusion
    clean = np.maximum(clean, floor)
    sigma = 0.04 + 0.6 * np.exp(-(clean - floor) / 4.0)       # le bruit croît près du plancher
    trace = clean + rng.normal(0, 1, n) * sigma

    tx = float(rng.uniform(1.5, 5.0))
    rx = -45.0 + rng.normal(0, 1) if label == "fiber_break" else tx - total_loss + rng.normal(0, 0.4)
    rec = {"label": label, "length_m": length, "expected_length_m": length * (1 + rng.normal(0, 0.005)),
           "fault_distance_m": fault_d, "fiber_att_db_km": att_km, "tx_power_dbm": tx,
           "rx_power_dbm": float(rx), "lambda_ratio": float(lam), "splitter_loss_db": split_loss,
           "splitter_pos_m": split_pos}
    return rec, trace.astype(np.float32)


def generate_fiber_dataset(n: int, rng: np.random.Generator):
    labels = rng.choice(FIBER_CLASSES, size=n, p=[0.30, 0.14, 0.14, 0.14, 0.14, 0.14])
    rows, traces = [], np.zeros((n, FIBER_N_POINTS), dtype=np.float32)
    for k, lab in enumerate(labels):
        rec, tr = simulate_fiber_link(str(lab), rng)
        rec["link_id"] = f"FO-{k:05d}"
        rows.append(rec); traces[k] = tr
    return pd.DataFrame(rows), traces


def simulate_rx_history(kind: str, rng: np.random.Generator, days: int = HISTORY_DAYS) -> dict:
    """Puissance reçue d'un ONU (dBm), 1 mesure/jour, et jours réels avant la sensibilité (-28 dBm).

    kind : "stable" | "step" (saut de perte puis stabilité : pas de panne à venir) |
           "degrading" (pente linéaire : connecteur qui se salit, courbure) |
           "accelerating" (dégradation exponentielle : fibre sous contrainte, infiltration d'eau)
    """
    t = np.arange(days)
    fail = FIBER_RX_FAIL_DBM
    rx_last = rng.uniform(-25.5, -14.0)
    season = rng.uniform(0.0, 0.25) * np.sin(2 * np.pi * (t / rng.uniform(10, 30)) + rng.uniform(0, 6))  # température
    if kind == "stable":
        slope = rng.normal(0, 0.004)
        clean = rx_last + slope * (t - (days - 1))
        ttf = 365.0 if slope > -5e-4 else min(365.0, (rx_last - fail) / abs(slope))
    elif kind == "step":
        slope = rng.normal(0, 0.003)
        clean = rx_last + slope * (t - (days - 1))
        clean = clean - rng.uniform(1.0, 4.0) * (t >= rng.integers(5, days - 5))  # perte brutale puis palier
        ttf = 365.0 if slope > -5e-4 else min(365.0, (rx_last - fail) / abs(slope))
    elif kind == "degrading":
        slope = -rng.uniform(0.01, 0.12)
        clean = rx_last + slope * (t - (days - 1))
        ttf = min(365.0, (rx_last - fail) / abs(slope))
    else:  # accelerating
        K, tau, rx0 = rng.uniform(1, 12), rng.uniform(4, 10), rng.uniform(-20, -14)
        A = (rx0 - fail) / (np.exp((days - 1 + K) / tau) - 1)
        clean = rx0 - A * (np.exp(t / tau) - 1)
        ttf = float(K)
    rx = clean + season + rng.normal(0, rng.uniform(0.08, 0.35), days)
    label = "imminent" if ttf <= 7 else "degrading" if ttf <= 90 else "stable"
    return {"rx_dbm": rx, "ttf_days": float(ttf), "label": label}


def generate_fiber_history_dataset(n: int, rng: np.random.Generator):
    kinds = rng.choice(FIBER_HISTORY_KINDS, size=n, p=[0.30, 0.20, 0.25, 0.25])
    R, meta = np.zeros((n, HISTORY_DAYS), dtype=np.float32), []
    for k, kind in enumerate(kinds):
        h = simulate_rx_history(str(kind), rng)
        R[k] = h["rx_dbm"]
        meta.append({"series_id": f"FH-{k:05d}", "kind": kind, "label": h["label"], "ttf_days": h["ttf_days"]})
    return R, pd.DataFrame(meta)


# =============================================================================
# 4. RADIO 4G / 5G / 6G
# =============================================================================
_PEAK_MBPS = {0: 150.0, 1: 900.0, 2: 3000.0}


def simulate_radio_kpi(label: str, tech_id: int, rng: np.random.Generator) -> dict:
    """KPI mesurés par un terminal / scanner : santé de la cellule et sécurité radio."""
    rsrp = float(np.clip(rng.normal(-88, 10), -115, -60))
    sinr = float(np.clip(10 + 0.35 * (rsrp + 95) + rng.normal(0, 3), -3, 30))
    rsrq = float(np.clip(-9 + 0.15 * (sinr - 10) + rng.normal(0, 1), -19.5, -3))
    rssi = rsrp + 22 + rng.normal(0, 1.5)
    n_neigh = int(rng.poisson(5)) + 1
    tac, fallback, cipher = rng.poisson(0.2), 0, 0
    if rng.random() < 0.02:
        fallback = 1
    if label == "interference":
        sinr -= rng.uniform(8, 16); rsrq -= rng.uniform(3, 7); rssi += rng.uniform(3, 8)
    elif label == "coverage_hole":
        rsrp = rng.uniform(-125, -108); sinr = rng.uniform(-8, 2); rssi = rsrp + 22
        rsrq = rng.uniform(-19.5, -14); n_neigh = int(rng.integers(0, 2))
    elif label == "jamming":
        rssi = rng.uniform(-60, -40); rsrp = rng.uniform(-115, -98); sinr = rng.uniform(-15, -5)
        rsrq = rng.uniform(-19.5, -16); n_neigh = int(rng.integers(0, 3))
    elif label == "rogue_cell":
        rsrp = rng.uniform(-72, -55); sinr = rng.uniform(20, 30); rssi = rsrp + 22
        rsrq = rng.uniform(-8, -4); n_neigh = int(rng.integers(0, 2))
        tac = int(rng.integers(2, 11)); fallback = int(rng.random() < 0.6); cipher = int(rng.random() < 0.5)
    sinr = float(np.clip(sinr, -20, 35)); rsrq = float(np.clip(rsrq, -19.5, -3))
    bler = float(np.clip(10 * np.exp(-0.35 * sinr) + abs(rng.normal(0, 0.5)), 0, 100))
    if label == "interference":
        bler = float(np.clip(bler + rng.uniform(5, 25), 0, 100))
    if label == "jamming":
        bler = rng.uniform(50, 100)
    cqi = int(np.clip(round(sinr * 0.5 + 6 + rng.normal(0, 1)), 0, 15))
    thr = _PEAK_MBPS[tech_id] * (cqi / 15) ** 1.3 * (1 - bler / 100) * rng.uniform(0.2, 1.0)
    return {"label": label, "tech_id": tech_id, "rsrp_dbm": rsrp, "rsrq_db": rsrq, "sinr_db": sinr,
            "rssi_dbm": float(rssi), "bler_pct": bler, "cqi": cqi,
            "throughput_mbps": float(thr), "n_neighbors": n_neigh,
            "tac_changes_per_h": float(tac), "rat_fallback": fallback, "cipher_null": cipher}


def generate_radio_dataset(n: int, rng: np.random.Generator) -> pd.DataFrame:
    labels = rng.choice(RADIO_CLASSES, size=n, p=[0.40, 0.15, 0.15, 0.15, 0.15])
    techs = rng.choice([0, 1, 2], size=n, p=[0.4, 0.4, 0.2])
    return pd.DataFrame([simulate_radio_kpi(str(l), int(t), rng) for l, t in zip(labels, techs)])


# =============================================================================
# CLI
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Génère les datasets synthétiques d'AI Line Analyzer")
    ap.add_argument("--n-copper", type=int, default=6000)
    ap.add_argument("--n-history", type=int, default=3000)
    ap.add_argument("--n-fiber", type=int, default=2500)
    ap.add_argument("--n-radio", type=int, default=3000)
    ap.add_argument("--n-fiber-history", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    DATA_DIR.mkdir(exist_ok=True)

    df, tdr = generate_copper_dataset(args.n_copper, rng)
    df.to_csv(DATA_DIR / "copper_lines.csv", index=False); np.save(DATA_DIR / "copper_tdr.npy", tdr)
    print(f"[cuivre] {len(df)} lignes\n{df['label'].value_counts().to_string()}\n")

    S, A, C, meta = generate_history_dataset(args.n_history, rng)
    np.savez_compressed(DATA_DIR / "history.npz", snr=S, att=A, crc=C)
    meta.to_csv(DATA_DIR / "history_labels.csv", index=False)
    print(f"[historique] {len(meta)} séries\n{meta['label'].value_counts().to_string()}\n")

    fdf, otdr = generate_fiber_dataset(args.n_fiber, rng)
    fdf.to_csv(DATA_DIR / "fiber_links.csv", index=False); np.save(DATA_DIR / "fiber_otdr.npy", otdr)
    print(f"[fibre] {len(fdf)} liaisons\n{fdf['label'].value_counts().to_string()}\n")

    rdf = generate_radio_dataset(args.n_radio, rng)
    rdf.to_csv(DATA_DIR / "radio_kpi.csv", index=False)
    print(f"[radio] {len(rdf)} mesures\n{rdf['label'].value_counts().to_string()}\n")

    R, fmeta = generate_fiber_history_dataset(args.n_fiber_history, rng)
    np.savez_compressed(DATA_DIR / "fiber_history.npz", rx=R)
    fmeta.to_csv(DATA_DIR / "fiber_history_labels.csv", index=False)
    print(f"[historique fibre] {len(fmeta)} séries\n{fmeta['label'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
