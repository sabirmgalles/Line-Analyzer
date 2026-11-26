"""Modèles physiques simplifiés des lignes cuivre (xDSL).

Ces modèles servent à la fois à SIMULER des lignes (data_generator.py) et à
CALCULER les débits atteignables par profil (recommandation de modulation).
Ils sont volontairement simples (atténuation en racine de f, gap de Shannon)
mais cohérents avec les ordres de grandeur terrain.
"""
from __future__ import annotations

import numpy as np

from config import REF_FREQ_MHZ

# Atténuation du câble à 1 MHz (dB/km) selon le diamètre du conducteur (mm)
GAUGE_K_DB_KM = {0.4: 17.0, 0.5: 13.5, 0.6: 11.0}
NOMINAL_VOLTAGE_V = 48.0
NOISE_BASE_DBM_HZ = -130.0      # bruit de fond avec diaphonie "typique"
SNR_GAP_DB = 12.0               # gap de Shannon + marge 6 dB - gain de codage

# Profils xDSL : bande utile, espacement des tonalités, PSD d'émission, etc.
PROFILES = {
    "ADSL2+": dict(f_min=0.138, f_max=2.208, tone_khz=4.3125, sym_rate=4000,
                   max_bits=15, ds_share=1.00),
    "VDSL2": dict(f_min=0.138, f_max=17.664, tone_khz=4.3125, sym_rate=4000,
                  max_bits=15, ds_share=0.55),
    "G.fast": dict(f_min=2.0, f_max=106.0, tone_khz=51.75, sym_rate=48000,
                   max_bits=12, ds_share=0.70),
}


def cable_loss_db(length_m, f_mhz, gauge_mm: float = 0.4):
    """Atténuation (dB) d'un câble de longueur `length_m` à la fréquence f (MHz)."""
    return GAUGE_K_DB_KM[gauge_mm] * np.sqrt(f_mhz) * np.asarray(length_m, dtype=float) / 1000.0


def expected_snr_db(att_ref_db):
    """SNR moyen attendu d'une ligne saine en fonction de son atténuation."""
    return 48.0 - 0.55 * np.asarray(att_ref_db, dtype=float)


def snr_deficit_db(snr_db, att_ref_db):
    """Dégradation du SNR par rapport à une ligne saine de même atténuation (>= 0)."""
    return np.maximum(0.0, expected_snr_db(att_ref_db) - np.asarray(snr_db, dtype=float))


def _psd_dbm_hz(profile: str, f_mhz: np.ndarray) -> np.ndarray:
    if profile == "ADSL2+":
        return np.full_like(f_mhz, -40.0)
    if profile == "VDSL2":
        return np.where(f_mhz < 1.1, -40.0, -60.0)
    return np.full_like(f_mhz, -76.0)  # G.fast


def estimate_rate_mbps(profile: str, att_ref_db: float, snr_db: float, crc: float = 0.0) -> float:
    """Débit descendant estimé (Mbit/s) d'un profil pour l'état réel de la ligne.

    L'atténuation mesurée à 300 kHz est extrapolée à toute la bande (loi en racine
    de f). Le déficit de SNR et les erreurs CRC agissent comme un bruit additionnel.
    """
    p = PROFILES[profile]
    f = np.arange(p["f_min"], p["f_max"], p["tone_khz"] / 1000.0)
    atten = float(att_ref_db) * np.sqrt(f / REF_FREQ_MHZ)
    penalty = float(snr_deficit_db(snr_db, att_ref_db)) + min(8.0, float(crc) / 40.0)
    snr_tone = _psd_dbm_hz(profile, f) - atten - NOISE_BASE_DBM_HZ - penalty - SNR_GAP_DB
    bits = np.clip(np.floor(np.log2(1.0 + 10.0 ** (snr_tone / 10.0))), 0, p["max_bits"])
    return float(bits.sum() * p["sym_rate"] * p["ds_share"] * 0.9 / 1e6)


def best_profile(att_ref_db: float, snr_db: float, crc: float = 0.0):
    """Profil recommandé + débits estimés pour chaque profil (règle métier).

    Hystérésis volontaire : on ne passe à un profil plus exigeant que s'il apporte
    un gain net (VDSL2 > 1,3 x ADSL2+ ; G.fast > 1,5 x VDSL2 et >= 150 Mbit/s).
    """
    rates = {k: estimate_rate_mbps(k, att_ref_db, snr_db, crc) for k in PROFILES}
    if max(rates.values()) < 1.0:
        return "Aucun", rates
    choice = "ADSL2+"
    if rates["VDSL2"] > 1.3 * rates["ADSL2+"] and rates["VDSL2"] >= 5.0:
        choice = "VDSL2"
    if rates["G.fast"] > 1.5 * rates["VDSL2"] and rates["G.fast"] >= 150.0:
        choice = "G.fast"
    return choice, rates
