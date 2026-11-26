"""Configuration centrale d'AI Line Analyzer (chemins, constantes, classes)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
SEED = 42

# --- Réflectométrie cuivre (TDR) ---------------------------------------------
TDR_N_POINTS = 550            # nombre d'échantillons de la courbe TDR
TDR_MAX_RANGE_M = 5500.0      # portée de l'axe distance (m) -> 10 m / point
REF_FREQ_MHZ = 0.3            # fréquence de référence de l'atténuation mesurée

# --- Réflectométrie fibre (OTDR) ---------------------------------------------
FIBER_N_POINTS = 2000
FIBER_MAX_RANGE_M = 20000.0   # -> 10 m / point

# --- Maintenance prédictive ---------------------------------------------------
HISTORY_DAYS = 30             # fenêtre d'historique (1 mesure / jour)
SNR_FAIL_DB = 8.0             # en dessous : perte de synchronisation
FIBER_RX_FAIL_DBM = -28.0     # sensibilité ONU GPON classe B+ : en dessous, perte de liaison
FIBER_HISTORY_KINDS = ["stable", "step", "degrading", "accelerating"]

# --- Classes -------------------------------------------------------------------
COPPER_CLASSES = ["healthy", "open_cut", "short_circuit", "oxidation", "bad_splice", "wiretap"]
FIBER_CLASSES = ["healthy", "fiber_break", "macrobend", "dirty_connector", "bad_splice", "fiber_tap"]
RADIO_CLASSES = ["healthy", "interference", "coverage_hole", "jamming", "rogue_cell"]
HISTORY_CLASSES = ["stable", "degrading", "imminent"]
HISTORY_KINDS = ["stable", "degrading", "accelerating"]
MODULATION_PROFILES = ["ADSL2+", "VDSL2", "G.fast", "Aucun"]

LABELS_FR = {
    # cuivre
    "healthy": "Ligne saine",
    "open_cut": "Câble coupé (circuit ouvert)",
    "short_circuit": "Court-circuit",
    "oxidation": "Oxydation / corrosion",
    "bad_splice": "Épissure défectueuse",
    "wiretap": "Branchement pirate / écoute",
    # fibre
    "fiber_break": "Rupture de fibre",
    "macrobend": "Macro-courbure",
    "dirty_connector": "Connecteur sale / endommagé",
    "fiber_tap": "Prélèvement optique (tap)",
    # radio
    "interference": "Interférences",
    "coverage_hole": "Trou de couverture",
    "jamming": "Brouillage (jamming)",
    "rogue_cell": "Fausse antenne (IMSI-catcher)",
    # historique
    "stable": "Stable",
    "degrading": "Dégradation progressive",
    "imminent": "Coupure imminente",
}

RADIO_TECHS = {0: "4G", 1: "5G", 2: "6G"}
