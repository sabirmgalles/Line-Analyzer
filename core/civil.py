"""Génie civil FTTH : métré des casses, tranchées, conduits, chambres, forages, aérien + chiffrage.

Entrée : le plan `core.planning.plan_ftth` (avec routage OSM pour des longueurs réelles).
Sortie : quantités, bordereau (BOQ) chiffré en TND, détail par niveau (transport / distribution / branchement).

⚠ Les prix de `DEFAULT_PRICES` sont des ORDRES DE GRANDEUR d'avant-projet, non issus d'un bordereau officiel :
remplacez-les par votre BPU / vos marchés (Tunisie Telecom, Ooredoo, STEG-SONEDE, entreprises locales).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

LEVELS = ("feeder", "dist", "drop")
LEVEL_FR = {"feeder": "Transport (OLT→FDH)", "dist": "Distribution (FDH→NAP)", "drop": "Branchement (NAP→foyers)"}

# Bordereau de prix unitaires (TND, HT) — INDICATIF
DEFAULT_PRICES = {
    "demolition_chaussee":  ("m²", 14.0,  "Découpe + démolition enrobé / chaussée"),
    "demolition_trottoir":  ("m²", 11.0,  "Démolition trottoir (béton / pavés / dalles)"),
    "fouille_terrain":      ("m³", 16.0,  "Fouille en tranchée, terrain meuble"),
    "fouille_rocheuse":     ("m³", 70.0,  "Fouille en terrain rocheux (BRH / explosif exclu)"),
    "lit_sable":            ("m³", 38.0,  "Lit de pose et enrobage sable"),
    "remblai":              ("m³", 14.0,  "Remblai compacté (matériaux de réemploi)"),
    "evacuation":           ("m³", 12.0,  "Évacuation des déblais excédentaires"),
    "reprise_chaussee":     ("m²", 48.0,  "Réfection chaussée (grave bitume + enrobé)"),
    "reprise_trottoir":     ("m²", 38.0,  "Réfection trottoir"),
    "conduit_pehd":         ("ml", 4.5,   "Conduit PEHD Ø40/33 posé"),
    "grillage":             ("ml", 0.8,   "Grillage avertisseur"),
    "chambre_l1t":          ("u",  180.0, "Chambre L1T (branchement / NAP)"),
    "chambre_l2t":          ("u",  420.0, "Chambre L2T (distribution / FDH)"),
    "forage_dirige":        ("ml", 85.0,  "Forage dirigé (traversée de route / oued)"),
    "poteau":               ("u",  260.0, "Poteau support aérien posé"),
    "cable_aerien_pose":    ("ml", 6.0,   "Pose câble aérien (accessoires)"),
    "signalisation":        ("km", 1200.0, "Signalisation et sécurité de chantier"),
}

DEFAULT_PARAMS = {
    # largeur / profondeur de tranchée par niveau (m)
    "width_m":  {"feeder": 0.45, "dist": 0.40, "drop": 0.30},
    "depth_m":  {"feeder": 1.00, "dist": 0.90, "drop": 0.60},
    "ducts":    {"feeder": 2,    "dist": 2,    "drop": 1},
    # part des tronçons revêtus posés sous trottoir (le reste : sous chaussée)
    "sidewalk_share": {"feeder": 0.30, "dist": 0.50, "drop": 0.80},
    "lateral_cap_m": 20.0,        # raccord route→foyer plafonné par foyer (m) ; au-delà : façade / privé
    "lateral_paved_share": 0.5,   # part du raccord latéral sous trottoir/revêtement
    "overcut_m": 0.20,            # sur-largeur de démolition/réfection de chaque côté
    "rock_pct": 10.0,             # % du volume de fouille en terrain rocheux
    "chamber_spacing_m": 150.0,   # chambre de tirage tous les X m (transport + distribution)
    "crossings_per_km": 2.0,      # traversées de route / oued en forage dirigé
    "crossing_len_m": 15.0,
    "aerial_share_drop": 0.0,     # part du branchement posée en aérien (0–1)
    "pole_spacing_m": 40.0,
    "contingency_pct": 10.0,      # aléas
    "studies_pct": 6.0,           # études, contrôle, supervision
}


def default_price_table() -> pd.DataFrame:
    return pd.DataFrame([{"code": k, "désignation": v[2], "unité": v[0], "prix_tnd": v[1]}
                         for k, v in DEFAULT_PRICES.items()])


def _level_edges(plan: dict) -> dict:
    """Arêtes exclusives par niveau : une tranchée partagée est comptée une fois, au niveau le plus haut."""
    e = plan["edges"]
    f = set(e["feeder"])
    d = set(e["dist"]) - f
    r = set(e["drop"]) - f - d
    return {"feeder": f, "dist": d, "drop": r}


def trench_by_level(plan: dict, router, lateral_cap_m: float = 20.0, lateral_paved_share: float = 0.5) -> dict:
    """Longueurs de tranchée par niveau : {niveau: {'paved': m, 'unpaved': m}}."""
    out = {}
    for lv, ids in _level_edges(plan).items():
        p, u = router.edge_lengths_by_paving(ids)
        out[lv] = {"paved": p, "unpaved": u}
    # raccord latéral route → foyer : hors voirie, en terrain naturel / trottoir
    lat = float(np.minimum(plan.get("lateral_each_m", np.zeros(0)), lateral_cap_m).sum())
    out["drop"]["paved"] += lat * lateral_paved_share
    out["drop"]["unpaved"] += lat * (1 - lateral_paved_share)
    return out


def estimate_civil(plan: dict, router, params: dict | None = None, prices: dict | None = None) -> dict:
    """Métré + chiffrage du génie civil.

    prices : {code: prix TND} (surcharge partielle autorisée). params : voir DEFAULT_PARAMS.
    """
    if not plan.get("routed") or router is None:
        raise ValueError("Le génie civil détaillé nécessite le tracé sur routes (OSM ou rues fictives).")
    P = {**DEFAULT_PARAMS, **(params or {})}
    for k in ("width_m", "depth_m", "ducts", "sidewalk_share"):
        P[k] = {**DEFAULT_PARAMS[k], **(params or {}).get(k, {})}
    price = {k: v[1] for k, v in DEFAULT_PRICES.items()}
    price.update(prices or {})
    unit = {k: v[0] for k, v in DEFAULT_PRICES.items()}
    label = {k: v[2] for k, v in DEFAULT_PRICES.items()}

    tr = trench_by_level(plan, router, P["lateral_cap_m"], P["lateral_paved_share"])
    q = {k: 0.0 for k in DEFAULT_PRICES}
    per_level = {}
    for lv in LEVELS:
        paved, unpaved = tr[lv]["paved"], tr[lv]["unpaved"]
        aerial = 0.0
        if lv == "drop" and P["aerial_share_drop"] > 0:          # l'aérien remplace une part de la tranchée du branchement
            s = float(np.clip(P["aerial_share_drop"], 0, 1))
            aerial = s * (paved + unpaved)
            paved, unpaved = paved * (1 - s), unpaved * (1 - s)
        length = paved + unpaved
        w, h = P["width_m"][lv], P["depth_m"][lv]
        side = P["sidewalk_share"][lv]
        road_m, walk_m = paved * (1 - side), paved * side
        brk_w = w + 2 * P["overcut_m"]
        brk_road, brk_walk = road_m * brk_w, walk_m * brk_w          # surfaces cassées (m²)
        vol = length * w * h
        rock = vol * P["rock_pct"] / 100.0
        duct_vol = length * P["ducts"][lv] * math.pi * 0.0275 ** 2   # Ø55 mm extérieur
        sand = length * w * 0.20
        backfill = max(vol - sand - duct_vol, 0.0)
        surplus = sand + duct_vol                                    # volume remplacé par sable + conduits
        # chambres
        spacing = P["chamber_spacing_m"]
        n_ch = math.ceil(length / spacing) if lv != "drop" and length > 0 else 0
        # traversées en forage dirigé
        cross_m = 0.0
        if lv != "drop":
            cross_m = length / 1000.0 * P["crossings_per_km"] * P["crossing_len_m"]
        q["demolition_chaussee"] += brk_road
        q["demolition_trottoir"] += brk_walk
        q["reprise_chaussee"] += brk_road
        q["reprise_trottoir"] += brk_walk
        q["fouille_terrain"] += vol - rock
        q["fouille_rocheuse"] += rock
        q["lit_sable"] += sand
        q["remblai"] += backfill
        q["evacuation"] += surplus
        q["conduit_pehd"] += length * P["ducts"][lv]
        q["grillage"] += length
        q["forage_dirige"] += cross_m
        q["chambre_l2t"] += n_ch
        q["signalisation"] += length / 1000.0
        if aerial:
            q["poteau"] += math.ceil(aerial / P["pole_spacing_m"])
            q["cable_aerien_pose"] += aerial
        per_level[lv] = {"tranchée_m": length, "dont_revêtue_m": paved, "dont_terre_m": unpaved, "aérien_m": aerial,
                         "casse_chaussée_m2": brk_road, "casse_trottoir_m2": brk_walk, "fouille_m3": vol,
                         "chambres": n_ch, "forage_m": cross_m}
    q["chambre_l1t"] += plan["n_nap"]           # une chambre / un boîtier par NAP
    q["chambre_l2t"] += plan["n_fdh"]           # une chambre FDH

    rows = [{"code": k, "désignation": label[k], "unité": unit[k], "quantité": q[k], "prix_tnd": price[k],
             "montant_tnd": q[k] * price[k]} for k in DEFAULT_PRICES if q[k] > 1e-9]
    boq = pd.DataFrame(rows)
    sub = float(boq["montant_tnd"].sum()) if len(boq) else 0.0
    cont, stud = sub * P["contingency_pct"] / 100.0, sub * P["studies_pct"] / 100.0
    total = sub + cont + stud
    trench_total = sum(v["tranchée_m"] for v in per_level.values())
    broken = q["demolition_chaussee"] + q["demolition_trottoir"]
    return {"boq": boq, "subtotal_tnd": sub, "contingency_tnd": cont, "studies_tnd": stud, "total_tnd": total,
            "per_level": per_level, "trench_m": trench_total, "broken_m2": broken,
            "broken_road_m2": q["demolition_chaussee"], "broken_walk_m2": q["demolition_trottoir"],
            "excavation_m3": q["fouille_terrain"] + q["fouille_rocheuse"], "chambers": q["chambre_l1t"] + q["chambre_l2t"],
            "cost_per_home_tnd": total / max(plan["n_homes"], 1), "cost_per_m_tnd": total / max(trench_total, 1.0),
            "params": P}
