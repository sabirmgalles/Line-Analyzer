"""Calculateur de budget optique (liaison point à point ou PON GPON).

Valeurs typiques UIT-T G.652.D / G.984.2 ; à remplacer par les valeurs de vos fiches
techniques (SFP/OLT/ONT, câble, coupleurs) pour un dimensionnement contractuel.
"""
from __future__ import annotations

# Atténuation linéique typique (dB/km) par longueur d'onde (nm)
FIBER_ATT_DB_KM = {1310: 0.35, 1490: 0.25, 1550: 0.22, 1625: 0.24}

# Pertes d'insertion typiques des coupleurs PLC (dB, incluant l'excès)
SPLITTER_LOSS_DB = {"1:2": 3.8, "1:4": 7.4, "1:8": 10.7, "1:16": 13.8, "1:32": 17.5, "1:64": 21.0}

# Classes GPON (G.984.2 amendé) : puissances d'émission OLT/ONU, sensibilité et saturation
PON_CLASSES = {
    "B+": dict(budget_db=28.0, olt_tx=(1.5, 5.0), onu_tx=(0.5, 5.0), onu_sens=-28.0, onu_overload=-8.0,
               olt_sens=-28.0, olt_overload=-8.0),
    "C+": dict(budget_db=32.0, olt_tx=(3.0, 7.0), onu_tx=(0.5, 5.0), onu_sens=-32.0, onu_overload=-12.0,
               olt_sens=-32.0, olt_overload=-12.0),
}
CONNECTOR_DB = {"SC/APC (propre)": 0.25, "SC/APC (usage)": 0.4, "SC/UPC": 0.3}


def optical_budget(length_km: float, splitters: list[str] | None = None, n_connectors: int = 4,
                   n_splices: int = 6, connector_db: float = 0.3, splice_db: float = 0.1,
                   pon_class: str = "B+", extra_loss_db: float = 0.0, safety_margin_db: float = 3.0,
                   fiber_att_ds: float | None = None, fiber_att_us: float | None = None) -> dict:
    """Budget optique descendant (1490 nm) et montant (1310 nm) d'une liaison OLT → ONT.

    Marge = puissance reçue au pire cas (Tx min - pertes) - sensibilité du récepteur.
    Saturation = puissance reçue au meilleur cas (Tx max - pertes) > seuil de surcharge.
    """
    splitters = splitters or []
    unknown = [s for s in splitters if s not in SPLITTER_LOSS_DB]
    if unknown:
        raise ValueError(f"Coupleur inconnu : {unknown}. Valeurs : {list(SPLITTER_LOSS_DB)}")
    c = PON_CLASSES[pon_class]
    att_ds = FIBER_ATT_DB_KM[1490] if fiber_att_ds is None else fiber_att_ds
    att_us = FIBER_ATT_DB_KM[1310] if fiber_att_us is None else fiber_att_us
    fixed = {
        "Coupleurs": sum(SPLITTER_LOSS_DB[s] for s in splitters),
        "Connecteurs": n_connectors * connector_db,
        "Soudures": n_splices * splice_db,
        "Pertes supplémentaires": extra_loss_db,
    }
    fixed_total = sum(fixed.values())

    def direction(att, tx, sens, overload):
        loss = fixed_total + att * length_km
        rx_min, rx_max = tx[0] - loss, tx[1] - loss
        margin = rx_min - sens
        reach = max(0.0, (tx[0] - sens - safety_margin_db - fixed_total) / att)
        status = ("SATURATION" if rx_max > overload else "HORS BUDGET" if margin < 0
                  else "MARGE FAIBLE" if margin < safety_margin_db else "OK")
        return {"fiber_loss_db": att * length_km, "total_loss_db": loss, "rx_min_dbm": rx_min, "rx_max_dbm": rx_max,
                "sensitivity_dbm": sens, "overload_dbm": overload, "margin_db": margin,
                "max_reach_km": reach, "status": status}

    ds = direction(att_ds, c["olt_tx"], c["onu_sens"], c["onu_overload"])
    us = direction(att_us, c["onu_tx"], c["olt_sens"], c["olt_overload"])
    ds["fixed_losses"] = {**fixed, "Fibre": ds["fiber_loss_db"]}
    us["fixed_losses"] = {**fixed, "Fibre": us["fiber_loss_db"]}
    worst = min(ds["margin_db"], us["margin_db"])
    status = ("SATURATION" if "SATURATION" in (ds["status"], us["status"])
              else "HORS BUDGET" if worst < 0 else "MARGE FAIBLE" if worst < safety_margin_db else "OK")
    return {"pon_class": pon_class, "class_budget_db": c["budget_db"], "downstream": ds, "upstream": us,
            "worst_margin_db": worst, "status": status,
            "within_class_budget": ds["total_loss_db"] <= c["budget_db"]}


def diagnose_measurement(expected_rx_dbm: float, measured_rx_dbm: float, tolerance_db: float = 1.5) -> dict:
    """Compare la puissance mesurée (power-meter) à la puissance calculée et oriente le diagnostic.

    `expected_rx_dbm` = puissance attendue (milieu de la plage d'émission) ; la dispersion
    des émetteurs est absorbée par `tolerance_db`."""
    excess = expected_rx_dbm - measured_rx_dbm            # > 0 : plus de pertes que prévu
    if excess > 6:
        cause = ("Perte très supérieure au calcul : rupture partielle, coupleur mal raccordé, connecteur très sale "
                 "ou mauvais coupleur (ratio) installé. Lancer un OTDR.")
        level = "critical"
    elif excess > tolerance_db:
        cause = ("Pertes excédentaires : nettoyer et inspecter les connecteurs, rechercher une macro-courbure "
                 "ou une soudure défectueuse (OTDR).")
        level = "warning"
    elif excess < -3:
        cause = "Puissance supérieure au calcul : vérifier les hypothèses (coupleur, longueur) ou le risque de saturation."
        level = "warning"
    else:
        cause, level = "Mesure conforme au calcul.", "info"
    return {"excess_loss_db": excess, "level": level, "message": cause}


def budget_from_payload(p: dict) -> dict:
    """Point d'entrée commun API / interface (dict JSON -> budget + diagnostic de mesure éventuel)."""
    res = optical_budget(p["length_km"], p.get("splitters"), p.get("n_connectors", 4), p.get("n_splices", 6),
                         p.get("connector_db", 0.3), p.get("splice_db", 0.1), p.get("pon_class", "B+"),
                         p.get("extra_loss_db", 0.0), p.get("safety_margin_db", 3.0))
    if p.get("measured_rx_dbm") is not None:
        d = res["downstream"]
        expected = 0.5 * (d["rx_min_dbm"] + d["rx_max_dbm"])
        tol = 1.5 + 0.5 * (d["rx_max_dbm"] - d["rx_min_dbm"])
        res["measurement"] = {**diagnose_measurement(expected, p["measured_rx_dbm"], tol), "expected_rx_dbm": expected}
    return res
