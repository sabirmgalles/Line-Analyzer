"""API FastAPI d'AI Line Analyzer.

Lancer :  uvicorn api.main:app --reload --port 8000
Docs interactives : http://localhost:8000/docs
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import COPPER_CLASSES, FIBER_CLASSES, FIBER_HISTORY_KINDS, HISTORY_KINDS, RADIO_CLASSES
from core import ai_planning, assistant, civil, feedback, inference, optical, planning as P
from core.osm import RoadGraph
from core.features import COPPER_BASE
from data_generator import (simulate_copper_line, simulate_fiber_link, simulate_history, simulate_radio_kpi,
                            simulate_rx_history)

app = FastAPI(title="AI Line Analyzer API", version="1.0",
              description="Diagnostic IA de lignes cuivre, fibre optique et radio 4G/5G/6G, + planification.")


# ----------------------------------------------------------------------------- schémas
class History(BaseModel):
    snr_db: list[float] = Field(..., min_length=14)
    attenuation_db: list[float]
    crc_errors_per_min: list[float]


class CopperRequest(BaseModel):
    voltage_mean_v: float = 48.0
    voltage_min_v: float = 47.7
    voltage_std_v: float = 0.08
    voltage_max_step_v: float = 0.25
    impedance_ohm: float = 100.0
    snr_db: float = 35.0
    attenuation_db: float = 15.0
    crc_errors_per_min: float = 2.0
    noise_floor_dbm_hz: float = -133.0
    impulse_noise_ratio: float = 0.005
    tdr: list[float] = Field(..., min_length=50, description="Amplitudes TDR de 0 à 5,5 km (ré-échantillonnées si besoin)")
    history: History | None = None


class FiberRequest(BaseModel):
    otdr: list[float] = Field(..., min_length=200, description="Trace OTDR (dB) de 0 à 20 km")
    expected_length_m: float
    fiber_att_db_km: float = 0.22
    lambda_ratio: float = 1.0
    rx_power_dbm: float | None = None
    tx_power_dbm: float = 3.0


class FiberHistoryRequest(BaseModel):
    rx_dbm: list[float] = Field(..., min_length=14, description="Puissance reçue ONU (dBm), 1 mesure par jour")
    sensitivity_dbm: float = -28.0


class OpticalBudgetRequest(BaseModel):
    length_km: float = Field(..., gt=0, le=80)
    splitters: list[str] = Field(default_factory=lambda: ["1:4", "1:8"], description="ex. ['1:4','1:8']")
    n_connectors: int = 4
    n_splices: int = 6
    connector_db: float = 0.3
    splice_db: float = 0.1
    extra_loss_db: float = 0.0
    pon_class: Literal["B+", "C+"] = "B+"
    safety_margin_db: float = 3.0
    measured_rx_dbm: float | None = Field(None, description="Mesure power-meter à comparer au calcul")


class RadioRequest(BaseModel):
    tech_id: int = Field(1, ge=0, le=2, description="0 = 4G, 1 = 5G, 2 = 6G")
    rsrp_dbm: float
    rsrq_db: float
    sinr_db: float
    rssi_dbm: float
    bler_pct: float
    cqi: float
    throughput_mbps: float
    n_neighbors: float
    tac_changes_per_h: float = 0
    rat_fallback: int = 0
    cipher_null: int = 0


class RadioPlanRequest(BaseModel):
    tech: str = "5G NR (3,5 GHz)"
    environment: str = "Urbain"
    coverage: str = "Indoor léger"
    radius_km: float = Field(3.0, gt=0.1, le=30)
    pop_density_km2: float | None = None
    market_share: float = 0.4
    gb_month: float | None = None


class FtthPlanRequest(BaseModel):
    radius_m: float = Field(1500.0, gt=100, le=10000)
    homes_per_km2: float = 600.0
    nap_capacity: int = 8
    fdh_naps: int = 4
    route_factor: float = 1.25
    olt_class_db: float = 28.0


# ----------------------------------------------------------------------------- routes
@app.get("/health")
def health():
    return {"status": "ok", "models_ready": inference.models_ready()}


def _need_models():
    if not inference.models_ready():
        raise HTTPException(503, "Modèles absents : exécutez `python train_model.py`.")


@app.post("/analyze/copper")
def analyze_copper(req: CopperRequest):
    _need_models()
    return inference.analyze_copper(req.model_dump())


@app.post("/analyze/fiber")
def analyze_fiber(req: FiberRequest):
    _need_models()
    return inference.analyze_fiber(req.model_dump())


@app.post("/analyze/fiber-history")
def analyze_fiber_history(req: FiberHistoryRequest):
    _need_models()
    return inference.analyze_fiber_history(req.model_dump())


@app.get("/simulate/fiber-history")
def simulate_fiber_history(kind: Literal[tuple(FIBER_HISTORY_KINDS)] = "stable", seed: int = 0):  # type: ignore[valid-type]
    h = simulate_rx_history(kind, np.random.default_rng(seed))
    return {"rx_dbm": h["rx_dbm"].tolist(), "_truth": {"kind": kind, "label": h["label"], "ttf_days": h["ttf_days"]}}


@app.post("/plan/optical-budget")
def plan_optical_budget(req: OpticalBudgetRequest):
    try:
        return optical.budget_from_payload(req.model_dump())
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.post("/analyze/radio")
def analyze_radio(req: RadioRequest):
    _need_models()
    return inference.analyze_radio(req.model_dump())


@app.get("/simulate/copper")
def simulate_copper(scenario: Literal[tuple(COPPER_CLASSES)] = "healthy",  # type: ignore[valid-type]
                    length_m: float = 1500, trend: Literal[tuple(HISTORY_KINDS)] = "stable",  # type: ignore[valid-type]
                    seed: int = 0):
    """Fabrique une mesure de démonstration prête à être envoyée à /analyze/copper."""
    rng = np.random.default_rng(seed)
    rec, tr = simulate_copper_line(scenario, rng, length_m=length_m)
    h = simulate_history(trend, rng)
    body = {k: rec[k] for k in COPPER_BASE}
    body["tdr"] = tr.tolist()
    body["history"] = {k: h[k].tolist() for k in ("snr_db", "attenuation_db", "crc_errors_per_min")}
    body["_truth"] = {"label": scenario, "fault_distance_m": None if math.isnan(rec["fault_distance_m"]) else rec["fault_distance_m"]}
    return body


@app.get("/simulate/fiber")
def simulate_fiber(scenario: Literal[tuple(FIBER_CLASSES)] = "healthy",  # type: ignore[valid-type]
                   length_m: float = 6000, seed: int = 0):
    rec, tr = simulate_fiber_link(scenario, np.random.default_rng(seed), length_m=length_m)
    return {"otdr": tr.tolist(), "expected_length_m": rec["expected_length_m"], "fiber_att_db_km": rec["fiber_att_db_km"],
            "lambda_ratio": rec["lambda_ratio"], "rx_power_dbm": rec["rx_power_dbm"], "tx_power_dbm": rec["tx_power_dbm"],
            "_truth": {"label": scenario, "fault_distance_m": None if math.isnan(rec["fault_distance_m"]) else rec["fault_distance_m"]}}


@app.get("/simulate/radio")
def simulate_radio(scenario: Literal[tuple(RADIO_CLASSES)] = "healthy",  # type: ignore[valid-type]
                   tech_id: int = 1, seed: int = 0):
    kpi = simulate_radio_kpi(scenario, tech_id, np.random.default_rng(seed))
    return {**{k: v for k, v in kpi.items() if k != "label"}, "_truth": {"label": scenario}}


@app.get("/plan/catalog")
def plan_catalog():
    return {"technologies": {k: v["note"] for k, v in P.TECHS.items()}, "environnements": list(P.ENVIRONMENTS),
            "couvertures": list(P.COVERAGE_TYPES)}


@app.post("/plan/radio")
def plan_radio(req: RadioPlanRequest):
    if req.tech not in P.TECHS or req.environment not in P.ENVIRONMENTS or req.coverage not in P.COVERAGE_TYPES:
        raise HTTPException(422, "Technologie, environnement ou couverture inconnu : voir /plan/catalog")
    area = math.pi * req.radius_km ** 2
    pop = req.pop_density_km2 if req.pop_density_km2 is not None else P.ENVIRONMENTS[req.environment]["pop"]
    r = P.dimension_radio(req.tech, req.environment, req.coverage, area, pop, req.market_share, req.gb_month)
    xy = P.hex_layout(req.radius_km * 1000, r["sites_total"])
    margin = P.coverage_grid(xy, req.tech, req.environment, req.coverage, req.radius_km * 1000)
    r["sites_placed"] = len(xy)
    r["capex_usd"] = len(xy) * P.TECHS[req.tech]["capex_site"]      # CAPEX sur le placement réel
    r["coverage_pct"] = P.coverage_pct(margin)
    r["sites_xy_m"] = xy[:500].round(1).tolist()
    return r


@app.post("/plan/ftth")
def plan_ftth(req: FtthPlanRequest):
    homes = P.synthetic_homes(req.radius_m, req.homes_per_km2)
    plan = P.plan_ftth(homes, (0.0, 0.0), req.nap_capacity, req.fdh_naps, req.route_factor, olt_class_db=req.olt_class_db)
    capex = P.ftth_capex(plan)
    keep = ("n_homes", "n_nap", "n_fdh", "olt_ports", "drop_m", "dist_m", "feeder_m", "worst_loss_db", "mean_loss_db",
            "margin_db", "pct_within_budget")
    return {**{k: plan[k] for k in keep}, "capex_tnd": capex}


class CivilRequest(BaseModel):
    radius_m: float = Field(1200, ge=300, le=4000)
    homes_per_km2: float = Field(600, ge=50, le=6000)
    rock_pct: float = Field(10, ge=0, le=100)
    aerial_share_drop: float = Field(0, ge=0, le=1)
    contingency_pct: float = Field(10, ge=0, le=40)
    prices: dict[str, float] = Field(default_factory=dict, description="Surcharge du BPU : {code: prix TND}, voir /plan/civil/prices")


@app.get("/plan/civil/prices")
def civil_prices():
    """Bordereau de prix unitaires par défaut (TND, indicatif)."""
    return civil.default_price_table().to_dict(orient="records")


@app.post("/plan/civil")
def plan_civil(req: CivilRequest):
    """Métré + chiffrage du génie civil FTTH sur un réseau de rues fictif (la version OSM est dans l'application)."""
    bad = set(req.prices) - set(civil.DEFAULT_PRICES)
    if bad:
        raise HTTPException(422, f"Codes de prix inconnus : {sorted(bad)}")
    g = RoadGraph.synthetic_grid(req.radius_m * 1.1)
    plan = P.plan_ftth(P.synthetic_homes(req.radius_m, req.homes_per_km2), (0.0, 0.0), router=g)
    r = civil.estimate_civil(plan, g, {"rock_pct": req.rock_pct, "aerial_share_drop": req.aerial_share_drop,
                                       "contingency_pct": req.contingency_pct}, req.prices)
    r["boq"] = r["boq"].round(2).to_dict(orient="records")
    r.pop("params")
    return r


class CivilPredictRequest(BaseModel):
    radius_m: float = Field(700, ge=100, le=5000)
    density_km2: float = Field(800, ge=20, le=8000)
    rock_pct: float = Field(10, ge=0, le=100)
    aerial_share: float = Field(0, ge=0, le=1)
    sw_feeder: float = Field(0.3, ge=0, le=1)
    sw_dist: float = Field(0.5, ge=0, le=1)
    sw_drop: float = Field(0.8, ge=0, le=1)


@app.post("/plan/civil/predict")
def civil_predict(req: CivilPredictRequest):
    """Coût de génie civil prédit par le modèle IA (instantané, avec intervalle à 80 %)."""
    if not ai_planning.surrogate_ready():
        raise HTTPException(503, "Modèle absent : lancez python train_model.py")
    return ai_planning.predict_civil(**req.model_dump())


class SitesOptRequest(RadioPlanRequest):
    target_pct: float = Field(98, ge=80, le=99.5)


@app.post("/plan/radio/optimize")
def radio_optimize(req: SitesOptRequest):
    """Placement optimisé des sites (recouvrement glouton) comparé au maillage hexagonal."""
    if req.tech not in P.TECHS or req.environment not in P.ENVIRONMENTS or req.coverage not in P.COVERAGE_TYPES:
        raise HTTPException(422, "Technologie, environnement ou couverture inconnu : voir /plan/catalog")
    pop = req.pop_density_km2 if req.pop_density_km2 is not None else P.ENVIRONMENTS[req.environment]["pop"]
    r = P.dimension_radio(req.tech, req.environment, req.coverage, math.pi * req.radius_km ** 2, pop, req.market_share, req.gb_month)
    c = ai_planning.compare_site_plans(req.radius_km * 1000, r["cell_radius_m"], r["sites_total"], req.target_pct / 100)
    return {"sites_opt": c["sites_opt"], "sites_hex": c["sites_hex"], "cov_opt": c["cov_opt"], "cov_hex": c["cov_hex"],
            "saving_pct": c["saving_pct"], "sites_opt_xy_m": c["opt_xy"][:500].round(1).tolist()}


class AssistantRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    results: dict[str, dict] = Field(default_factory=dict, description="Dernières analyses : clés copper, fiber, fiber_history, radio, budget")
    use_llm: bool = False


@app.post("/assistant")
def assistant_ask(req: AssistantRequest):
    """Assistant : répond en français ou en dialecte tunisien à partir des résultats d'analyse fournis."""
    bad = set(req.results) - set(assistant.DOMAINS)
    if bad:
        raise HTTPException(422, f"Clés inconnues : {sorted(bad)}")
    return assistant.ask(req.question, req.results, list(req.results), None, req.use_llm)


class FeedbackRequest(BaseModel):
    domain: str = Field(..., description="copper | fiber | radio")
    features: dict[str, float] = Field(..., description="Le champ `features` renvoyé par /analyze/*")
    predicted: str
    confidence: float = Field(0.0, ge=0, le=1)
    confirmed: str = Field(..., description="Défaut réellement constaté")
    distance_pred_m: float | None = None
    distance_true_m: float | None = Field(None, ge=0)
    tech: str = ""


@app.post("/feedback")
def post_feedback(req: FeedbackRequest):
    """Enregistre le défaut réel confirmé par un technicien (alimente le ré-entraînement)."""
    try:
        fid = feedback.add_feedback(req.domain, req.features, req.predicted, req.confidence, req.confirmed,
                                    req.distance_pred_m, req.distance_true_m, req.tech)
    except (ValueError, KeyError) as e:
        raise HTTPException(422, str(e))
    return {"id": fid}


@app.get("/feedback/stats/{domain}")
def feedback_stats(domain: str):
    """Précision réelle mesurée sur les retours terrain."""
    if domain not in feedback.CFG:
        raise HTTPException(404, "Domaine inconnu : copper | fiber | radio")
    lp = feedback.live_performance(domain)
    lp.pop("confusion", None)
    return lp


@app.post("/feedback/retrain/{domain}")
def feedback_retrain(domain: str):
    """Ré-entraînement contrôlé : déployé seulement si meilleur sur des retours mis de côté, sans régression."""
    if domain not in feedback.CFG:
        raise HTTPException(404, "Domaine inconnu : copper | fiber | radio")
    return feedback.retrain(domain)
