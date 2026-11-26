"""Tests de bout en bout (exécuter : pytest -q). Entraîne les modèles si besoin."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import COPPER_CLASSES, FIBER_CLASSES, RADIO_CLASSES  # noqa: E402
from core import inference, planning as P  # noqa: E402
from core.features import COPPER_BASE  # noqa: E402
from data_generator import (simulate_copper_line, simulate_fiber_link, simulate_history,  # noqa: E402
                            simulate_radio_kpi)


@pytest.fixture(scope="session", autouse=True)
def models():
    if not inference.models_ready():
        import train_model
        train_model.main()


def _copper(label, seed=1, trend="stable"):
    rng = np.random.default_rng(seed)
    rec, tr = simulate_copper_line(label, rng)
    h = simulate_history(trend, rng)
    return rec, inference.analyze_copper({**{k: rec[k] for k in COPPER_BASE}, "tdr": tr.tolist(),
                                          "history": {k: h[k].tolist() for k in ("snr_db", "attenuation_db", "crc_errors_per_min")}})


@pytest.mark.parametrize("label", COPPER_CLASSES)
def test_copper_classification(label):
    ok = sum(_copper(label, s)[1]["fault"]["label"] == label for s in range(10))
    assert ok >= 8, f"{label}: {ok}/10"


def test_copper_distance_and_security():
    rec, res = _copper("open_cut", seed=3)
    assert abs(res["distance_m"] - rec["fault_distance_m"]) < 150
    _, tap = _copper("wiretap", seed=5)
    assert tap["tscm"]["risk_level"] in ("MOYEN", "ÉLEVÉ")
    _, ok = _copper("healthy", seed=5)
    assert ok["tscm"]["risk_level"] == "FAIBLE" and ok["health_score"] > 80


def test_predictive_maintenance():
    _, res = _copper("healthy", seed=2, trend="accelerating")
    assert res["maintenance"]["label"] in ("imminent", "degrading")


@pytest.mark.parametrize("label", FIBER_CLASSES)
def test_fiber(label):
    ok = 0
    for s in range(10):
        rec, tr = simulate_fiber_link(label, np.random.default_rng(s))
        r = inference.analyze_fiber({"otdr": tr.tolist(), "expected_length_m": rec["expected_length_m"],
                                     "lambda_ratio": rec["lambda_ratio"], "rx_power_dbm": rec["rx_power_dbm"],
                                     "tx_power_dbm": rec["tx_power_dbm"], "fiber_att_db_km": rec["fiber_att_db_km"]})
        ok += r["fault"]["label"] == label
    assert ok >= 7, f"{label}: {ok}/10"


@pytest.mark.parametrize("label", RADIO_CLASSES)
def test_radio(label):
    ok = sum(inference.analyze_radio(simulate_radio_kpi(label, 1, np.random.default_rng(s)))["fault"]["label"] == label
             for s in range(10))
    assert ok >= 9


def test_planning_radio_coverage():
    for tech in P.TECHS:
        r = P.dimension_radio(tech, "Urbain", "Indoor léger", math.pi * 4, 3000)
        xy = P.hex_layout(2000, r["sites_total"])
        assert P.coverage_pct(P.coverage_grid(xy, tech, "Urbain", "Indoor léger", 2000)) > 95


def test_planning_ftth():
    homes = P.synthetic_homes(1000, 500)
    plan = P.plan_ftth(homes)
    assert plan["nap_labels"].shape[0] == len(homes)
    assert np.bincount(plan["nap_labels"]).max() <= 8          # contrainte de capacité respectée
    assert np.bincount(plan["fdh_labels"]).max() <= 4


# ----------------------------------------------------------------------------- fibre : budget, historique Rx, routage
def test_optical_budget():
    from core import optical
    ok = optical.optical_budget(6, ["1:4", "1:8"])
    assert ok["status"] == "OK" and ok["downstream"]["margin_db"] > 5
    assert optical.optical_budget(40, ["1:4", "1:8"])["status"] == "HORS BUDGET"
    assert optical.optical_budget(0.2, ["1:2"])["status"] == "SATURATION"
    with pytest.raises(ValueError):
        optical.optical_budget(5, ["1:3"])
    m = optical.budget_from_payload({"length_km": 8, "splitters": ["1:4", "1:8"], "measured_rx_dbm": -27.0})
    assert m["measurement"]["level"] == "critical"


@pytest.mark.parametrize("kind,expected", [("stable", {"stable"}), ("step", {"stable"}),
                                           ("accelerating", {"imminent", "degrading"})])
def test_fiber_history(kind, expected):
    from data_generator import simulate_rx_history
    ok = sum(inference.analyze_fiber_history({"rx_dbm": simulate_rx_history(kind, np.random.default_rng(s))["rx_dbm"].tolist()})
             ["label"] in expected for s in range(12))
    assert ok >= 9, f"{kind}: {ok}/12"


def test_road_routing():
    from core.osm import RoadGraph
    g = RoadGraph.synthetic_grid(1200, 110)
    homes = P.synthetic_homes(1200, 300)
    plan = P.plan_ftth(homes, (0.0, 0.0), router=g)
    nap, lab = plan["nap_xy"], plan["nap_labels"]
    routed, _ = g.routes(nap, homes, lab)
    assert (routed >= np.linalg.norm(homes - nap[lab], axis=1) - 1e-6).all()   # jamais plus court que la ligne droite
    assert plan["trench_m"]["total"] < plan["drop_m"] + plan["dist_m"] + plan["feeder_m"]   # tranchées mutualisées
    assert P.ftth_capex(plan)["Tranchées / pose"] > 0


def test_osm_parsing():
    from core.osm import RoadGraph, parse_buildings
    data = {"elements": [
        {"type": "node", "id": 1, "lat": 34.425, "lon": 8.784}, {"type": "node", "id": 2, "lat": 34.425, "lon": 8.786},
        {"type": "node", "id": 3, "lat": 34.427, "lon": 8.785}, {"type": "way", "id": 9, "nodes": [1, 2, 3]}]}
    g = RoadGraph.from_overpass(data, 34.425, 8.785)
    assert len(g.nodes) == 3 and len(g.edges) == 2
    assert parse_buildings({"elements": [{"type": "way", "center": {"lat": 1.0, "lon": 2.0}}, {"type": "way"}]}).shape == (1, 2)


def test_civil_works():
    from core import civil
    from core.osm import RoadGraph
    g = RoadGraph.synthetic_grid(1200, 110)
    plan = P.plan_ftth(P.synthetic_homes(1200, 300), (0.0, 0.0), router=g)
    r = civil.estimate_civil(plan, g)
    lv = r["per_level"]
    assert abs(r["trench_m"] - sum(v["tranchée_m"] for v in lv.values())) < 1e-6
    assert r["broken_m2"] > 0 and r["total_tnd"] > r["subtotal_tnd"] > 0
    base = r["total_tnd"]
    assert civil.estimate_civil(plan, g, {"aerial_share_drop": 0.5})["total_tnd"] < base     # aérien moins cher
    assert civil.estimate_civil(plan, g, {"rock_pct": 60})["total_tnd"] > base               # roche plus chère
    assert civil.estimate_civil(plan, g, prices={"fouille_terrain": 0.0})["total_tnd"] < base  # BPU modifiable
    assert P.ftth_capex(plan, civil_total=r["total_tnd"])["Génie civil (métré détaillé)"] == r["total_tnd"]
    with pytest.raises(ValueError):
        civil.estimate_civil(P.plan_ftth(P.synthetic_homes(500, 300)), None)


def test_osm_paving_tags():
    from core.osm import RoadGraph
    nodes = [{"type": "node", "id": i, "lat": 34.425 + i * 1e-3, "lon": 8.785} for i in (1, 2, 3)]
    data = {"elements": nodes + [{"type": "way", "id": 1, "nodes": [1, 2], "tags": {"highway": "residential"}},
                                 {"type": "way", "id": 2, "nodes": [2, 3], "tags": {"highway": "track", "surface": "dirt"}}]}
    g = RoadGraph.from_overpass(data, 34.425, 8.785)
    assert sorted(g.paved.tolist()) == [False, True]


def test_civil_surrogate():
    from core import ai_planning as A
    if not A.surrogate_ready():
        A.train_civil_surrogate(60, 6)
    lo, hi = A.predict_civil(700, 800), A.predict_civil(700, 800, rock_pct=55)
    assert hi["cost_per_home_tnd"] > lo["cost_per_home_tnd"] > 0        # plus de roche = plus cher
    assert lo["interval_tnd_per_home"][0] < lo["cost_per_home_tnd"] < lo["interval_tnd_per_home"][1]
    assert A.predict_civil(4000, 800)["out_of_domain"] == ["radius_m"]
    assert lo["metrics"]["mape_pct"] < 15


def test_site_optimization():
    from core import ai_planning as A
    c = A.compare_site_plans(3000, 800, 20, target=0.95)
    assert c["cov_opt"] >= 0.95 and c["sites_opt"] <= c["sites_hex"]


def test_deep_learning_optional():
    from core import deep
    if not deep.deep_ready():
        pytest.skip("PyTorch ou modèles deep absents")
    rec, tr = simulate_copper_line("open_cut", np.random.default_rng(3))
    r = deep.predict_tdr(tr)
    assert max(r["proba"], key=r["proba"].get) == "open_cut" and abs(r["distance_m"] - rec["fault_distance_m"]) < 250
    h = simulate_history("accelerating", np.random.default_rng(2))
    p = deep.predict_history(h["snr_db"], h["attenuation_db"], h["crc_errors_per_min"])
    assert max(p["proba"], key=p["proba"].get) in ("imminent", "degrading")


def test_assistant_bilingual():
    from core import assistant as A
    rec, res = _copper("open_cut", seed=3)
    ctx = {"copper": res}
    fr = A.ask("Où se trouve le défaut ?", ctx)
    tn = A.ask("Ouin el défaut ?", ctx)
    assert fr["lang"] == "fr" and tn["lang"] == "tn"
    assert f"{res['distance_m']:.0f} m" in fr["answer"] and f"{res['distance_m']:.0f} m" in tn["answer"]   # chiffres tirés du résultat
    assert "Priorité haute" in A.ask("Est-ce urgent ?", ctx)["answer"]
    assert "Marche à suivre" in A.ask("Que faire ?", ctx)["answer"] and "El étapes" in A.ask("Chnowa ndir ?", ctx)["answer"]
    assert "TDR" in A.ask("C'est quoi le TDR ?", ctx)["answer"]
    assert "aucun résultat" in A.ask("résumé", {})["answer"]


def test_assistant_dialect_is_latin_only():
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "core" / "assistant.py").read_text(encoding="utf-8")
    assert not re.search(r"[\u0400-\u06FF]", src)     # dialecte écrit uniquement en lettres latines


def test_feedback_loop(tmp_path, monkeypatch):
    """Boucle complète sur une copie des modèles : dérive -> retours -> ré-entraînement contrôlé -> retour arrière."""
    import shutil
    import joblib
    import pandas as pd
    from config import MODELS_DIR
    from core import feedback as F
    shutil.copytree(MODELS_DIR, tmp_path / "models")
    monkeypatch.setattr(F, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(F, "ARCHIVE", tmp_path / "models" / "archive")
    monkeypatch.setattr(F, "REGISTRY", tmp_path / "models" / "registry.json")
    monkeypatch.setattr(F, "DB_PATH", tmp_path / "fb.db")
    assert F.retrain("copper")["status"] == "insuffisant"                     # trop peu de retours : rien ne change
    F.simulate_field_feedback("copper", 90, seed=1)
    lp = F.live_performance("copper")
    assert lp["n"] == 90 and lp["accuracy"] < 0.9                             # la dérive dégrade la précision réelle
    with pytest.raises(ValueError):
        F.add_feedback("copper", {k: 0.0 for k in F.CFG["copper"]["features"]}, "healthy", 0.5, "inconnu")
    # jeu de contrôle indépendant, même dérive
    F.DB_PATH = tmp_path / "chk.db"
    F.simulate_field_feedback("copper", 120, seed=77)
    chk = F.load_feedback("copper")
    X, y = F._feature_matrix(chk, "copper"), chk["confirmed"].values
    F.DB_PATH = tmp_path / "fb.db"
    before = (joblib.load(tmp_path / "models" / "copper_fault.joblib")["model"].predict(X) == y).mean()
    rep = F.retrain("copper")
    after = (joblib.load(tmp_path / "models" / "copper_fault.joblib")["model"].predict(X) == y).mean()
    assert rep["promoted"] and rep["base_acc_after"] >= rep["base_acc_before"] - 0.01   # pas de régression sur l'origine
    assert after > before + 0.1                                               # gain réel sur des données jamais vues
    assert F.rollback("copper")["status"] == "restauré"
    assert abs((joblib.load(tmp_path / "models" / "copper_fault.joblib")["model"].predict(X) == y).mean() - before) < 1e-9
