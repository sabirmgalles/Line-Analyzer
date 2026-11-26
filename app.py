"""AI Line Analyzer - interface Streamlit.

Lancer :  streamlit run app.py

Onglets : 🔌 Ligne cuivre | 💡 Fibre optique | 📡 Radio 4G/5G/6G | 🛰️ Planification (carte satellite) | ℹ️ À propos
Les mesures sont SIMULÉES (data_generator.py) ou saisies manuellement ; l'analyse est faite
soit en local, soit via l'API FastAPI (case dans la barre latérale).
"""
from __future__ import annotations

import json
import time
import math
import subprocess
import sys

import folium
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_folium import st_folium

from config import (COPPER_CLASSES, FIBER_CLASSES, FIBER_HISTORY_KINDS, FIBER_RX_FAIL_DBM, HISTORY_KINDS,
                    LABELS_FR, MODELS_DIR, RADIO_CLASSES, RADIO_TECHS, ROOT)
from core import ai_planning, assistant, civil, deep, feedback, inference, optical, osm, planning as P, viz
from core.features import COPPER_BASE, COPPER_DEFAULTS
from data_generator import (simulate_copper_line, simulate_fiber_link, simulate_history,
                            simulate_radio_kpi, simulate_rx_history)

st.set_page_config(page_title="AI Line Analyzer", page_icon="📡", layout="wide")

ESRI_IMAGERY = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ESRI_LABELS = ("https://server.arcgisonline.com/ArcGIS/rest/services/Reference/"
               "World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}")
ESRI_ATTR = "Imagerie : Esri, Maxar, Earthstar Geographics, GIS User Community"
TREND_FR = {"stable": "Stable", "degrading": "Dégradation lente", "accelerating": "Dégradation accélérée",
            "step": "Chute brutale puis palier"}


# ============================================================================= utilitaires
def show_fig(fig):
    st.plotly_chart(fig, width="stretch")


def show_alerts(alerts: list[dict]):
    for a in alerts:
        msg = f"**{a['title']}**" + (f"  \n{a['detail']}" if a.get("detail") else "")
        {"critical": st.error, "warning": st.warning}.get(a["level"], st.info)(msg)


ASSISTANT_DOMAINS = {"/analyze/copper": "copper", "/analyze/fiber": "fiber", "/analyze/fiber-history": "fiber_history",
                     "/analyze/radio": "radio", "/plan/optical-budget": "budget"}


def remember(endpoint: str, res: dict):
    """Garde le dernier résultat de chaque analyse pour l'assistant ; mémorise laquelle a changé en dernier."""
    dom = ASSISTANT_DOMAINS.get(endpoint)
    if not dom:
        return
    last = st.session_state.setdefault("last", {})
    digest = st.session_state.setdefault("digest", {})
    d = hash(json.dumps(res, sort_keys=True, default=str))
    if dom in digest and digest[dom] != d:               # résultat modifié par l'utilisateur : devient « le plus récent »
        rec = st.session_state.setdefault("recent", [])
        if dom in rec:
            rec.remove(dom)
        rec.append(dom)
    digest[dom], last[dom] = d, res


def call(endpoint: str, payload: dict, local_fn):
    """Appelle l'API FastAPI si activée dans la barre latérale, sinon l'inférence locale."""
    if st.session_state.get("use_api"):
        r = requests.post(st.session_state["api_url"].rstrip("/") + endpoint, json=payload, timeout=60)
        r.raise_for_status()
        res = r.json()
    else:
        res = local_fn(payload)
    remember(endpoint, res)
    return res


def feedback_form(domain: str, res: dict):
    """Le technicien confirme ou corrige le défaut réel : alimente l'auto-apprentissage."""
    cfg = feedback.CFG[domain]
    pred = res["fault"]["label"]
    with st.expander("✅ Confirmer le défaut réel (auto-apprentissage)"):
        with st.form(f"fb_{domain}", clear_on_submit=False):
            real = st.selectbox("Défaut réellement constaté", cfg["classes"], index=cfg["classes"].index(pred),
                                format_func=lambda k: LABELS_FR.get(k, "Cellule saine" if k == "healthy" else k))
            dist_true = None
            if domain == "copper":
                d = st.number_input("Distance réelle du défaut (m, 0 = non relevée)", 0.0, 6000.0, 0.0, 5.0)
                dist_true = d if d > 0 and real != "healthy" else None
            tech = st.text_input("Technicien (facultatif)")
            if st.form_submit_button("Enregistrer le retour"):
                feedback.add_feedback(domain, res["features"], pred, res["fault"]["confidence"], real,
                                      res.get("distance_m"), dist_true, tech)
                st.success("Retour enregistré. Il servira au prochain ré-entraînement (onglet Auto-apprentissage).")
        st.caption("Les données de démonstration de cette application sont simulées : en usage réel, ne confirmez que ce qui a été constaté sur le terrain.")


def kpi_block(col, label: str, value: str, sub: str = ""):
    """Indicateur à texte long (st.metric tronque les libellés)."""
    col.markdown(f"<div style='font-size:.85rem;opacity:.7'>{label}</div>"
                 f"<div style='font-size:1.35rem;font-weight:600;line-height:1.25'>{value}</div>"
                 f"<div style='font-size:.8rem;opacity:.7'>{sub}</div>", unsafe_allow_html=True)


def new_measure_button(key: str) -> int:
    """Bouton 'nouvelle mesure' : incrémente un compteur qui fait varier la graine."""
    if st.button("🎲 Nouvelle mesure", key=f"btn_{key}", type="primary"):
        st.session_state[f"n_{key}"] = st.session_state.get(f"n_{key}", 0) + 1
    return st.session_state.get(f"n_{key}", 0)


# ============================================================================= barre latérale
def sidebar():
    st.sidebar.title("📡 AI Line Analyzer")
    st.sidebar.caption("Analyse intelligente de lignes cuivre, fibre et radio")
    st.sidebar.checkbox("Utiliser l'API FastAPI", key="use_api",
                        help="Sinon, les modèles sont appelés directement dans Streamlit.")
    st.sidebar.text_input("URL de l'API", "http://localhost:8000", key="api_url",
                          disabled=not st.session_state.get("use_api"))
    if st.session_state.get("use_api"):
        try:
            requests.get(st.session_state["api_url"].rstrip("/") + "/health", timeout=3).raise_for_status()
            st.sidebar.success("API joignable")
        except Exception:
            st.sidebar.error("API injoignable : lancez `uvicorn api.main:app`")
    metrics = MODELS_DIR / "metrics.json"
    if metrics.exists():
        m = json.loads(metrics.read_text())
        st.sidebar.divider()
        st.sidebar.caption("Performances (jeu de test simulé)")
        st.sidebar.write(f"Défaut cuivre : **{m['copper_fault_acc']:.1%}**  \n"
                         f"Fibre : **{m['fiber_acc']:.1%}**  \nRadio : **{m['radio_acc']:.1%}**  \n"
                         f"Maintenance : **{m['history_acc']:.1%}**")


# ============================================================================= onglet cuivre
MANUAL_SPEC = [  # clé, libellé, min, max, pas
    ("voltage_mean_v", "Tension moyenne (V)", 0.0, 60.0, 0.1), ("voltage_min_v", "Tension minimale (V)", 0.0, 60.0, 0.1),
    ("voltage_std_v", "Écart-type tension (V)", 0.0, 10.0, 0.01), ("voltage_max_step_v", "Saut de tension max (V)", 0.0, 20.0, 0.05),
    ("impedance_ohm", "Impédance (Ω)", 0.0, 25000.0, 1.0), ("snr_db", "SNR (dB)", 0.0, 70.0, 0.1),
    ("attenuation_db", "Atténuation @300 kHz (dB)", 0.0, 100.0, 0.1), ("crc_errors_per_min", "CRC / min", 0.0, 10000.0, 1.0),
    ("noise_floor_dbm_hz", "Plancher de bruit (dBm/Hz)", -160.0, -90.0, 0.5), ("impulse_noise_ratio", "Bruit impulsif (ratio)", 0.0, 1.0, 0.001),
]


def tab_copper():
    left, right = st.columns([1, 3], gap="large")
    with left:
        st.subheader("Mesure")
        scen = st.selectbox("Scénario simulé", COPPER_CLASSES, format_func=LABELS_FR.get, key="cu_scen")
        length = st.slider("Longueur de la ligne (m)", 50, 5000, 1500, 50, key="cu_len")
        trend = st.selectbox("Tendance des 30 derniers jours", HISTORY_KINDS, format_func=TREND_FR.get, key="cu_trend")
        n = new_measure_button("cu")
        seed = int(st.number_input("Graine", 0, 10**6, 7, key="cu_seed")) + n

        rng = np.random.default_rng(seed)
        rec, trace = simulate_copper_line(scen, rng, length_m=length)
        hist = simulate_history(trend, rng)
        measure = {k: float(rec[k]) for k in COPPER_BASE}

        with st.expander("✏️ Saisie manuelle / import TDR"):
            if st.checkbox("Remplacer les valeurs simulées", key="cu_manual"):
                sig = f"{scen}{length}{seed}"
                for k, label, lo, hi, step in MANUAL_SPEC:
                    measure[k] = st.number_input(label, lo, hi, float(np.clip(measure[k], lo, hi)), step, key=f"m_{k}_{sig}")
            up = st.file_uploader("Courbe TDR (CSV, 1 colonne d'amplitudes)", type=["csv", "txt"])
            if up is not None:
                trace = pd.read_csv(up, header=None).iloc[:, -1].astype(float).to_numpy()
                st.caption(f"{len(trace)} points importés (ré-échantillonnés sur 0–5,5 km).")

    payload = {**measure, "tdr": np.asarray(trace, dtype=float).tolist(),
               "history": {k: np.asarray(hist[k]).tolist() for k in ("snr_db", "attenuation_db", "crc_errors_per_min")}}
    try:
        res = call("/analyze/copper", payload, inference.analyze_copper)
    except Exception as e:
        st.error(f"Analyse impossible : {e}")
        return

    with right:
        c1, c2, c3, c4 = st.columns([1.4, 1.2, 1.2, 1.2])
        with c1:
            show_fig(viz.gauge(res["health_score"]))
        kpi_block(c2, "Diagnostic", res["fault"]["label_fr"], f"confiance {res['fault']['confidence']:.0%}")
        d = res["distance_m"]
        c3.metric("Distance du défaut", f"{d:.0f} m" if d is not None else "—",
                  f"± {res['distance_error_p90_m']:.0f} m (P90)" if d is not None else "ligne saine", delta_color="off")
        c4.metric("Risque TSCM", res["tscm"]["risk_level"], f"score {res['tscm']['risk_score']:.0%}", delta_color="off")
        c4.metric("Profil recommandé", res["modulation"]["recommended"])
        st.caption(f"Vérité terrain simulée : **{LABELS_FR[scen]}**" +
                   (f" à {rec['fault_distance_m']:.0f} m" if not math.isnan(rec["fault_distance_m"]) else ""))
        show_alerts(res["alerts"])
        feedback_form("copper", res)

        t1, t2, t3, t4, t5, t6 = st.tabs(["📈 TDR", "🔮 Maintenance prédictive", "📶 Modulation", "🕵️ Sécurité (TSCM)", "🔢 Mesures",
                                          "🧠 Deep learning"])
        with t1:
            show_fig(viz.tdr_figure(np.asarray(trace), d, res["distance_error_p90_m"]))
            show_fig(viz.proba_bar(res["fault"]["proba"]))
        with t2:
            m = res["maintenance"]
            if m:
                a, b, c = st.columns(3)
                a.metric("Tendance", m["label_fr"])
                b.metric("Jours avant coupure (IA)", f"{m['ttf_days_ml']:.0f} j")
                c.metric("Extrapolation linéaire", f"{m['ttf_days_linear']:.0f} j")
                show_fig(viz.history_figure(hist, m["ttf_days_ml"]))
            else:
                st.info("Pas d'historique fourni.")
        with t3:
            mod = res["modulation"]
            if mod["note"]:
                st.warning(mod["note"])
            show_fig(viz.rates_bar(mod["rates_mbps"], mod["recommended"]))
            st.caption(f"Confiance du modèle : {mod['confidence']:.0%}. Débits estimés par un modèle physique "
                       "(atténuation en √f, gap de Shannon 12 dB) à partir de l'atténuation, du SNR et des CRC.")
        with t4:
            ts = res["tscm"]
            a, b = st.columns(2)
            a.metric("Probabilité d'écoute (Random Forest)", f"{ts['wiretap_proba']:.0%}")
            b.metric("Score d'anomalie (Isolation Forest)", f"{ts['anomaly_score']:.0%}")
            if ts["indicators"]:
                st.write("**Indices relevés :**")
                for i in ts["indicators"]:
                    st.write(f"- {i}")
            else:
                st.success("Aucun indice de branchement ou d'écoute.")
            if ts["suspected_position_m"]:
                st.write(f"Position suspecte : ~{ts['suspected_position_m']:.0f} m")
        with t5:
            st.dataframe(pd.DataFrame({"valeur": res["features"]}).round(3), width="stretch")
        with t6:
            dv = res.get("deep")
            if not dv:
                st.info("Deep learning indisponible : installez PyTorch (`pip install torch`) puis relancez `python train_model.py`.")
            else:
                a, b, c = st.columns(3)
                a.metric("CNN 1D (courbe TDR brute)", dv["fault"]["label_fr"], f"confiance {dv['fault']['confidence']:.0%}", delta_color="off")
                b.metric("Distance (CNN)", f"{dv['distance_m']:.0f} m" if dv["distance_m"] is not None else "—")
                c.metric("Accord avec le Random Forest", "Oui" if dv["agrees_with_rf"] else "Non",
                         delta_color="off")
                if dv["history"]:
                    st.write(f"**LSTM sur 30 jours** : {dv['history']['label_fr']}, coupure estimée sous {dv['history']['ttf_days']:.0f} jours "
                             f"({'accord' if dv['history']['agrees_with_rf'] else 'avis différent du'} Random Forest).")
                show_fig(viz.proba_bar(dv["fault"]["proba"]))
                mt = json.loads((MODELS_DIR / "metrics.json").read_text()) if (MODELS_DIR / "metrics.json").exists() else {}
                if "cnn_tdr_acc" in mt:
                    st.dataframe(pd.DataFrame({
                        "Random Forest": [mt["copper_fault_acc"], mt["distance_mae_m"], mt["history_acc"], mt["ttf_mae_days_under90"]],
                        "Deep learning": [mt["cnn_tdr_acc"], mt["cnn_tdr_dist_mae_m"], mt["lstm_history_acc"], mt["lstm_ttf_mae_days_under90"]]},
                        index=["Précision défaut (TDR)", "Erreur distance (m)", "Précision tendance (30 j)", "Erreur jours avant coupure"]).round(3),
                        width="stretch")
                st.caption("Mêmes jeux de test pour les deux familles de modèles. Sur ces données simulées, les Random Forest sont au moins "
                           "aussi bons : le deep learning sert ici de second avis, et prendra de l'intérêt avec des courbes réelles plus variées.")


# ============================================================================= onglet fibre
def fiber_otdr_ui():
    left, right = st.columns([1, 3], gap="large")
    with left:
        st.subheader("Liaison FTTH / GPON")
        scen = st.selectbox("Scénario simulé", FIBER_CLASSES, format_func=LABELS_FR.get, key="fo_scen")
        length = st.slider("Longueur OLT → ONU (m)", 800, 15000, 6000, 100, key="fo_len")
        n = new_measure_button("fo")
        seed = int(st.number_input("Graine", 0, 10**6, 3, key="fo_seed")) + n
        rec, trace = simulate_fiber_link(scen, np.random.default_rng(seed), length_m=length)
    payload = {"otdr": trace.tolist(), "expected_length_m": rec["expected_length_m"],
               "fiber_att_db_km": rec["fiber_att_db_km"], "lambda_ratio": rec["lambda_ratio"],
               "rx_power_dbm": rec["rx_power_dbm"], "tx_power_dbm": rec["tx_power_dbm"]}
    try:
        res = call("/analyze/fiber", payload, inference.analyze_fiber)
    except Exception as e:
        st.error(f"Analyse impossible : {e}")
        return
    with right:
        c1, c2, c3, c4 = st.columns([1.4, 1.2, 1.2, 1.2])
        with c1:
            show_fig(viz.gauge(res["health_score"], "Santé de la liaison"))
        kpi_block(c2, "Diagnostic", res["fault"]["label_fr"], f"confiance {res['fault']['confidence']:.0%}")
        c3.metric("Localisation", f"{res['distance_m']:.0f} m" if res["distance_m"] is not None else "—")
        c4.metric("Puissance Rx ONU", f"{res['rx_power_dbm']:.1f} dBm", f"marge {res['margin_db']:.1f} dB", delta_color="off")
        st.caption(f"Vérité terrain simulée : **{LABELS_FR[scen]}**" +
                   (f" à {rec['fault_distance_m']:.0f} m" if not math.isnan(rec["fault_distance_m"]) else ""))
        show_alerts(res["alerts"])
        feedback_form("fiber", res)
        show_fig(viz.otdr_figure(trace, res["events"], res["end_dist_m"], res["distance_m"]))
        show_fig(viz.proba_bar(res["fault"]["proba"]))
        if res["events"]:
            st.dataframe(pd.DataFrame(res["events"]).rename(columns={"pos_m": "position (m)", "loss_db": "perte (dB)",
                                                                       "spike_db": "pic réflexion (dB)", "type": "type"}).round(2),
                         width="stretch")


def fiber_budget_ui():
    """Calculateur de budget optique (PON GPON B+/C+) avec comparaison à une mesure power-meter."""
    left, right = st.columns([1, 3], gap="large")
    with left:
        st.subheader("Liaison à calculer")
        pon = st.selectbox("Classe PON", list(optical.PON_CLASSES), key="fb_class")
        length = st.slider("Longueur de fibre (km)", 0.1, 40.0, 6.0, 0.1, key="fb_len")
        splitters = st.multiselect("Coupleurs (dans l'ordre)", list(optical.SPLITTER_LOSS_DB), default=["1:4", "1:8"], key="fb_spl")
        c1, c2 = st.columns(2)
        n_conn = c1.number_input("Connecteurs", 0, 20, 4, key="fb_nc")
        n_spl = c2.number_input("Soudures", 0, 60, 6, key="fb_ns")
        conn_kind = st.selectbox("Type de connecteur", list(optical.CONNECTOR_DB), key="fb_ck")
        splice_db = st.number_input("Perte par soudure (dB)", 0.0, 1.0, 0.10, 0.01, key="fb_sd")
        extra = st.number_input("Pertes supplémentaires (dB)", 0.0, 20.0, 0.0, 0.1, key="fb_ex",
                                help="Courbures connues, adaptateurs, vieillissement…")
        safety = st.slider("Marge de sécurité visée (dB)", 0.0, 6.0, 3.0, 0.5, key="fb_sm")
        measured = None
        if st.checkbox("J'ai une mesure power-meter (ONT)", key="fb_meas"):
            measured = float(st.number_input("Puissance mesurée (dBm)", -45.0, 5.0, -22.0, 0.1, key="fb_mv"))
    payload = {"length_km": float(length), "splitters": splitters, "n_connectors": int(n_conn), "n_splices": int(n_spl),
               "connector_db": optical.CONNECTOR_DB[conn_kind], "splice_db": float(splice_db), "extra_loss_db": float(extra),
               "pon_class": pon, "safety_margin_db": float(safety), "measured_rx_dbm": measured}
    try:
        res = call("/plan/optical-budget", payload, optical.budget_from_payload)
    except Exception as e:
        st.error(f"Calcul impossible : {e}")
        return
    ds, us = res["downstream"], res["upstream"]
    with right:
        a, b, c, d = st.columns(4)
        a.metric("Statut", res["status"])
        b.metric("Perte totale (descendant)", f"{ds['total_loss_db']:.1f} dB", f"budget classe {res['class_budget_db']:.0f} dB", delta_color="off")
        kpi_block(c, "Rx ONT (min … max)", f"{ds['rx_min_dbm']:.1f} … {ds['rx_max_dbm']:.1f} dBm", "selon la puissance d'émission OLT")
        d.metric("Portée max (marge visée)", f"{ds['max_reach_km']:.1f} km")
        (st.success if res["status"] == "OK" else st.error if res["status"] in ("HORS BUDGET", "SATURATION") else st.warning)(
            {"OK": "Liaison dans le budget avec la marge visée.",
             "MARGE FAIBLE": "Liaison fonctionnelle mais marge inférieure à la valeur visée.",
             "HORS BUDGET": "Puissance reçue sous la sensibilité du récepteur : liaison impossible en l'état.",
             "SATURATION": "Puissance reçue au-dessus du seuil de surcharge : ajouter un atténuateur."}[res["status"]])
        t1, t2 = st.tabs(["⬇️ Descendant (1490 nm)", "⬆️ Montant (1310 nm)"])
        for t, dr in ((t1, ds), (t2, us)):
            with t:
                x, y, z = st.columns(3)
                x.metric("Perte totale", f"{dr['total_loss_db']:.1f} dB")
                y.metric("Marge (pire cas)", f"{dr['margin_db']:.1f} dB")
                z.metric("Sensibilité / surcharge", f"{dr['sensitivity_dbm']:.0f} / {dr['overload_dbm']:.0f} dBm")
                show_fig(viz.budget_figure(dr["fixed_losses"]))
        if "measurement" in res:
            m = res["measurement"]
            show_alerts([{"level": m["level"], "title": f"Mesure : {m['excess_loss_db']:+.1f} dB par rapport au calcul "
                                                        f"(attendu {m['expected_rx_dbm']:.1f} dBm)", "detail": m["message"]}])
        st.caption("Valeurs typiques UIT-T G.652.D / G.984.2 (fibre 0,35 dB/km à 1310 nm, 0,25 dB/km à 1490 nm ; "
                   "coupleurs PLC 1:8 = 10,7 dB). À remplacer par vos fiches techniques.")


def fiber_history_ui():
    """Maintenance prédictive : évolution de la puissance reçue d'un ONT."""
    left, right = st.columns([1, 3], gap="large")
    with left:
        st.subheader("Historique de l'ONT")
        kind = st.selectbox("Tendance simulée", FIBER_HISTORY_KINDS, format_func=TREND_FR.get, index=2, key="fh_kind")
        n = new_measure_button("fh")
        seed = int(st.number_input("Graine", 0, 10**6, 11, key="fh_seed")) + n
        h = simulate_rx_history(kind, np.random.default_rng(seed))
        sens = float(st.number_input("Sensibilité de l'ONT (dBm)", -40.0, -20.0, FIBER_RX_FAIL_DBM, 0.5, key="fh_sens"))
    try:
        res = call("/analyze/fiber-history", {"rx_dbm": h["rx_dbm"].tolist(), "sensitivity_dbm": sens},
                   inference.analyze_fiber_history)
    except Exception as e:
        st.error(f"Analyse impossible : {e}")
        return
    with right:
        a, b, c, d = st.columns(4)
        kpi_block(a, "Tendance", res["label_fr"], f"confiance {res['confidence']:.0%}")
        b.metric("Jours avant seuil (IA)", f"{res['ttf_days_ml']:.0f} j")
        c.metric("Extrapolation linéaire", f"{res['ttf_days_linear']:.0f} j")
        d.metric("Marge actuelle", f"{res['margin_db']:.1f} dB", f"{res['rx_slope_db_per_day']:+.3f} dB/jour", delta_color="off")
        st.caption(f"Vérité terrain simulée : **{TREND_FR[kind]}**, {h['ttf_days']:.0f} jours avant le seuil "
                   "(365 = jamais ; au-delà de 90 jours, la ligne est classée « stable »).")
        show_alerts(res["alerts"])
        show_fig(viz.rx_history_figure(h["rx_dbm"], sens, res["ttf_days_ml"]))
        show_fig(viz.proba_bar(res["proba"], "Probabilités de la tendance"))


def tab_fiber():
    t1, t2, t3 = st.tabs(["🔍 Analyse OTDR", "🧮 Budget optique", "🔮 Maintenance prédictive"])
    with t1:
        fiber_otdr_ui()
    with t2:
        fiber_budget_ui()
    with t3:
        fiber_history_ui()


# ============================================================================= onglet radio (diagnostic)
def tab_radio():
    left, right = st.columns([1, 3], gap="large")
    with left:
        st.subheader("Mesure radio (scanner / terminal)")
        tech = st.selectbox("Technologie", list(RADIO_TECHS), format_func=RADIO_TECHS.get, index=1, key="ra_tech")
        scen = st.selectbox("Scénario simulé", RADIO_CLASSES, format_func=lambda k: LABELS_FR.get(k, "Cellule saine"), key="ra_scen")
        n = new_measure_button("ra")
        seed = int(st.number_input("Graine", 0, 10**6, 5, key="ra_seed")) + n
        kpi = simulate_radio_kpi(scen, tech, np.random.default_rng(seed))
    try:
        res = call("/analyze/radio", kpi, inference.analyze_radio)
    except Exception as e:
        st.error(f"Analyse impossible : {e}")
        return
    with right:
        a, b, c, d = st.columns(4)
        a.metric("RSRP", f"{kpi['rsrp_dbm']:.0f} dBm"); b.metric("SINR", f"{kpi['sinr_db']:.1f} dB")
        c.metric("BLER", f"{kpi['bler_pct']:.1f} %"); d.metric("Débit", f"{kpi['throughput_mbps']:.0f} Mbit/s")
        kpi_block(st, "Diagnostic", res["fault"]["label_fr"] if res["fault"]["label"] != "healthy" else f"Cellule {res['tech']} saine",
            f"confiance {res['fault']['confidence']:.0%}")
        show_alerts(res["alerts"])
        feedback_form("radio", res)
        show_fig(viz.proba_bar(res["fault"]["proba"]))
        with st.expander("Tous les KPI"):
            st.dataframe(pd.DataFrame({"valeur": {k: v for k, v in kpi.items() if k != "label"}}).round(2), width="stretch")
        st.caption("Détection de brouillage et de fausses antennes (IMSI-catcher) à partir de KPI : "
                   "voisinage isolé, changements de TAC, repli de RAT, chiffrement nul.")


# ============================================================================= onglet planification
def _map_base(lat: float, lon: float, zoom: int) -> folium.Map:
    m = folium.Map(location=[lat, lon], zoom_start=zoom, tiles=None, control_scale=True)
    folium.TileLayer(ESRI_IMAGERY, attr=ESRI_ATTR, name="Satellite (Esri World Imagery)", max_zoom=19).add_to(m)
    folium.TileLayer(ESRI_LABELS, attr=ESRI_ATTR, name="Noms de lieux", overlay=True, show=True).add_to(m)
    return m


def _legend(m: folium.Map, title: str, items: list[tuple[str, str]]):
    rows = "".join(f'<div><span style="background:{c};display:inline-block;width:12px;height:12px;'
                   f'margin-right:6px;border-radius:2px"></span>{t}</div>' for c, t in items)
    html = (f'<div style="position:fixed;bottom:28px;left:12px;z-index:9999;background:rgba(255,255,255,.92);'
            f'color:#222;padding:8px 10px;border-radius:6px;font:12px sans-serif"><b>{title}</b>{rows}</div>')
    m.get_root().html.add_child(folium.Element(html))


def _zoom_for(radius_km: float) -> int:
    return int(np.clip(round(15 - math.log2(max(radius_km, 0.1) / 0.5)), 9, 18))


def _handle_click(m: folium.Map, key: str):
    out = st_folium(m, height=600, use_container_width=True, returned_objects=["last_clicked"], key=key)
    lc = (out or {}).get("last_clicked")
    if lc and st.session_state.get("plan_move") and (lc["lat"], lc["lng"]) != st.session_state.get("last_click_seen"):
        st.session_state["last_click_seen"] = (lc["lat"], lc["lng"])
        st.session_state["pending_center"] = (round(lc["lat"], 5), round(lc["lng"], 5))
        st.rerun()


def tab_planning():
    # un clic sur la carte déplace le centre (traité AVANT l'instanciation des widgets)
    if "pending_center" in st.session_state:
        st.session_state["plan_lat"], st.session_state["plan_lon"] = st.session_state.pop("pending_center")
    st.session_state.setdefault("plan_lat", 34.425)
    st.session_state.setdefault("plan_lon", 8.784)

    domain = st.radio("Domaine", ["📡 Radio 4G / 5G / 6G", "💡 FTTH (fibre jusqu'au domicile)"], horizontal=True)
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.subheader("Zone d'étude")
        c1, c2 = st.columns(2)
        c1.number_input("Latitude", key="plan_lat", format="%.5f", step=0.002)
        c2.number_input("Longitude", key="plan_lon", format="%.5f", step=0.002)
        st.checkbox("🖱️ Un clic sur la carte déplace le centre", key="plan_move")
        radius_km = st.slider("Rayon de la zone (km)", 0.3, 15.0, 3.0, 0.1, key="plan_radius")
    lat, lon, radius_m = st.session_state["plan_lat"], st.session_state["plan_lon"], radius_km * 1000.0
    if domain.startswith("📡"):
        plan_radio(left, right, lat, lon, radius_m)
    else:
        plan_ftth_ui(left, right, lat, lon, radius_m)


def plan_radio(left, right, lat, lon, radius_m):
    with left:
        st.subheader("Paramètres radio")
        tech = st.selectbox("Technologie", list(P.TECHS), index=1)
        env = st.selectbox("Environnement", list(P.ENVIRONMENTS), index=1)
        cov = st.selectbox("Niveau de couverture visé", list(P.COVERAGE_TYPES), index=2)
        pop = st.number_input("Densité de population (hab/km²)", 1, 30000, int(P.ENVIRONMENTS[env]["pop"]), 50, key=f"pop_{env}")
        share = st.slider("Part de marché (%)", 5, 100, 40, 5) / 100
        gb = st.number_input("Trafic moyen (Go / mois / abonné)", 1, 500, int(P.TECHS[tech]["gb_month"]), 5, key=f"gb_{tech}")
        with st.expander("Hypothèses avancées"):
            bh = st.slider("Part du trafic en heure chargée (%)", 3.0, 15.0, 7.0, 0.5) / 100
            util = st.slider("Taux d'occupation cible des cellules (%)", 40, 90, 70, 5) / 100
            fade = st.slider("Marge d'évanouissement (dB)", 0.0, 15.0, 7.0, 0.5)

    area = math.pi * (radius_m / 1000.0) ** 2
    r = P.dimension_radio(tech, env, cov, area, pop, share, gb, bh, util, fade)
    sites_xy = P.hex_layout(radius_m, r["sites_total"])
    n_sites = len(sites_xy)
    margin = P.coverage_grid(sites_xy, tech, env, cov, radius_m, fade)
    covered = P.coverage_pct(margin)
    s_lat, s_lon = P.to_latlon(sites_xy[:, 0], sites_xy[:, 1], lat, lon)

    n_border = int((np.hypot(sites_xy[:, 0], sites_xy[:, 1]) > radius_m).sum())
    capex = n_sites * P.TECHS[tech]["capex_site"]
    with right:
        a, b, c, d, e = st.columns(5)
        a.metric("Rayon de cellule", f"{r['cell_radius_m']:.0f} m")
        b.metric("Sites (estimation)", f"{r['sites_total']}", f"limité par la {r['limiting_factor'].lower()}", delta_color="off")
        c.metric("Sites placés", f"{n_sites}", f"dont {n_border} de bordure", delta_color="off")
        d.metric("Couverture", f"{covered:.0f} %")
        e.metric("CAPEX indicatif", f"{capex / 1e6:.1f} M$")
        st.caption("« Estimation » = surface ÷ aire d'un site, ou capacité si elle domine. « Placés » = maillage hexagonal réel, "
                   "y compris les sites de bordure hors zone qui couvrent le pourtour du disque.")
        if covered < 90:
            st.warning("Couverture < 90 % avec ce placement hexagonal : zone trop petite par rapport à la maille, "
                       "ou paramètres de couverture trop exigeants.")
        if n_sites > 400:
            st.info(f"{n_sites} sites : seuls les 400 premiers sont dessinés (la carte de couverture reste complète).")

        m = _map_base(lat, lon, _zoom_for(radius_m / 1000))
        half = radius_m
        sw = P.to_latlon(-half, -half, lat, lon)
        ne = P.to_latlon(half, half, lat, lon)
        folium.raster_layers.ImageOverlay(image=P.margin_to_rgba(margin), bounds=[[float(sw[0]), float(sw[1])], [float(ne[0]), float(ne[1])]],
                                          opacity=1.0, name="Couverture estimée").add_to(m)
        folium.Circle([lat, lon], radius=radius_m, color="#ffffff", weight=2, fill=False, name="Zone d'étude").add_to(m)
        fg = folium.FeatureGroup(name="Sites")
        for i in range(min(n_sites, 400)):
            folium.CircleMarker([float(s_lat[i]), float(s_lon[i])], radius=5, color="#ffffff", weight=1.5, fill=True,
                                fill_color="#2a78d6", fill_opacity=1,
                                tooltip=f"Site {i + 1:03d} ({tech})").add_to(fg)
        fg.add_to(m)
        folium.LayerControl(collapsed=True).add_to(m)
        _legend(m, "Marge de liaison", [("#d62728", "< 0 dB : pas de service"), ("#ff8c00", "0–8 dB : bord de cellule"),
                                         ("#c8c828", "8–16 dB : correct"), ("#3caa50", "16–25 dB : bon"), ("#006e3c", "> 25 dB : excellent")])
        _handle_click(m, "map_radio")

        with st.expander("🤖 Optimisation IA du placement des sites"):
            st.caption("Recouvrement glouton + élagage sur une carte de demande à quartiers denses (synthétique), comparé au maillage hexagonal.")
            tgt = st.slider("Part de la demande à couvrir (%)", 90, 99, 98) / 100
            oc = ai_planning.compare_site_plans(radius_m, r["cell_radius_m"], r["sites_total"], tgt)
            q1, q2, q3 = st.columns(3)
            q1.metric("Sites optimisés", oc["sites_opt"], f"{-oc['saving_pct']:.0f} % vs hexagonal", delta_color="inverse")
            q2.metric("Sites hexagonaux", oc["sites_hex"])
            q3.metric("Demande couverte", f"{oc['cov_opt']:.1%}", f"hexagonal : {oc['cov_hex']:.1%}", delta_color="off")
            ox, oy = oc["opt_xy"][:, 0], oc["opt_xy"][:, 1]
            fig = go.Figure()
            fig.add_scatter(x=oc["pts"][:, 0], y=oc["pts"][:, 1], mode="markers", name="Demande", marker=dict(size=3, color="#9aa7a1", opacity=.5))
            fig.add_scatter(x=oc["hex_xy"][:, 0], y=oc["hex_xy"][:, 1], mode="markers", name="Hexagonal", marker=dict(size=9, color="#2a78d6", symbol="circle-open"))
            fig.add_scatter(x=ox, y=oy, mode="markers", name="Optimisé", marker=dict(size=10, color="#eb6834"))
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), yaxis_scaleanchor="x", xaxis_title="Est (m)", yaxis_title="Nord (m)")
            show_fig(fig)
            st.caption("Le maillage hexagonal couvre toute la zone de façon uniforme ; l'optimisation concentre les sites là où se trouve la demande. "
                       "Les zones peu denses restent donc moins bien couvertes : à valider sur de vraies données de trafic.")

        lb = r["link_budget"]
        with st.expander("Détail du dimensionnement"):
            st.dataframe(pd.DataFrame({"valeur": {
                "PIRE (dBm)": lb["eirp_dbm"], "Sensibilité terminal (dBm)": lb["sensitivity_dbm"],
                "Pénétration (dB)": lb["penetration_db"], "Marge d'évanouissement (dB)": lb["fade_margin_db"],
                "Affaiblissement max admissible (dB)": lb["max_path_loss_db"], "Rayon de cellule (m)": r["cell_radius_m"],
                "Surface par site (km²)": r["site_area_km2"], "Sites (couverture)": r["sites_coverage"],
                "Sites (capacité)": r["sites_capacity"], "Abonnés": r["subscribers"],
                "Débit par abonné en heure chargée (Mbit/s)": r["mbps_per_sub_busy_hour"],
                "Demande (Mbit/s)": r["demand_mbps"], "Capacité d'un site (Mbit/s)": r["site_capacity_mbps"]}}).round(3), width="stretch")
            st.caption(f"{P.TECHS[tech]['note']}. Modèle d'affaiblissement 3GPP TR 38.901 UMa + décalage de clutter. "
                       "Valeurs d'avant-projet à calibrer ; le 6G est prospectif.")
        df = pd.DataFrame({"site": [f"S{i + 1:03d}" for i in range(n_sites)], "latitude": s_lat, "longitude": s_lon, "technologie": tech})
        st.download_button("⬇️ Exporter les sites (CSV)", df.to_csv(index=False).encode(), "sites_radio.csv", "text/csv")


@st.cache_resource(show_spinner="Téléchargement du réseau routier OpenStreetMap…", ttl=3600)
def cached_osm_roads(lat: float, lon: float, radius_m: float):
    """(graphe, None) ou (None, message d'erreur) : l'échec est mis en cache pour ne pas bloquer chaque rafraîchissement."""
    try:
        return osm.RoadGraph.from_overpass(osm.fetch_roads(lat, lon, radius_m * 1.1), lat, lon), None
    except Exception as e:
        return None, str(e)[:160]


@st.cache_resource(ttl=3600)
def cached_grid_roads(radius_m: float) -> osm.RoadGraph:
    return osm.RoadGraph.synthetic_grid(radius_m * 1.1)


@st.cache_data(show_spinner="Téléchargement des bâtiments OpenStreetMap…", ttl=3600)
def cached_osm_buildings(lat: float, lon: float, radius_m: float):
    try:
        return osm.fetch_buildings(lat, lon, radius_m), None
    except Exception as e:
        return None, str(e)[:160]


ROUTE_MODES = ["Distance de Manhattan (sans routes)", "Routes OpenStreetMap (en ligne)", "Rues fictives (hors-ligne, démo)"]


def plan_ftth_ui(left, right, lat, lon, radius_m):
    with left:
        st.subheader("Paramètres FTTH")
        route_mode = st.selectbox("Tracé des câbles", ROUTE_MODES, index=1,
                                  help="Avec les routes, NAP/FDH/OLT sont posés sur la voirie et les câbles suivent "
                                       "les plus courts chemins ; les tronçons partagés ne comptent qu'une tranchée.")
        src = st.selectbox("Foyers à raccorder", ["Bâtiments OpenStreetMap (en ligne)", "Synthétiques (densité)",
                                                  "Fichier CSV (latitude, longitude)"])
        density = st.number_input("Densité de foyers (par km²)", 50, 6000, 600, 50, disabled=not src.startswith("Synth"))
        clustered = st.checkbox("Répartition en quartiers", True, disabled=not src.startswith("Synth"))
        up = st.file_uploader("CSV avec colonnes lat / lon", type="csv") if src.startswith("Fichier") else None
        c1, c2 = st.columns(2)
        nap_cap = c1.selectbox("Foyers par NAP", [4, 8, 16], index=1)
        fdh_nap = c2.selectbox("NAP par FDH", [2, 4, 8], index=1)
        pon = st.selectbox("Classe optique PON", ["Classe B+ (28 dB)", "Classe C+ (32 dB)"])
        route = st.slider("Facteur de cheminement (mode Manhattan)", 1.0, 1.8, 1.25, 0.05,
                          disabled=not route_mode.startswith("Distance"))
        if st.button("🔄 Recharger OpenStreetMap", help="Vide le cache des routes et des bâtiments (après une panne réseau)"):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()
        c3, c4 = st.columns(2)
        olt_e = c3.slider("OLT : décalage Est (m)", -int(radius_m), int(radius_m), 0, 50)
        olt_n = c4.slider("OLT : décalage Nord (m)", -int(radius_m), int(radius_m), 0, 50)
        with st.expander("Coûts unitaires équipements (TND, indicatifs)"):
            c_cable = st.number_input("Câble + pose (TND/m, mode Manhattan)", 1.0, 200.0, 18.0)
            c_trench = st.number_input("Génie civil forfaitaire (TND/m de tranchée)", 5.0, 600.0, 78.0,
                                       help="Utilisé seulement si le métré détaillé est désactivé.")
            c_fiber = st.number_input("Câble optique (TND/m de câble, mode routes)", 0.5, 40.0, 3.5)
            c_nap = st.number_input("NAP (TND)", 100.0, 10000.0, 1400.0)
            c_fdh = st.number_input("FDH (TND)", 500.0, 30000.0, 5500.0)
            c_onu = st.number_input("ONT (TND)", 50.0, 1500.0, 280.0)
        civil_on = st.checkbox("🚧 Génie civil détaillé (casses, tranchées…)", True,
                               disabled=route_mode.startswith("Distance"),
                               help="Nécessite le tracé sur routes (OSM ou rues fictives).")
        civil_params, civil_prices = {}, {}
        if civil_on and not route_mode.startswith("Distance"):
            with st.expander("Hypothèses de génie civil"):
                g1, g2 = st.columns(2)
                civil_params["rock_pct"] = g1.slider("Roche (% fouille)", 0, 100, 10, 5)
                civil_params["aerial_share_drop"] = g2.slider("Branchement aérien", 0.0, 1.0, 0.0, 0.05)
                civil_params["chamber_spacing_m"] = g1.number_input("Chambre tous les (m)", 50.0, 500.0, 150.0, 10.0)
                civil_params["crossings_per_km"] = g2.number_input("Traversées / km", 0.0, 10.0, 2.0, 0.5)
                civil_params["contingency_pct"] = g1.number_input("Aléas (%)", 0.0, 40.0, 10.0, 1.0)
                civil_params["studies_pct"] = g2.number_input("Études / contrôle (%)", 0.0, 30.0, 6.0, 1.0)
                civil_params["lateral_cap_m"] = g1.number_input("Raccord max / foyer (m)", 0.0, 60.0, 20.0, 1.0)
                st.caption("Trottoir vs chaussée (part des tronçons revêtus sous trottoir)")
                h1, h2, h3 = st.columns(3)
                civil_params["sidewalk_share"] = {"feeder": h1.slider("Transport", 0.0, 1.0, 0.3, 0.05),
                                                  "dist": h2.slider("Distrib.", 0.0, 1.0, 0.5, 0.05),
                                                  "drop": h3.slider("Branch.", 0.0, 1.0, 0.8, 0.05)}
            with st.expander("Bordereau de prix unitaires (TND) — modifiable"):
                st.caption("⚠ Prix indicatifs d'avant-projet : remplacez-les par votre BPU / vos marchés.")
                ed = st.data_editor(civil.default_price_table(), hide_index=True, width="stretch",
                                    disabled=["code", "désignation", "unité"], key="civil_bpu")
                civil_prices = dict(zip(ed["code"], ed["prix_tnd"].astype(float)))

    router = None
    if route_mode.startswith("Routes"):
        router, err = cached_osm_roads(round(lat, 5), round(lon, 5), float(radius_m))
        if err:
            st.warning(f"Routes OpenStreetMap indisponibles ({err}). Repli sur la distance de Manhattan.")
    elif route_mode.startswith("Rues"):
        router = cached_grid_roads(float(radius_m))

    if src.startswith("Bâtiments"):
        b, err = cached_osm_buildings(round(lat, 5), round(lon, 5), float(radius_m))
        if err or b is None or len(b) < 2:
            st.warning(f"Bâtiments OpenStreetMap indisponibles ({err or 'aucun bâtiment dans la zone'}). "
                       "Repli sur des foyers synthétiques.")
            homes = P.synthetic_homes(radius_m, density, clustered)
        else:
            x, y = P.to_xy(b[:, 0], b[:, 1], lat, lon)
            homes = np.column_stack([x, y])
            homes = homes[np.hypot(homes[:, 0], homes[:, 1]) <= radius_m]
    elif up is not None:
        d = pd.read_csv(up)
        d.columns = [c.lower().strip() for c in d.columns]
        la = next((c for c in d.columns if c.startswith("lat")), None)
        lo = next((c for c in d.columns if c.startswith(("lon", "lng"))), None)
        if not la or not lo:
            st.error("Colonnes latitude / longitude introuvables.")
            return
        x, y = P.to_xy(d[la].to_numpy(), d[lo].to_numpy(), lat, lon)
        homes = np.column_stack([x, y])[:5000]
    else:
        homes = P.synthetic_homes(radius_m, density, clustered)
    if len(homes) < 2:
        st.warning("Pas assez de foyers dans la zone.")
        return
    homes = homes[:5000]
    plan = P.plan_ftth(homes, (olt_e, olt_n), nap_cap, fdh_nap, route, olt_class_db=28.0 if "B+" in pon else 32.0,
                       router=router)
    civ = civil.estimate_civil(plan, router, civil_params, civil_prices) if (civil_on and plan["routed"]) else None
    cap = P.ftth_capex(plan, c_cable, c_nap, c_fdh, c_onu, c_trench_m=c_trench, c_fiber_m=c_fiber,
                       civil_total=civ["total_tnd"] if civ else None)

    with right:
        a, b, c, d_, e_ = st.columns(5)
        a.metric("Foyers", plan["n_homes"]); b.metric("NAP / FDH", f"{plan['n_nap']} / {plan['n_fdh']}")
        c.metric("Pire perte optique", f"{plan['worst_loss_db']:.1f} dB", f"marge {plan['margin_db']:+.1f} dB", delta_color="normal")
        d_.metric("Tranchée" if plan["routed"] else "Câble total",
                  f"{(plan['trench_m']['total'] if plan['routed'] else plan['drop_m'] + plan['dist_m'] + plan['feeder_m']) / 1000:.1f} km")
        e_.metric("CAPEX indicatif", f"{cap['TOTAL'] / 1e6:.2f} M TND", f"{cap['TOTAL'] / plan['n_homes']:.0f} TND/foyer", delta_color="off")
        if plan["margin_db"] < 0:
            st.error(f"{100 - plan['pct_within_budget']:.0f} % des ONT dépassent le budget optique : "
                     "rapprocher l'OLT, réduire le split ou passer en classe C+.")
        else:
            st.success(f"100 % des ONT dans le budget optique ({pon.split(' (')[0]}).")

        m = _map_base(lat, lon, _zoom_for(radius_m / 1000))
        h_lat, h_lon = P.to_latlon(homes[:, 0], homes[:, 1], lat, lon)
        n_lat, n_lon = P.to_latlon(plan["nap_xy"][:, 0], plan["nap_xy"][:, 1], lat, lon)
        f_lat, f_lon = P.to_latlon(plan["fdh_xy"][:, 0], plan["fdh_xy"][:, 1], lat, lon)
        o_lat, o_lon = P.to_latlon(plan["olt_xy"][0], plan["olt_xy"][1], lat, lon)
        folium.Circle([lat, lon], radius=radius_m, color="#ffffff", weight=2, fill=False).add_to(m)
        lines = folium.FeatureGroup(name="Câbles")
        if plan["routed"]:        # câbles dessinés le long des routes (tronçons réellement utilisés)
            for key, color, weight in (("drop", "#8fd0ff", 1.5), ("dist", "#eda100", 2.5), ("feeder", "#e87ba4", 3.5)):
                seg = router.edge_segments_latlon(plan["edges"][key], lat, lon)[:8000]
                if seg:
                    folium.PolyLine(seg, color=color, weight=weight, opacity=0.9).add_to(lines)
        else:
            if len(homes) <= 800:
                folium.PolyLine([[[float(h_lat[i]), float(h_lon[i])], [float(n_lat[g]), float(n_lon[g])]]
                                 for i, g in enumerate(plan["nap_labels"])], color="#8fd0ff", weight=1, opacity=0.7).add_to(lines)
            folium.PolyLine([[[float(n_lat[i]), float(n_lon[i])], [float(f_lat[g]), float(f_lon[g])]]
                             for i, g in enumerate(plan["fdh_labels"])], color="#eda100", weight=2).add_to(lines)
            folium.PolyLine([[[float(f_lat[i]), float(f_lon[i])], [float(o_lat), float(o_lon)]] for i in range(len(f_lat))],
                            color="#e87ba4", weight=2.5).add_to(lines)
        lines.add_to(m)
        if civ:        # tranchées : revêtues (casse à prévoir) vs terre
            all_e = plan["edges"]["drop"] | plan["edges"]["dist"] | plan["edges"]["feeder"]
            ids = np.fromiter(all_e, dtype=int)
            cas = folium.FeatureGroup(name="Casses (tranchées)", show=False)
            for flag, color in ((True, "#e34948"), (False, "#b08968")):
                sel = ids[router.paved[ids] == flag]
                seg = router.edge_segments_latlon(sel, lat, lon)[:8000]
                if seg:
                    folium.PolyLine(seg, color=color, weight=4, opacity=0.9).add_to(cas)
            cas.add_to(m)
        fh = folium.FeatureGroup(name="Foyers")
        for i in range(min(len(homes), 2000)):
            ok = plan["onu_loss_db"][i] <= (28.0 if "B+" in pon else 32.0)
            folium.CircleMarker([float(h_lat[i]), float(h_lon[i])], radius=2.5, weight=0, fill=True,
                                fill_color="#ffffff" if ok else "#e34948", fill_opacity=0.95).add_to(fh)
        fh.add_to(m)
        for i in range(len(n_lat)):
            folium.CircleMarker([float(n_lat[i]), float(n_lon[i])], radius=5, color="#fff", weight=1.2, fill=True,
                                fill_color="#eb6834", fill_opacity=1, tooltip=f"NAP {i + 1} ({int((plan['nap_labels'] == i).sum())} foyers)").add_to(m)
        for i in range(len(f_lat)):
            folium.CircleMarker([float(f_lat[i]), float(f_lon[i])], radius=8, color="#fff", weight=1.5, fill=True,
                                fill_color="#6250d6", fill_opacity=1, tooltip=f"FDH {i + 1}").add_to(m)
        folium.Marker([float(o_lat), float(o_lon)], icon=folium.Icon(color="red", icon="signal"), tooltip="OLT").add_to(m)
        folium.LayerControl(collapsed=True).add_to(m)
        _legend(m, "Réseau FTTH", [("#ffffff", "Foyer (dans le budget)"), ("#e34948", "Foyer hors budget"), ("#eb6834", "NAP 1:%d" % nap_cap),
                                   ("#6250d6", "FDH 1:%d" % fdh_nap), ("#e87ba4", "Câble de transport (OLT→FDH)"), ("#eda100", "Câble de distribution")])
        _handle_click(m, "map_ftth")

        with st.expander("Longueurs, coûts et architecture"):
            st.write(f"Transport : **{plan['feeder_m'] / 1000:.1f} km** · Distribution : **{plan['dist_m'] / 1000:.1f} km** · "
                     f"Raccordements : **{plan['drop_m'] / 1000:.1f} km** · Ports OLT : **{plan['olt_ports']}** "
                     f"(split total 1:{nap_cap * fdh_nap})")
            if plan["routed"]:
                tr = plan["trench_m"]
                st.write(f"Tranchée réelle : **{tr['total'] / 1000:.1f} km** (transport {tr['feeder'] / 1000:.1f} · "
                         f"distribution {tr['dist'] / 1000:.1f} · raccordements {tr['drop'] / 1000:.1f}, tronçons partagés comptés une fois) "
                         f"pour **{(plan['drop_m'] + plan['dist_m'] + plan['feeder_m']) / 1000:.0f} km** de câble posé.")
                st.caption("Routes et bâtiments © contributeurs OpenStreetMap (ODbL).")
            st.dataframe(pd.DataFrame({"TND": cap}).round(0), width="stretch")
        if civ:
            _civil_section(civ, plan)
            _ai_cost_panel(civ, plan, radius_m)
        nap_df = pd.DataFrame({"nap": [f"N{i + 1:03d}" for i in range(len(n_lat))], "latitude": n_lat, "longitude": n_lon,
                               "foyers": np.bincount(plan["nap_labels"]), "fdh": [f"F{g + 1:02d}" for g in plan["fdh_labels"]]})
        st.download_button("⬇️ Exporter les NAP (CSV)", nap_df.to_csv(index=False).encode(), "nap_ftth.csv", "text/csv")


def _ai_cost_panel(civ: dict, plan: dict, radius_m: float):
    """Modèle IA (Gradient Boosting) : coût génie civil par foyer prédit sans routage, comparé au métré calculé."""
    if not ai_planning.surrogate_ready():
        return
    pr = civ["params"]
    dens = plan["n_homes"] / (math.pi * (radius_m / 1000.0) ** 2)
    out = ai_planning.predict_civil(radius_m, dens, pr["rock_pct"], pr["aerial_share_drop"], pr["sidewalk_share"]["feeder"],
                                    pr["sidewalk_share"]["dist"], pr["sidewalk_share"]["drop"], 110, pr["chamber_spacing_m"])
    with st.expander("🤖 Prédiction IA du coût de génie civil"):
        a, b, c = st.columns(3)
        a.metric("Coût prédit / foyer", f"{out['cost_per_home_tnd']:,.0f} TND",
                 f"intervalle 80 % : {out['interval_tnd_per_home'][0]:,.0f} – {out['interval_tnd_per_home'][1]:,.0f}", delta_color="off")
        b.metric("Coût calculé / foyer (métré)", f"{civ['cost_per_home_tnd']:,.0f} TND",
                 f"{(civ['cost_per_home_tnd'] / out['cost_per_home_tnd'] - 1) * 100:+.0f} % vs IA", delta_color="off")
        m = out["metrics"]
        c.metric("Erreur du modèle (test)", f"{m['mape_pct']:.1f} %", f"R² {m['r2']:.2f}", delta_color="off")
        if out["out_of_domain"]:
            st.warning("Hors du domaine d'apprentissage (" + ", ".join(out["out_of_domain"]) + ") : la prédiction est peu fiable.")
        st.caption("Appris sur des plans simulés (rues fictives, rayon 500–900 m, 200–1 500 foyers/km²). Sur de vraies routes OSM, "
                   "l'écart avec le métré peut être plus grand : le métré calculé fait foi, l'IA donne un ordre de grandeur instantané pour comparer des scénarios.")
        imp = pd.Series(m["importance"]).clip(lower=0)
        st.bar_chart(imp[imp > 0], horizontal=True, height=200)


def _civil_section(civ: dict, plan: dict):
    st.markdown("#### 🚧 Génie civil et casses")
    a, b, c, d = st.columns(4)
    a.metric("Surface cassée", f"{civ['broken_m2'] / 1000:.1f} k m²",
             f"chaussée {civ['broken_road_m2'] / 1000:.1f} · trottoir {civ['broken_walk_m2'] / 1000:.1f}", delta_color="off")
    b.metric("Fouille", f"{civ['excavation_m3'] / 1000:.1f} k m³")
    c.metric("Chambres", f"{civ['chambers']:.0f}")
    d.metric("Génie civil", f"{civ['total_tnd'] / 1e6:.2f} M TND", f"{civ['cost_per_m_tnd']:.0f} TND/m", delta_color="off")
    lv = pd.DataFrame(civ["per_level"]).T
    lv.index = [civil.LEVEL_FR[i] for i in lv.index]
    lv = lv.rename(columns={"tranchée_m": "Tranchée (m)", "dont_revêtue_m": "Revêtue (m)", "dont_terre_m": "Terre (m)",
                            "aérien_m": "Aérien (m)", "casse_chaussée_m2": "Casse chaussée (m²)",
                            "casse_trottoir_m2": "Casse trottoir (m²)", "fouille_m3": "Fouille (m³)",
                            "chambres": "Chambres", "forage_m": "Forage (m)"})
    st.dataframe(lv.round(0), width="stretch")
    boq = civ["boq"].rename(columns={"code": "Code", "désignation": "Désignation", "unité": "Unité",
                                     "quantité": "Quantité", "prix_tnd": "PU (TND)", "montant_tnd": "Montant (TND)"})
    st.dataframe(boq.round(1), hide_index=True, width="stretch")
    st.caption(f"Sous-total {civ['subtotal_tnd']:,.0f} + aléas {civ['contingency_tnd']:,.0f} + études {civ['studies_tnd']:,.0f} "
               f"= **{civ['total_tnd']:,.0f} TND** · {civ['cost_per_home_tnd']:,.0f} TND/foyer. "
               "Prix indicatifs : à remplacer par votre BPU.")
    st.download_button("⬇️ Exporter le métré (CSV)", boq.to_csv(index=False).encode(), "metre_genie_civil.csv", "text/csv")


# ============================================================================= à propos
def tab_assistant():
    st.subheader("🤖 Assistant IA")
    ctx, recent = st.session_state.get("last", {}), st.session_state.get("recent", [])
    llm_ok = assistant.llm_available()
    c1, c2 = st.columns([3, 1])
    c1.caption("Posez vos questions en français ou en dialecte tunisien (lettres latines). L'assistant répond à partir des dernières analyses : "
               + (", ".join(assistant.DOMAINS[d] for d in ctx) or "aucune pour l'instant") + ".")
    use_llm = c2.toggle("Mode LLM", False, disabled=not llm_ok,
                        help="Nécessite ANTHROPIC_API_KEY et le paquet anthropic. Sans cela, l'assistant répond avec ses règles hors-ligne.")
    chat = st.session_state.setdefault("chat", [])
    lang = "tn" if chat and assistant.detect_language(chat[-1]["content"] if chat[-1]["role"] == "user" else "") == "tn" else "fr"
    cols = st.columns(4)
    pending = None
    for i, sug in enumerate(assistant.SUGGESTIONS[lang]):
        if cols[i % 4].button(sug, key=f"sug_{i}", width="stretch"):
            pending = sug
    for m in chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("meta"):
                st.caption(m["meta"])
    q = st.chat_input("Ex. : Où est le défaut ? / Chnowa ndir fil terrain ?") or pending
    if q:
        with st.chat_message("user"):
            st.markdown(q)
        out = assistant.ask(q, ctx, recent, chat, use_llm)
        meta = f"mode {out['mode']}" + (f" · {out['note']}" if out.get("note") else "")
        with st.chat_message("assistant"):
            st.markdown(out["answer"])
            st.caption(meta)
        chat += [{"role": "user", "content": q}, {"role": "assistant", "content": out["answer"], "meta": meta}]
    if chat and st.button("Effacer la conversation"):
        st.session_state["chat"] = []
        st.rerun()
    st.caption("Conseils génériques d'aide à la décision : ils ne remplacent ni les procédures de l'opérateur ni la vérification sur le terrain.")


def tab_learning():
    st.subheader("🔁 Auto-apprentissage")
    st.caption("Les retours des techniciens mesurent la précision réelle des modèles et permettent un ré-entraînement contrôlé : "
               "le nouveau modèle n'est déployé que s'il est meilleur sur des retours mis de côté, sans régression sur le jeu de test d'origine.")
    dom = st.radio("Domaine", list(feedback.CFG), format_func=lambda d: feedback.CFG[d]["label"], horizontal=True, key="lr_dom")
    lp = feedback.live_performance(dom)
    mt = json.loads((MODELS_DIR / "metrics.json").read_text()) if (MODELS_DIR / "metrics.json").exists() else {}
    test_acc = mt.get({"copper": "copper_fault_acc", "fiber": "fiber_acc", "radio": "radio_acc"}[dom])
    name = feedback.CFG[dom]["model"]
    try:
        ver = joblib.load(MODELS_DIR / f"{name}.joblib").get("version", "d'origine")
    except Exception:
        ver = "—"
    a, b, c, d = st.columns(4)
    a.metric("Retours enregistrés", lp["n"], f"dont {lp.get('n_simulated', 0)} simulés" if lp["n"] else "", delta_color="off")
    b.metric("Précision réelle (retours)", f"{lp['accuracy']:.0%}" if lp["n"] else "—",
             f"test d'origine : {test_acc:.0%}" if test_acc else "", delta_color="off")
    c.metric("Version du modèle", ver)
    d.metric("Erreur distance (retours)", f"{lp['distance_mae_m']:.0f} m" if lp.get("distance_mae_m") is not None else "—",
             f"sur {lp.get('n_distance', 0)} mesures" if lp["n"] else "", delta_color="off")
    if lp["n"]:
        if test_acc and lp["accuracy"] < test_acc - 0.05 and lp["n"] >= 20:
            st.warning(f"Dérive probable : précision réelle {lp['accuracy']:.0%} contre {test_acc:.0%} au test. Un ré-entraînement est conseillé.")
        l, r = st.columns(2)
        with l:
            fig = go.Figure(go.Scatter(y=lp["rolling"], mode="lines", line=dict(color="#2a78d6", width=2), name="précision glissante"))
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), title="Précision glissante (20 derniers retours)",
                              yaxis=dict(range=[0, 1.02], tickformat=".0%"), xaxis_title="retour n°")
            show_fig(fig)
        with r:
            cm = lp["confusion"]
            fig = go.Figure(go.Heatmap(z=cm, x=[LABELS_FR.get(c, c) for c in lp["classes"]], y=[LABELS_FR.get(c, c) for c in lp["classes"]],
                                       colorscale="Blues", showscale=False, text=cm, texttemplate="%{text}"))
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), title="Réel (lignes) vs prédit (colonnes)")
            show_fig(fig)
        if lp["top_confusions"]:
            st.write("**Erreurs les plus fréquentes :** " + " ; ".join(f"{LABELS_FR.get(x, x)} pris pour {LABELS_FR.get(y, y)} ({n}×)" for x, y, n in lp["top_confusions"]))
        if lp["conf_wrong"] is not None:
            st.caption(f"Confiance moyenne : {lp['conf_correct']:.0%} quand le modèle a raison, {lp['conf_wrong']:.0%} quand il se trompe.")
    else:
        st.info("Aucun retour pour ce domaine : confirmez un diagnostic dans l'onglet correspondant, ou générez des retours simulés ci-dessous.")

    st.markdown("#### Ré-entraîner")
    c1, c2, c3 = st.columns(3)
    if c1.button("Ré-entraîner avec les retours", type="primary", width="stretch"):
        with st.spinner("Entraînement du candidat et comparaison…"):
            try:
                rep = feedback.retrain(dom)
            except Exception as e:
                rep = {"status": "erreur", "message": str(e)}
        st.session_state["lr_report"] = rep
    if c2.button("Retour à la version précédente", width="stretch"):
        st.session_state["lr_report"] = feedback.rollback(dom)
    rep = st.session_state.get("lr_report")
    if rep:
        if rep.get("holdout_acc_before") is not None:
            (st.success if rep["promoted"] and rep["status"] == "déployé" else st.warning)(f"Résultat : {rep['status']} ({rep['reason']}).")
            st.dataframe(pd.DataFrame({"avant": [rep["holdout_acc_before"], rep["base_acc_before"]], "candidat": [rep["holdout_acc_after"], rep["base_acc_after"]]},
                                      index=[f"Retours mis de côté ({rep['n_holdout']})", "Jeu de test d'origine"]).round(3), width="stretch")
            if rep.get("distance"):
                dd = rep["distance"]
                if "holdout_mae_before" in dd:
                    st.caption(f"Distance : erreur {dd['holdout_mae_before']:.0f} → {dd['holdout_mae_after']:.0f} m sur les retours mis de côté "
                               f"({'déployé' if dd['promoted'] else 'non déployé'}).")
        else:
            st.info(rep.get("message") or rep.get("status"))

    with st.expander("Démonstration : retours terrain simulés avec dérive"):
        st.caption("Génère des retours dont les mesures dérivent par rapport à la simulation d'origine (impédance, bruit, amplitude TDR…), "
                   "pour voir la boucle fonctionner. Ils sont marqués « simulés » ; à supprimer avant tout usage réel.")
        n = st.slider("Nombre de retours", 40, 300, 120, 20, key="lr_n")
        drift = st.slider("Intensité de la dérive", 0.5, 4.0, 2.5, 0.5, key="lr_drift")
        e1, e2 = st.columns(2)
        if e1.button("Générer les retours simulés", width="stretch"):
            with st.spinner("Simulation…"):
                feedback.simulate_field_feedback(dom, n, drift, seed=int(time.time()) % 10**6)
            st.rerun()
        if e2.button("Supprimer les retours simulés", width="stretch"):
            feedback.clear_feedback(dom, only_simulated=True)
            st.rerun()
    reg = feedback.registry()
    if reg:
        with st.expander("Historique des ré-entraînements"):
            keep = ["ts", "domain", "status", "reason", "n_feedback", "holdout_acc_before", "holdout_acc_after", "base_acc_before", "base_acc_after"]
            st.dataframe(pd.DataFrame(reg).reindex(columns=keep).round(3).iloc[::-1], hide_index=True, width="stretch")


def tab_about():
    st.markdown("""
### Architecture
`data_generator.py` → jeux de données simulés · `train_model.py` → modèles (Random Forest, Isolation Forest) ·
`core/inference.py` → diagnostics · `api/main.py` → API FastAPI · `app.py` → ce tableau de bord.

| Mission | Modèle | Entrées |
|---|---|---|
| Défaut & distance (cuivre) | Random Forest (classif. + régression) | tension, impédance, SNR, atténuation, CRC, TDR |
| Maintenance prédictive | Random Forest (classe + jours avant coupure) | 30 jours de SNR / atténuation / CRC |
| Profil ADSL2+/VDSL2/G.fast | Random Forest étiqueté par un modèle physique | atténuation, SNR, CRC, impédance |
| Sécurité TSCM | Isolation Forest (lignes saines) + Random Forest | chutes de tension, bruit, impédance, échos TDR négatifs |
| Fibre : OTDR | Random Forest sur événements OTDR | pertes, réflexions, fin de fibre, Rx ONU |
| Fibre : budget optique | Calcul physique (B+/C+) | longueur, coupleurs, connecteurs, soudures |
| Fibre : maintenance | Random Forest (classe + jours avant seuil) | 30 jours de puissance Rx ONU |
| Radio 4G/5G/6G | Random Forest sur KPI | RSRP, SINR, BLER, voisinage, TAC, repli RAT |
| Planification | Bilan de liaison 3GPP 38.901, clustering sous contrainte, routage sur routes OSM | zone, densité, trafic, réseau routier |

### Limites à connaître
- Les modèles sont entraînés sur des **données simulées** : les scores affichés sont optimistes. Il faut ré-entraîner
  avec des mesures réelles (TDR, historiques DSLAM/OLT, KPI de supervision) avant tout usage opérationnel.
- Le 6G est **prospectif** (hypothèses d'études, pas de standard publié).
- L'imagerie satellite (Esri World Imagery) sert de fond visuel : vérifiez ses conditions d'utilisation pour un usage commercial.
  L'environnement (urbain/rural) est choisi manuellement ; la classification automatique du bâti est une évolution prévue.
""")
    if (MODELS_DIR / "metrics.json").exists():
        st.json(json.loads((MODELS_DIR / "metrics.json").read_text()), expanded=False)


# ============================================================================= main
def main():
    sidebar()
    if not inference.models_ready():
        st.error("Les modèles ne sont pas encore entraînés.")
        if st.button("⚙️ Générer les données et entraîner maintenant (≈ 30 s)", type="primary"):
            with st.spinner("Entraînement en cours..."):
                subprocess.run([sys.executable, str(ROOT / "train_model.py")], check=True, cwd=ROOT)
            st.rerun()
        return
    t = st.tabs(["🔌 Ligne cuivre", "💡 Fibre optique", "📡 Radio 4G/5G/6G", "🛰️ Planification (carte satellite)", "🤖 Assistant IA", "🔁 Auto-apprentissage", "ℹ️ À propos"])
    with t[0]:
        tab_copper()
    with t[1]:
        tab_fiber()
    with t[2]:
        tab_radio()
    with t[3]:
        tab_planning()
    with t[4]:
        tab_assistant()
    with t[5]:
        tab_learning()
    with t[6]:
        tab_about()


main()
