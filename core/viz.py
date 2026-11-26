"""Graphiques Plotly (palette catégorielle fixe, un seul axe par graphique, marques fines)."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import FIBER_MAX_RANGE_M, LABELS_FR, SNR_FAIL_DB
from core.features import tdr_axis

BLUE, ORANGE, AQUA, VIOLET, RED, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948", "#8a8a85"
GRID = "rgba(128,128,128,0.18)"


def _base(fig: go.Figure, height: int = 320, **kw) -> go.Figure:
    kw.setdefault("hovermode", "x unified")
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=-0.2), **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def gauge(score: float, title: str = "Santé de la ligne") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number={"suffix": " /100"}, title={"text": title},
        gauge={"axis": {"range": [0, 100], "tickvals": [0, 25, 50, 75, 100], "tickfont": {"size": 11}},
               "bar": {"color": BLUE, "thickness": 0.28},
               "steps": [{"range": [0, 40], "color": "rgba(227,73,72,0.25)"},
                         {"range": [40, 75], "color": "rgba(237,161,0,0.25)"},
                         {"range": [75, 100], "color": "rgba(12,163,12,0.22)"}]}))
    fig.update_layout(height=230, margin=dict(l=40, r=40, t=50, b=0), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def tdr_figure(trace, distance_m=None, err_m=None, peak_m=None) -> go.Figure:
    x = tdr_axis(len(trace))
    fig = go.Figure(go.Scatter(x=x, y=trace, mode="lines", line=dict(color=BLUE, width=1.6), name="TDR",
                               hovertemplate="%{x:.0f} m : %{y:.3f}<extra></extra>"))
    fig.add_vrect(x0=0, x1=90, fillcolor=GRAY, opacity=0.15, line_width=0,
                  annotation_text="zone aveugle", annotation_position="top left")
    if distance_m is not None:
        if err_m:
            fig.add_vrect(x0=distance_m - err_m, x1=distance_m + err_m, fillcolor=ORANGE, opacity=0.15, line_width=0)
        fig.add_vline(x=distance_m, line=dict(color=ORANGE, width=2, dash="dash"),
                      annotation_text=f"défaut ≈ {distance_m:.0f} m", annotation_position="top")
    fig.update_xaxes(title="Distance (m)")
    fig.update_yaxes(title="Coefficient de réflexion")
    return _base(fig, 330, title="Réflectométrie TDR")


def history_figure(h: dict, ttf_days=None) -> go.Figure:
    """SNR (haut) et atténuation (bas) : deux graphiques, jamais deux axes Y."""
    days = np.arange(len(h["snr_db"])) - (len(h["snr_db"]) - 1)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("SNR (dB)", "Atténuation (dB)"))
    fig.add_trace(go.Scatter(x=days, y=h["snr_db"], mode="lines+markers", name="SNR", line=dict(color=BLUE, width=2),
                             marker=dict(size=5)), 1, 1)
    fig.add_hline(y=SNR_FAIL_DB, row=1, col=1, line=dict(color=RED, dash="dot", width=1.5),
                  annotation_text="seuil de coupure", annotation_position="bottom right")
    if ttf_days is not None and ttf_days < 120:
        slope = np.polyfit(days[-14:], np.asarray(h["snr_db"])[-14:], 1)[0]
        last = float(np.mean(h["snr_db"][-3:]))
        fut = np.arange(0, min(ttf_days * 1.05, 60) + 1)
        fig.add_trace(go.Scatter(x=fut, y=last + slope * fut, mode="lines", name="tendance",
                                 line=dict(color=ORANGE, dash="dash", width=1.6)), 1, 1)
    fig.add_trace(go.Scatter(x=days, y=h["attenuation_db"], mode="lines+markers", name="Atténuation",
                             line=dict(color=VIOLET, width=2), marker=dict(size=5)), 2, 1)
    fig.update_xaxes(title_text="Jours (0 = aujourd'hui)", row=2, col=1)
    return _base(fig, 430)


def proba_bar(proba: dict, title: str = "Probabilités du modèle") -> go.Figure:
    items = sorted(proba.items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(x=[v * 100 for _, v in items], y=[LABELS_FR.get(k, k) for k, _ in items],
                           orientation="h", marker_color=BLUE, hovertemplate="%{y} : %{x:.1f} %<extra></extra>"))
    fig.update_xaxes(title="%", range=[0, 100])
    return _base(fig, 260, title=title, hovermode="closest")


def rates_bar(rates: dict, recommended: str) -> go.Figure:
    names = list(rates)
    fig = go.Figure(go.Bar(x=names, y=[rates[n] for n in names],
                           marker_color=[ORANGE if n == recommended else BLUE for n in names],
                           text=[f"{rates[n]:.0f}" for n in names], textposition="outside",
                           hovertemplate="%{x} : %{y:.1f} Mbit/s<extra></extra>"))
    fig.update_yaxes(title="Mbit/s (descendant estimé)", type="log" if max(rates.values()) > 200 else "linear")
    return _base(fig, 280, title="Débit atteignable par profil (orange = recommandé)", hovermode="closest")


def rx_history_figure(rx, sens_dbm: float, ttf_days=None) -> go.Figure:
    """Puissance reçue ONU : historique, seuil de sensibilité et tendance."""
    days = np.arange(len(rx)) - (len(rx) - 1)
    fig = go.Figure(go.Scatter(x=days, y=rx, mode="lines+markers", name="Rx ONU", line=dict(color=BLUE, width=2),
                               marker=dict(size=5), hovertemplate="J%{x} : %{y:.2f} dBm<extra></extra>"))
    fig.add_hline(y=sens_dbm, line=dict(color=RED, dash="dot", width=1.5), annotation_text="sensibilité ONU",
                  annotation_position="bottom right")
    if ttf_days is not None and ttf_days < 120:
        slope = np.polyfit(days[-14:], np.asarray(rx)[-14:], 1)[0]
        fut = np.arange(0, min(ttf_days * 1.05, 60) + 1)
        fig.add_trace(go.Scatter(x=fut, y=float(np.mean(rx[-3:])) + slope * fut, mode="lines", name="tendance",
                                 line=dict(color=ORANGE, dash="dash", width=1.6)))
    fig.update_xaxes(title="Jours (0 = aujourd'hui)")
    fig.update_yaxes(title="Puissance reçue (dBm)")
    return _base(fig, 330, title="Puissance reçue par l'ONU")


def budget_figure(losses: dict, title: str = "Répartition des pertes (dB)") -> go.Figure:
    items = [(k, v) for k, v in losses.items() if v > 0]
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h", marker_color=BLUE,
                           text=[f"{v:.1f}" for _, v in items], textposition="outside",
                           hovertemplate="%{y} : %{x:.2f} dB<extra></extra>"))
    fig.update_xaxes(title="dB")
    fig.update_yaxes(autorange="reversed")
    return _base(fig, 250, title=title, hovermode="closest")


def otdr_figure(trace, events, end_dist_m, fault_pos_m=None) -> go.Figure:
    x = np.linspace(0, FIBER_MAX_RANGE_M, len(trace), endpoint=False) / 1000.0
    fig = go.Figure(go.Scatter(x=x, y=trace, mode="lines", line=dict(color=BLUE, width=1.3), name="OTDR",
                               hovertemplate="%{x:.2f} km : %{y:.1f} dB<extra></extra>"))
    names = {"splitter": "coupleur PON", "reflective": "événement réflectif", "loss": "perte"}
    for e in events:
        fig.add_vline(x=e["pos_m"] / 1000.0, line=dict(color=GRAY, width=1, dash="dot"))
        fig.add_annotation(x=e["pos_m"] / 1000.0, y=max(trace) if len(trace) else 0, showarrow=False, yshift=-8,
                           text=f"{names.get(e['type'], '')} {e['loss_db']:.1f} dB" if e["loss_db"] else names.get(e["type"], ""),
                           font=dict(size=10))
    fig.add_vline(x=end_dist_m / 1000.0, line=dict(color=VIOLET, width=1.5), annotation_text="fin visible",
                  annotation_position="bottom")
    if fault_pos_m is not None:
        fig.add_vline(x=fault_pos_m / 1000.0, line=dict(color=ORANGE, width=2.5, dash="dash"),
                      annotation_text=f"défaut ≈ {fault_pos_m:.0f} m", annotation_position="top")
    fig.update_xaxes(title="Distance (km)")
    fig.update_yaxes(title="Niveau relatif (dB)")
    return _base(fig, 360, title="Trace OTDR")
