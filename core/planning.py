"""Dimensionnement et planification : radio 4G / 5G / 6G et FTTH.

- Bilan de liaison + modèle d'affaiblissement 3GPP TR 38.901 (UMa, simplifié)
- Dimensionnement couverture ET capacité -> nombre de sites
- Placement hexagonal des sites dans une zone circulaire (lat/lon)
- Grille de couverture (marge de liaison) pour surimpression sur imagerie satellite
- Plan FTTH : clustering à capacité contrainte (NAP 1:8 -> FDH 1:4), budget optique

Les paramètres sont des ordres de grandeur d'avant-projet ; à calibrer sur vos mesures,
vos équipements et vos tarifs avant tout engagement.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

R_EARTH = 6371008.8

# ----------------------------------------------------------------------------- paramètres radio
TECHS = {
    "4G LTE (1800 MHz)": dict(f_ghz=1.8, bw_mhz=20, ptx_dbm=46, gain_dbi=17, nf_db=7, sinr_edge_db=0.0,
                              se=2.0, gb_month=15, capex_site=60_000, note="2x2 MIMO, 20 MHz"),
    "5G NR (3,5 GHz)": dict(f_ghz=3.5, bw_mhz=100, ptx_dbm=53, gain_dbi=25, nf_db=7, sinr_edge_db=2.0,
                            se=8.0, gb_month=40, capex_site=90_000, note="Massive MIMO 64T64R, 100 MHz"),
    "5G mmWave (26 GHz)": dict(f_ghz=26.0, bw_mhz=400, ptx_dbm=40, gain_dbi=30, nf_db=9, sinr_edge_db=2.0,
                               se=4.5, gb_month=60, capex_site=45_000, note="Petites cellules, 400 MHz"),
    "6G FR3 (13 GHz) - prospectif": dict(f_ghz=13.0, bw_mhz=400, ptx_dbm=50, gain_dbi=30, nf_db=8,
                                         sinr_edge_db=2.0, se=12.0, gb_month=100, capex_site=100_000,
                                         note="Hypothèses d'études IMT-2030, non normalisé"),
}
ENVIRONMENTS = {   # décalage de clutter (dB) et densité de population typique (hab/km²)
    "Urbain dense": dict(offset_db=5.0, pop=8000),
    "Urbain": dict(offset_db=0.0, pop=3000),
    "Suburbain": dict(offset_db=-8.0, pop=800),
    "Rural": dict(offset_db=-16.0, pop=60),
}
COVERAGE_TYPES = {"Extérieur": 0.0, "Véhicule": 8.0, "Indoor léger": 15.0, "Indoor profond": 22.0}
H_BS, H_UT = 25.0, 1.5


def path_loss_db(d_m, f_ghz: float, offset_db: float = 0.0):
    """Affaiblissement UMa (38.901) : max(LOS, NLOS) + décalage de clutter."""
    d3 = np.sqrt(np.asarray(d_m, dtype=float) ** 2 + (H_BS - H_UT) ** 2)
    d3 = np.maximum(d3, 10.0)
    los = 28.0 + 22.0 * np.log10(d3) + 20.0 * np.log10(f_ghz)
    nlos = 13.54 + 39.08 * np.log10(d3) + 20.0 * np.log10(f_ghz) - 0.6 * (H_UT - 1.5)
    return np.maximum(los, nlos) + offset_db


def penetration_db(base_db: float, f_ghz: float) -> float:
    """La pénétration indoor croît avec la fréquence (≈ +2 dB par octave au-dessus de 3,5 GHz)."""
    return base_db + (2.0 * math.log2(f_ghz / 3.5) if f_ghz > 3.5 and base_db > 0 else 0.0)


def link_budget(tech: str, env: str, cov: str, fade_margin_db: float = 7.0) -> dict:
    t = TECHS[tech]
    eirp = t["ptx_dbm"] + t["gain_dbi"]
    sens = -174.0 + 10 * math.log10(t["bw_mhz"] * 1e6) + t["nf_db"] + t["sinr_edge_db"]
    pen = penetration_db(COVERAGE_TYPES[cov], t["f_ghz"])
    max_pl = eirp - sens - pen - fade_margin_db
    return {"eirp_dbm": eirp, "sensitivity_dbm": sens, "penetration_db": pen,
            "fade_margin_db": fade_margin_db, "max_path_loss_db": max_pl}


def max_range_m(tech: str, env: str, cov: str, fade_margin_db: float = 7.0) -> float:
    """Rayon de cellule max (dichotomie sur l'affaiblissement admissible)."""
    t, off = TECHS[tech], ENVIRONMENTS[env]["offset_db"]
    max_pl = link_budget(tech, env, cov, fade_margin_db)["max_path_loss_db"]
    lo, hi = 10.0, 60_000.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if path_loss_db(mid, t["f_ghz"], off) <= max_pl:
            lo = mid
        else:
            hi = mid
    return lo


def dimension_radio(tech: str, env: str, cov: str, area_km2: float, pop_density: float,
                    market_share: float = 0.4, gb_month: float | None = None,
                    busy_hour_share: float = 0.07, utilisation: float = 0.7,
                    fade_margin_db: float = 7.0) -> dict:
    """Nombre de sites = max(couverture, capacité)."""
    t = TECHS[tech]
    gb = t["gb_month"] if gb_month is None else gb_month
    R = max_range_m(tech, env, cov, fade_margin_db)
    site_area = 1.949 * (R / 1000.0) ** 2                          # site tri-sectoriel hexagonal
    n_cov = max(1, math.ceil(area_km2 / site_area))

    subs = pop_density * area_km2 * market_share
    mbps_per_sub = gb * 8000.0 * busy_hour_share / 30.0 / 3600.0   # Mbit/s en heure chargée
    demand = subs * mbps_per_sub
    cap_site = 3 * t["bw_mhz"] * t["se"] * utilisation
    n_cap = max(1, math.ceil(demand / cap_site))
    n = max(n_cov, n_cap)
    return {"cell_radius_m": R, "site_area_km2": site_area, "sites_coverage": n_cov, "sites_capacity": n_cap,
            "sites_total": n, "limiting_factor": "Couverture" if n_cov >= n_cap else "Capacité",
            "subscribers": subs, "mbps_per_sub_busy_hour": mbps_per_sub, "demand_mbps": demand,
            "site_capacity_mbps": cap_site, "capex_usd": n * t["capex_site"],
            "link_budget": link_budget(tech, env, cov, fade_margin_db)}


# ----------------------------------------------------------------------------- géométrie
def to_xy(lat, lon, lat0, lon0):
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    return (R_EARTH * math.cos(math.radians(lat0)) * np.radians(lon - lon0),
            R_EARTH * np.radians(lat - lat0))


def to_latlon(x, y, lat0, lon0):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return lat0 + np.degrees(y / R_EARTH), lon0 + np.degrees(x / (R_EARTH * math.cos(math.radians(lat0))))


def hex_layout(radius_m: float, n_sites: int) -> np.ndarray:
    """Sites sur réseau hexagonal (x, y en m) pour couvrir un disque de rayon `radius_m`.

    On garde les sites jusqu'à radius + ISD/√3 : ainsi chaque point du disque a son
    serveur le plus proche dans le plan (les sites de bordure sont donc inclus)."""
    area = math.pi * radius_m ** 2
    isd = math.sqrt(2.0 * area / (math.sqrt(3.0) * max(n_sites, 1)))
    pts = []
    ny, nx = int(radius_m / (isd * 0.866)) + 2, int(radius_m / isd) + 2
    for j in range(-ny, ny + 1):
        y = j * isd * 0.866
        off = isd / 2.0 if j % 2 else 0.0
        for i in range(-nx, nx + 1):
            x = i * isd + off
            if x * x + y * y <= (radius_m + 0.58 * isd) ** 2:
                pts.append((x, y))
    return np.array(pts) if pts else np.zeros((1, 2))


def coverage_grid(sites_xy: np.ndarray, tech: str, env: str, cov: str, radius_m: float,
                  fade_margin_db: float = 7.0, res: int = 140):
    """Marge de liaison (dB) sur une grille carrée ; NaN hors du disque d'étude.

    Tous les sites étant identiques, le meilleur serveur est le plus proche (cKDTree).
    """
    t, off = TECHS[tech], ENVIRONMENTS[env]["offset_db"]
    lb = link_budget(tech, env, cov, fade_margin_db)
    g = np.linspace(-radius_m, radius_m, res)
    gx, gy = np.meshgrid(g, g)
    inside = gx ** 2 + gy ** 2 <= radius_m ** 2
    d, _ = cKDTree(sites_xy).query(np.column_stack([gx.ravel(), gy.ravel()]))
    rx = lb["eirp_dbm"] - path_loss_db(d.reshape(gx.shape), t["f_ghz"], off) - lb["penetration_db"]
    margin = rx - lb["sensitivity_dbm"] - fade_margin_db
    margin[~inside] = np.nan
    return margin[::-1]          # origine en haut pour ImageOverlay (nord en haut)


def coverage_pct(margin: np.ndarray) -> float:
    """% de la zone d'étude (points du disque uniquement) avec une marge de liaison >= 0 dB."""
    inside = margin[np.isfinite(margin)]
    return float((inside >= 0).mean() * 100) if inside.size else 0.0


def margin_to_rgba(margin: np.ndarray) -> np.ndarray:
    """Colorie la marge de liaison : rouge (<0), orange, jaune-vert, vert, vert foncé."""
    rgba = np.zeros(margin.shape + (4,), dtype=np.uint8)
    bins = [(-np.inf, 0, (214, 39, 40, 150)), (0, 8, (255, 140, 0, 150)), (8, 16, (200, 200, 40, 150)),
            (16, 25, (60, 170, 80, 150)), (25, np.inf, (0, 110, 60, 150))]
    for lo, hi, col in bins:
        m = (margin >= lo) & (margin < hi)
        rgba[m] = col
    rgba[np.isnan(margin)] = (0, 0, 0, 0)
    return rgba


# ----------------------------------------------------------------------------- FTTH
def synthetic_homes(radius_m: float, density_km2: float, clustered: bool = True, seed: int = 0) -> np.ndarray:
    """Foyers synthétiques (x, y en m) ; mode 'clustered' = regroupements type quartiers."""
    rng = np.random.default_rng(seed)
    n = int(min(5000, max(10, density_km2 * math.pi * (radius_m / 1000.0) ** 2)))
    if clustered:
        k = max(3, n // 40)
        parents = rng.uniform(-1, 1, (k, 2)) * radius_m * 0.8
        pts = parents[rng.integers(0, k, n)] + rng.normal(0, radius_m * 0.07, (n, 2))
    else:
        pts = rng.uniform(-1, 1, (n * 2, 2)) * radius_m
    pts = pts[(pts ** 2).sum(1) <= radius_m ** 2]
    return pts[:n]


def capacitated_clusters(points: np.ndarray, cap: int, seed: int = 0):
    """K-means puis affectation gloutonne sous contrainte de capacité (<= cap points/cluster)."""
    n = len(points)
    k = max(1, math.ceil(n / cap))
    centers = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(points).cluster_centers_ if k > 1 else points.mean(0, keepdims=True)
    dist = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
    pref = np.argsort(dist, axis=1)
    if k > 1:
        regret = dist[np.arange(n), pref[:, 1]] - dist[np.arange(n), pref[:, 0]]
    else:
        regret = np.zeros(n)
    load, labels = np.zeros(k, dtype=int), -np.ones(n, dtype=int)
    for i in np.argsort(-regret):
        for c in pref[i]:
            if load[c] < cap:
                labels[i], load[c] = c, load[c] + 1
                break
    keep = [c for c in range(k) if (labels == c).any()]
    remap = {c: j for j, c in enumerate(keep)}
    labels = np.array([remap[c] for c in labels])
    new_centers = np.array([points[labels == j].mean(0) for j in range(len(keep))])
    return labels, new_centers


def plan_ftth(homes_xy: np.ndarray, olt_xy=(0.0, 0.0), nap_cap: int = 8, fdh_nap: int = 4,
              route_factor: float = 1.25, fiber_db_km: float = 0.35, olt_class_db: float = 28.0,
              router=None) -> dict:
    """Architecture OLT -> FDH (1:4) -> NAP (1:8) -> foyers, longueurs et budget optique.

    Sans `router` : distance de Manhattan x facteur de cheminement.
    Avec un `core.osm.RoadGraph` : NAP/FDH/OLT posés sur le réseau routier, câbles suivant
    les plus courts chemins, longueur de tranchée réelle (tronçons partagés comptés une fois).
    """
    olt = np.asarray(olt_xy, dtype=float)
    nap_lab, nap_xy = capacitated_clusters(homes_xy, nap_cap)
    fdh_lab, fdh_xy = capacitated_clusters(nap_xy, fdh_nap)

    edges = {}
    if router is None:
        manh = lambda a, b: np.abs(a - b).sum(axis=-1) * route_factor
        drop = manh(homes_xy, nap_xy[nap_lab])                  # foyer -> NAP
        dist = manh(nap_xy, fdh_xy[fdh_lab])                    # NAP -> FDH
        feeder = manh(fdh_xy, olt)                              # FDH -> OLT
        trench_m = None
    else:
        nap_xy, fdh_xy, olt = router.snap_points(nap_xy), router.snap_points(fdh_xy), router.snap_points(olt)[0]
        drop, edges["drop"] = router.routes(nap_xy, homes_xy, nap_lab, want_edges=True)
        dist, edges["dist"] = router.routes(fdh_xy, nap_xy, fdh_lab, want_edges=True)
        feeder, edges["feeder"] = router.routes(olt, fdh_xy, np.zeros(len(fdh_xy), dtype=int), want_edges=True)
        all_edges = edges["drop"] | edges["dist"] | edges["feeder"]
        trench_m = {"total": router.edge_lengths_sum(all_edges),
                    **{k: router.edge_lengths_sum(v) for k, v in edges.items()}}

    total_m = drop + dist[nap_lab] + feeder[fdh_lab[nap_lab]]
    split_db = 10 * math.log10(fdh_nap) * 1.0 + 10 * math.log10(nap_cap) * 1.0 + 1.2   # excès par étage
    n_splice = 4 + total_m / 500.0
    loss = total_m / 1000.0 * fiber_db_km + split_db + 5 * 0.3 + n_splice * 0.1
    return {"nap_xy": nap_xy, "fdh_xy": fdh_xy, "nap_labels": nap_lab, "fdh_labels": fdh_lab,
            "n_homes": len(homes_xy), "n_nap": len(nap_xy), "n_fdh": len(fdh_xy),
            "drop_m": float(drop.sum()), "dist_m": float(dist.sum()), "feeder_m": float(feeder.sum()),
            "onu_loss_db": loss, "worst_loss_db": float(loss.max()), "mean_loss_db": float(loss.mean()),
            "margin_db": float(olt_class_db - loss.max()),
            "pct_within_budget": float((loss <= olt_class_db).mean() * 100),
            "olt_ports": len(fdh_xy), "olt_xy": olt, "routed": router is not None,
            "trench_m": trench_m, "edges": edges,
            # longueur hors-voirie entre la route et chaque foyer (raccord latéral)
            "lateral_each_m": router.snap(homes_xy)[1] if router is not None else np.zeros(len(homes_xy))}


def ftth_capex(plan: dict, c_cable_m: float = 18.0, c_nap: float = 1400.0, c_fdh: float = 5500.0,
               c_onu: float = 280.0, c_port: float = 1100.0, c_drop_install: float = 370.0,
               c_trench_m: float = 78.0, c_fiber_m: float = 3.5, civil_total: float | None = None) -> dict:
    """CAPEX indicatif en TND (ordres de grandeur à remplacer par vos prix).

    Sans routage : câble + pose forfaitaires (TND/m de câble).
    Avec routage : câble + génie civil — soit le forfait c_trench_m sur la tranchée réelle, soit,
    si `civil_total` est fourni, le métré détaillé de `core.civil.estimate_civil`."""
    cable_m = plan["drop_m"] + plan["dist_m"] + plan["feeder_m"]
    if plan.get("trench_m"):
        items = {"Câbles": cable_m * c_fiber_m}
        if civil_total is not None:
            items["Génie civil (métré détaillé)"] = civil_total
        else:
            items["Tranchées / pose"] = plan["trench_m"]["total"] * c_trench_m
    else:
        items = {"Câbles & pose": cable_m * c_cable_m}
    items.update({"NAP (1:8)": plan["n_nap"] * c_nap, "FDH (1:4)": plan["n_fdh"] * c_fdh,
                  "ONT": plan["n_homes"] * c_onu, "Ports OLT": plan["olt_ports"] * c_port,
                  "Raccordement foyers": plan["n_homes"] * c_drop_install})
    items["TOTAL"] = sum(items.values())
    return {k: float(v) for k, v in items.items()}
