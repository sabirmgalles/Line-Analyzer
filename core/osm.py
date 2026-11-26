"""Réseau routier et bâtiments OpenStreetMap pour tracer le FTTH sur des routes réelles.

- fetch_roads / fetch_buildings : requêtes Overpass (internet requis, miroirs de secours)
- RoadGraph : graphe routier projeté en mètres, plus courts chemins (scipy), tranchées utilisées
- RoadGraph.synthetic_grid : réseau de rues fictif pour travailler hors-ligne et pour les tests

Données © contributeurs OpenStreetMap (licence ODbL) : à citer dans tout livrable.
"""
from __future__ import annotations

import numpy as np
import requests
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from core.planning import to_latlon, to_xy

OVERPASS_URLS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
HEADERS = {"User-Agent": "AI-Line-Analyzer/1.0 (planification FTTH)"}
ROAD_CLASSES = "primary|secondary|tertiary|unclassified|residential|living_street|service|road|track"
SKIP_BUILDINGS = "garage|garages|shed|roof|hut|carport|service|greenhouse|barn|farm_auxiliary|ruins"


def _overpass(query: str, timeout: int = 30) -> dict:
    last = None
    for url in OVERPASS_URLS:
        try:
            r = requests.post(url, data={"data": query}, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:           # miroir suivant
            last = e
    raise ConnectionError(f"Overpass injoignable : {last}")


def fetch_roads(lat: float, lon: float, radius_m: float) -> dict:
    q = (f'[out:json][timeout:50];(way["highway"~"^({ROAD_CLASSES})$"](around:{radius_m:.0f},{lat:.6f},{lon:.6f}););'
         "(._;>;);out body qt;")
    return _overpass(q)


def fetch_buildings(lat: float, lon: float, radius_m: float, limit: int = 5000, seed: int = 0) -> np.ndarray:
    """Centroïdes (lat, lon) des bâtiments OSM, sous-échantillonnés à `limit`."""
    q = (f'[out:json][timeout:50];(way["building"]["building"!~"^({SKIP_BUILDINGS})$"]'
         f"(around:{radius_m:.0f},{lat:.6f},{lon:.6f}););out center;")
    pts = parse_buildings(_overpass(q))
    if len(pts) > limit:
        pts = pts[np.random.default_rng(seed).choice(len(pts), limit, replace=False)]
    return pts


def parse_buildings(data: dict) -> np.ndarray:
    pts = [(e["center"]["lat"], e["center"]["lon"]) for e in data.get("elements", []) if "center" in e]
    return np.array(pts, dtype=float).reshape(-1, 2)


UNPAVED = {"unpaved", "dirt", "ground", "earth", "grass", "gravel", "fine_gravel", "sand", "mud", "compacted", "dirt/sand"}


def _way_is_paved(tags: dict) -> bool:
    """Tronçon revêtu ? Étiquette `surface` si présente, sinon déduit de la classe de route."""
    surf = str(tags.get("surface", "")).lower()
    if surf:
        return surf not in UNPAVED
    return tags.get("highway") not in ("track", "path") and tags.get("tracktype") is None


class RoadGraph:
    """Graphe routier non orienté (nœuds en mètres, arêtes pondérées par leur longueur)."""

    def __init__(self, nodes_xy: np.ndarray, edges: np.ndarray, paved: np.ndarray | None = None):
        # on ne garde que la plus grande composante connexe
        n = len(nodes_xy)
        w = np.linalg.norm(nodes_xy[edges[:, 0]] - nodes_xy[edges[:, 1]], axis=1)
        g = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
        _, lab = connected_components(g, directed=False)
        keep = lab == np.bincount(lab).argmax()
        remap = -np.ones(n, dtype=int)
        remap[keep] = np.arange(keep.sum())
        ok = keep[edges[:, 0]] & keep[edges[:, 1]]
        self.nodes = nodes_xy[keep]
        self.edges = remap[edges[ok]]
        self.lengths = w[ok]
        # revêtement par tronçon : True = route revêtue (casse de chaussée), False = piste / terre
        self.paved = np.ones(len(edges), dtype=bool)[ok] if paved is None else np.asarray(paved, dtype=bool)[ok]
        m = len(self.nodes)
        a = coo_matrix((self.lengths, (self.edges[:, 0], self.edges[:, 1])), shape=(m, m))
        self.csr = (a + a.T).tocsr()
        self.edge_id = {(min(u, v), max(u, v)): i for i, (u, v) in enumerate(self.edges)}
        self.tree = cKDTree(self.nodes)

    # ------------------------------------------------------------------ constructeurs
    @classmethod
    def from_overpass(cls, data: dict, lat0: float, lon0: float) -> "RoadGraph":
        els = data.get("elements", [])
        node_ll = {e["id"]: (e["lat"], e["lon"]) for e in els if e["type"] == "node"}
        idx, pts, edges, paved = {}, [], [], []
        for e in els:
            if e["type"] != "way" or "nodes" not in e:
                continue
            is_paved = _way_is_paved(e.get("tags", {}))
            for a, b in zip(e["nodes"][:-1], e["nodes"][1:]):
                if a in node_ll and b in node_ll:
                    for nid in (a, b):
                        if nid not in idx:
                            idx[nid] = len(pts)
                            pts.append(node_ll[nid])
                    edges.append((idx[a], idx[b]))
                    paved.append(is_paved)
        if not edges:
            raise ValueError("Aucune route trouvée dans la zone.")
        ll = np.array(pts)
        x, y = to_xy(ll[:, 0], ll[:, 1], lat0, lon0)
        return cls(np.column_stack([x, y]), np.array(edges, dtype=int), np.array(paved))

    @classmethod
    def synthetic_grid(cls, radius_m: float, block_m: float = 110.0, seed: int = 0) -> "RoadGraph":
        """Rues orthogonales irrégulières (quelques tronçons manquants) pour démonstration hors-ligne."""
        rng = np.random.default_rng(seed)
        k = int(radius_m / block_m) + 1
        gx, gy = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1))
        xy = np.column_stack([gx.ravel(), gy.ravel()]) * block_m + rng.normal(0, block_m * 0.05, ((2 * k + 1) ** 2, 2))
        inside = np.linalg.norm(xy, axis=1) <= radius_m * 1.05
        ids = -np.ones(len(xy), dtype=int)
        ids[inside] = np.arange(inside.sum())
        edges = []
        for i in range(2 * k + 1):
            for j in range(2 * k + 1):
                a = i * (2 * k + 1) + j
                for b in (a + 1 if j < 2 * k else None, a + (2 * k + 1) if i < 2 * k else None):
                    if b is not None and ids[a] >= 0 and ids[b] >= 0 and rng.random() > 0.08:
                        edges.append((ids[a], ids[b]))
        edges = np.array(edges, dtype=int)
        mid = xy[inside][edges].mean(axis=1)
        paved = np.linalg.norm(mid, axis=1) < 0.75 * radius_m      # périphérie = pistes en terre (démonstration)
        return cls(xy[inside], edges, paved)

    # ------------------------------------------------------------------ requêtes
    def snap(self, pts_xy: np.ndarray):
        """Plus proche nœud routier de chaque point : (indices, distance en m)."""
        d, i = self.tree.query(np.atleast_2d(pts_xy))
        return i, d

    def snap_points(self, pts_xy: np.ndarray) -> np.ndarray:
        return self.nodes[self.snap(pts_xy)[0]]

    def routes(self, src_xy: np.ndarray, tgt_xy: np.ndarray, assign: np.ndarray, want_edges: bool = False):
        """Longueur de câble de chaque cible vers sa source assignée en suivant les routes.

        Longueur = raccord (cible → route) + plus court chemin + raccord (route → source).
        Retourne (longueurs[m], ensemble d'identifiants d'arêtes utilisées).
        """
        src_xy, tgt_xy = np.atleast_2d(src_xy), np.atleast_2d(tgt_xy)
        s_nodes, s_snap = self.snap(src_xy)
        t_nodes, t_snap = self.snap(tgt_xy)
        used_src = np.unique(assign)
        row = {int(s): r for r, s in enumerate(used_src)}
        dist, pred = dijkstra(self.csr, directed=False, indices=s_nodes[used_src], return_predecessors=True)
        out = np.empty(len(tgt_xy))
        edges_used: set[int] = set()
        for j in range(len(tgt_xy)):
            s = int(assign[j])
            r = row[s]
            d = dist[r, t_nodes[j]]
            if not np.isfinite(d):           # pas de chemin : repli en ligne droite pondérée
                out[j] = 1.4 * np.abs(tgt_xy[j] - src_xy[s]).sum()
                continue
            out[j] = d + t_snap[j] + s_snap[s]
            if want_edges:
                node = t_nodes[j]
                while pred[r, node] >= 0:
                    p = pred[r, node]
                    edges_used.add(self.edge_id[(min(p, node), max(p, node))])
                    node = p
        return out, edges_used

    def edge_lengths_by_paving(self, edge_ids) -> tuple[float, float]:
        """(longueur revêtue, longueur en terre) des tronçons demandés."""
        ids = np.fromiter(edge_ids, dtype=int) if len(edge_ids) else np.zeros(0, dtype=int)
        L, pv = self.lengths[ids], self.paved[ids]
        return float(L[pv].sum()), float(L[~pv].sum())

    def edge_lengths_sum(self, edge_ids) -> float:
        return float(self.lengths[list(edge_ids)].sum()) if edge_ids else 0.0

    def edge_segments_latlon(self, edge_ids, lat0: float, lon0: float):
        """Segments [[lat,lon],[lat,lon]] des arêtes demandées (pour tracer sur la carte)."""
        ids = np.fromiter(edge_ids, dtype=int) if len(edge_ids) else np.zeros(0, dtype=int)
        if not len(ids):
            return []
        a, b = self.nodes[self.edges[ids, 0]], self.nodes[self.edges[ids, 1]]
        la, lo = to_latlon(a[:, 0], a[:, 1], lat0, lon0)
        lb, ob = to_latlon(b[:, 0], b[:, 1], lat0, lon0)
        return [[[float(p), float(q)], [float(r), float(s)]] for p, q, r, s in zip(la, lo, lb, ob)]
