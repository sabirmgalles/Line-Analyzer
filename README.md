# AI Line Analyzer

Analyse intelligente de lignes **cuivre (xDSL)**, **fibre optique (FTTH/GPON)** et **radio 4G/5G/6G**,
avec **dimensionnement et planification sur imagerie satellite**.

## Démarrage rapide

```bash
pip install -r requirements.txt
python data_generator.py        # 1. génère les jeux de données simulés  -> ./data
python train_model.py           # 2. entraîne les modèles (~30 s)         -> ./models
streamlit run app.py            # 3. tableau de bord                      -> http://localhost:8501
uvicorn api.main:app --reload   # 4. (option) API temps réel              -> http://localhost:8000/docs
pytest -q                       # 5. (option) tests de bout en bout
pip install torch               # 6. (option) deep learning, puis relancer train_model.py
```
Dans l'application, cochez « Utiliser l'API FastAPI » dans la barre latérale pour que les analyses passent par l'API.

## Arborescence

```
ai_line_analyzer/
├── config.py            constantes, classes, libellés français
├── data_generator.py    simulateurs cuivre / historique / fibre / radio  (+ CLI)
├── train_model.py       entraînement + métriques (RandomForest, IsolationForest)
├── app.py               interface Streamlit (5 onglets)
├── requirements.txt
├── api/
│   └── main.py          API FastAPI : /analyze/*, /simulate/*, /plan/*
├── core/
│   ├── physics.py       modèle de câble, débits ADSL2+/VDSL2/G.fast
│   ├── features.py      extraction TDR / OTDR / historique
│   ├── inference.py     diagnostics (santé, alertes, TSCM, modulation, maintenance)
│   ├── planning.py      bilan de liaison 3GPP 38.901, sites hexagonaux, plan FTTH, CAPEX
│   ├── deep.py          deep learning optionnel (PyTorch) : CNN 1D sur courbe TDR, LSTM sur historique
│   ├── ai_planning.py   IA de planification : coût génie civil (Gradient Boosting), optimisation des sites radio
│   ├── assistant.py     assistant IA (français + dialecte tunisien) : règles hors-ligne, mode LLM optionnel
│   ├── feedback.py      auto-apprentissage : retours terrain, suivi de précision, ré-entraînement contrôlé, retour arrière
│   ├── civil.py         génie civil FTTH : métré des casses, tranchées, chambres, BPU (TND)
│   ├── optical.py       budget optique PON (B+/C+), diagnostic power-meter
│   ├── osm.py           routes/bâtiments OpenStreetMap, graphe routier, plus courts chemins
│   └── viz.py           graphiques Plotly
├── data/                CSV / NPY / NPZ générés
├── models/              modèles .joblib + metrics.json
└── tests/test_smoke.py
```

## Les 4 missions (cuivre) + extensions

| Mission | Méthode |
|---|---|
| 1. Défaut + distance | Random Forest (6 classes) sur mesures électriques + features TDR ; Random Forest régresseur pour la distance, avec incertitude P90 par type de défaut |
| 2. Maintenance prédictive | Random Forest (stable / dégradation / imminente) + régresseur « jours avant coupure » sur 30 jours de SNR, atténuation, CRC |
| 3. Profil de modulation | Modèle physique (atténuation en √f, gap de Shannon) → débits ADSL2+/VDSL2/G.fast ; Random Forest appris sur ces règles |
| 4. TSCM | Isolation Forest entraîné **uniquement sur lignes saines** + probabilité « écoute » du Random Forest ; indices explicables (chute de tension, écho TDR négatif, bruit, impédance) |
| Fibre : analyse OTDR | Détection d'événements (pertes, réflexions, coupleur PON, fin de fibre) → Random Forest ; localisation, puissance Rx ONU |
| Fibre : budget optique | Calcul descendant 1490 nm / montant 1310 nm : fibre, coupleurs, connecteurs, soudures ; classes GPON B+/C+ ; portée max, saturation ; comparaison à une mesure power-meter |
| Fibre : maintenance prédictive | Random Forest + régresseur sur 30 jours de puissance Rx ONU : jours avant la sensibilité de l'ONT ; un saut brutal suivi d'un palier ne déclenche pas d'alerte |
| Radio 4G/5G/6G | Random Forest sur KPI (RSRP, SINR, BLER, voisinage, TAC, repli RAT) : interférence, trou de couverture, brouillage, IMSI-catcher |
| Génie civil FTTH | Tranchées réelles par niveau (transport / distribution / branchement), revêtement OSM (`surface`, `highway`) → casses chaussée / trottoir, fouille (part rocheuse), sable, remblai, conduits, chambres, forages dirigés, option aérienne, aléas ; bordereau de prix modifiable, export CSV |
| Deep learning (optionnel) | CNN 1D + canal de position sur la courbe TDR brute (type de défaut + distance, multi-tâche) ; LSTM sur 30 jours de SNR / atténuation / CRC. Même jeu de test que les Random Forest, affichés côte à côte : second avis, désaccord signalé en alerte |
| IA planification : coût | Gradient Boosting appris sur ~960 scénarios FTTH simulés : coût et casses par foyer en une milliseconde, intervalle à 80 % par calibration conforme, alerte hors domaine |
| IA planification : sites radio | Recouvrement glouton pondéré + élagage sur une carte de demande à quartiers denses, comparé au maillage hexagonal |
| Assistant IA | Chat en français ou en dialecte tunisien (lettres latines) : résumé, position, marche à suivre, urgence, sécurité, prévision, profil, budget, glossaire (TDR, OTDR, SNR…). Les chiffres viennent des dernières analyses. Mode règles hors-ligne par défaut ; mode LLM si `ANTHROPIC_API_KEY` est défini (`pip install anthropic`, modèle via `ASSISTANT_MODEL`). API : `POST /assistant` |
| Auto-apprentissage | Le technicien confirme le défaut réel (et la distance trouvée) → précision réelle, matrice de confusion, alerte de dérive → ré-entraînement : candidat = données d'origine + retours (poids ×5), déployé **seulement** s'il bat le modèle actuel sur des retours mis de côté sans perdre plus de 1 pt sur le test d'origine ; ancien modèle archivé, retour arrière en un clic. Cuivre, fibre, radio ; distance cuivre incluse. API : `POST /feedback`, `GET /feedback/stats/{domaine}`, `POST /feedback/retrain/{domaine}` |
| Planification | Bilan de liaison + couverture/capacité → nombre de sites ; placement hexagonal, carte de couverture ; FTTH : clustering sous contrainte de capacité (NAP 1:8 → FDH 1:4), budget optique, CAPEX |

## Carte satellite

L'onglet **Planification** utilise l'imagerie **Esri World Imagery** (réelle, mondiale) avec une surcouche de
couverture radio ou le plan FTTH. Fixez la latitude/longitude (ou cochez « un clic sur la carte déplace le centre »),
choisissez le rayon, la technologie et l'environnement. Export CSV des sites / NAP.

- La carte nécessite internet côté navigateur (tuiles + bibliothèque Leaflet).
- Vérifiez les conditions d'utilisation d'Esri pour un usage commercial ; pour la production, des alternatives
  existent (Mapbox, Maxar, Sentinel-2 via EOX / Copernicus).
- Le centre par défaut est Gafsa (34,425 ; 8,784).

## Limites importantes

- **Données simulées** : les scores (> 95 %) sont optimistes. Les classes simulées sont plus séparables que sur le terrain.
  Ré-entraînez avec des mesures réelles (TDR, historiques DSLAM/OLT, KPI de supervision) avant tout usage opérationnel.
- Le **6G est prospectif** (hypothèses d'études IMT-2030, pas de standard publié). Les paramètres d'équipement, de
  trafic et de coûts sont des ordres de grandeur d'avant-projet à calibrer.
- Le placement radio suppose un terrain homogène (modèle UMa + décalage de clutter) : ni relief, ni bâti réel,
  ni interférences inter-cellules (SINR) dans la carte de couverture.
- **Routes et bâtiments OpenStreetMap** : récupérés en ligne via l'API Overpass (internet requis ; données © contributeurs
  OpenStreetMap, ODbL). Sans connexion, l'application bascule sur la distance de Manhattan ; un mode « rues fictives »
  permet de tester hors-ligne. Cette récupération en ligne n'a pas pu être testée contre le service réel pendant le
  développement (seuls l'analyse des réponses et le routage ont été testés). Le tracé ne tient pas compte des sens interdits,
  du domaine privé, du relief ni de l'existant (fourreaux, poteaux) ; les bâtiments OSM peuvent inclure des annexes.

- **Deep learning** : sur les données simulées, CNN (97,8 %) et LSTM (91,8 %) font un peu moins bien que les Random Forest
  (99,8 % / 93,0 %) : ils servent de second avis. Ils nécessitent PyTorch (`pip install torch`, optionnel) ; sans lui, tout le reste fonctionne.
- **IA de planification** : le modèle de coût n'est valable que dans son domaine d'apprentissage (rues fictives, rayon 500–900 m,
  200–1 500 foyers/km²) et ne remplace pas le métré calculé. L'optimisation des sites suppose une carte de demande
  synthétique : le gain affiché (moins de sites) vient du fait que la demande est concentrée ; à refaire avec vos données de trafic.
- **Assistant** : en mode règles, il reconnaît des mots-clés et des questions fréquentes, pas toute formulation ; le dialecte est
  compris et écrit en lettres latines (variantes d'orthographe possibles). Le mode LLM n'a pas pu être testé contre le service réel
  pendant le développement (seul le repli vers les règles l'a été). Les conseils sont génériques et ne remplacent pas vos procédures.
- **Auto-apprentissage** : les retours sont stockés en local (SQLite `data/feedback.db`, sous forme de features). La démonstration utilise
  des retours SIMULÉS avec dérive : sur le cuivre, la précision passe d'environ 74 % à 98 % sur des données jamais vues ; sur la fibre
  et la radio la dérive simulée est trop faible et le ré-entraînement est refusé, ce qui est le comportement voulu. Avec de vrais retours, la qualité
  dépend de leur fiabilité (un retour faux dégrade le modèle) et de leur nombre (au moins 20, idéalement des centaines, répartis sur les classes).
  Le modèle déployé est ré-entraîné sur tous les retours, y compris ceux mis de côté pour la validation. Le deep learning (CNN, LSTM) n'est pas ré-entraîné par cette boucle.
- **Génie civil** : les prix par défaut (TND) sont des ordres de grandeur, pas un bordereau officiel : saisissez votre BPU
  (éditable dans l'application ou via `/plan/civil`). Le métré ne tient pas compte des réseaux enterrés existants, du
  relief, de la nappe, des autorisations ni de la réutilisation de fourreaux ; la répartition trottoir/chaussée est un
  paramètre (OSM ne la donne pas), et le revêtement OSM est parfois absent (déduit alors de la classe de route).

## Pistes d'évolution

1. Remplacer le TDR « features » par un **CNN 1D** sur la courbe brute, et le Random Forest historique par un **LSTM / TCN**.
2. Classification automatique du clutter (urbain/rural) depuis **Sentinel-2 / OpenStreetMap** ; réutilisation des fourreaux
   et poteaux existants dans le tracé FTTH ; import de traces OTDR réelles (.sor).
3. Cartes de couverture avec relief (SRTM) et SINR multi-cellules ; optimisation des sites (algorithmes génétiques).
4. Frontend **Flutter** pour les techniciens terrain en consommant l'API (`/analyze/*`), mode hors-ligne.
5. Boucle d'apprentissage : retour du technicien (défaut confirmé) → ré-entraînement.
