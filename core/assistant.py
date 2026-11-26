"""Assistant IA : le technicien pose une question en français ou en dialecte tunisien (lettres latines) et obtient
le diagnostic et des conseils tirés des DERNIERS RÉSULTATS d'analyse.

Deux modes :
- « règles » (par défaut, hors-ligne, sans clé) : détection d'intention (français + dialecte) + fiches terrain, les chiffres
  viennent uniquement des résultats d'analyse ;
- « LLM » (optionnel) : si la variable ANTHROPIC_API_KEY est définie et le paquet `anthropic` installé, les résultats sont
  transmis au modèle (variable ASSISTANT_MODEL) avec une consigne stricte de ne pas inventer de chiffres. En cas d'échec,
  retour automatique au mode règles.

Les conseils sont des procédures génériques d'aide à la décision : ils ne remplacent ni les procédures de l'opérateur
ni la vérification sur le terrain.
"""
from __future__ import annotations

import json
import os
import re

from config import LABELS_FR

DOMAINS = {"copper": "cuivre", "fiber": "fibre (OTDR)", "fiber_history": "fibre (maintenance)", "radio": "radio", "budget": "budget optique"}

DIALECT = {"chnowa", "chnia", "chniya", "kifech", "kifah", "kifeh", "3lech", "3leh", "aleh", "ouin", "win", "wakt", "waqt", "barcha", "behi", "nheb", "n7eb", "ya3ni",
           "tawa", "mouch", "mch", "besh", "bech", "3andi", "ma3andi", "mta3", "mtaa", "mt3", "tnajjem", "ndir", "nsalah", "nsalli", "lazem", "famma", "fama", "famech",
           "goulli", "9olli", "3tini", "aatini", "9adech", "9addech", "el", "ay", "ayy", "chkoun", "kifech", "yosla7", "ysalah", "taw", "nchouf", "nchoufou", "khoudmet", "7aja", "haja"}

# ----------------------------------------------------------------------------- intentions
INTENTS = {
    "where":    r"\b(où|ou se|distance|position|localis|ouin|win\b|wayn|9adech|9addech|combien de m|mètres|metres|kilom)",
    "actions":  r"(que faire|quoi faire|conseil|comment (r[ée]parer|corriger|intervenir)|proc[ée]dure|[ée]tapes|recommand|chnowa ndir|chnia ndir|kifech (nsalah|nsalli|n3aml|ndir)|nsalah|ndir)",
    "urgency":  r"(urgent|grave|dangere|priorit|critique|s[ée]rieux|mouch behi|khatir|khatar|sa3b)",
    "security": r"(écoute|ecoute|pirate|tscm|espion|intrusion|sécurité|securite|sniff|d[ée]rivation|tap\b|imsi|jamming|brouill)",
    "future":   r"(quand|combien de jours|panne future|pr[ée]dic|pr[ée]vision|d[ée]grad|tendance|maintenance|wakt|9addech men|9adech men|youm|ayem)",
    "modulation": r"(profil|vdsl|adsl|g\.?fast|d[ée]bit|modulation|vitesse|mbit)",
    "reliability": r"(fiable|s[ûu]r\b|confiance|certain|pr[ée]cis|thi9a|ta2akked|sa7i7|vrai|erreur)",
    "budget":   r"(budget|marge|bilan|perte totale|puissance re[çc]ue|rx\b|sensibilit)",
    "glossary": r"(c'est quoi|cest quoi|qu'est[- ]ce que|que signifie|d[ée]finition|explique|chnowa (houwa|hiya)|chnowa ya3ni|ya3ni chnowa|chnia (houwa|hiya)|ma3na)",
    "summary":  r"(r[ée]sum|diagnostic|probl[èe]me|[ée]tat|situation|qu'est[- ]ce qui|chnowa el|chnowa 3andou|chnowa fih|chnowa lmochkel|mochkel|alors|\bbilan\b)",
    "hello":    r"^\s*(bonjour|salut|salam|aslema|3aslema|hello|bonsoir|marhba|ahla)\b",
}
DOMAIN_WORDS = {
    "copper": r"(cuivre|xdsl|dsl\b|tdr|paire|adsl|vdsl|ligne t[ée]l|crc|snr)",
    "fiber": r"(fibre|otdr|onu|ont\b|gpon|olt|splitter|coupleur|soudure|[ée]pissure optique|connecteur)",
    "radio": r"(radio|4g|5g|6g|lte|rsrp|sinr|cellule|antenne|imsi|jamming|brouill|interf[ée]rence|couverture)",
}
GLOSSARY = {
    "tdr": ("TDR (réflectométrie temporelle) : on envoie une impulsion dans la paire et on mesure les échos. Un écho positif = circuit ouvert ou épissure, négatif = court-circuit ou dérivation. Le temps d'aller-retour donne la distance.",
            "TDR (réflectométrie) : nab3thou impulsion fil paire w n9isou el échos. Écho positif = câble coupé wella épissure, négatif = court-circuit wella dérivation. Wa9t el aller-retour y3atik el distance."),
    "otdr": ("OTDR : réflectomètre optique. Il envoie des impulsions dans la fibre et trace la puissance rétrodiffusée selon la distance : les marches sont des pertes (soudures, courbures), les pics des réflexions (connecteurs, cassure).",
             "OTDR : réflectomètre lel fibre. Yab3ath impulsions w yersem el puissance 3la 7asb el distance : el marches = pertes (soudures, courbures), el pics = réflexions (connecteurs, cassure)."),
    "snr": ("SNR : rapport signal sur bruit (dB). Plus il est élevé, plus le débit possible est grand. Sous ~8 dB la ligne DSL perd la synchronisation.",
            "SNR : rapport signal/bruit (dB). Kol ma yzid, el débit yzid. Taw7et men 8 dB, el ligne DSL tkhsar el synchronisation."),
    "crc": ("CRC : compteur de trames corrompues. Des CRC qui montent avec un SNR correct signalent souvent du bruit impulsif ou une épissure défectueuse.",
            "CRC : 3add el trames fasdin. CRC yetla3 m3a SNR behi = bruit impulsif wella épissure mouch behia."),
    "atténuation": ("Atténuation : perte de signal sur la ligne (dB), proportionnelle à la longueur et à la fréquence. Elle limite le débit.",
                    "Atténuation : el signal yn9es (dB) 3la 7asb el tool w el fréquence. Hiya li t7added el débit."),
    "splitter": ("Splitter / coupleur : composant passif qui divise le signal optique (1:4, 1:8…). Chaque division coûte ~3,5 dB par étage de 1:2.",
                 "Splitter : composant passif y9assem el signal optique (1:4, 1:8…). Kol étage 1:2 yekel ~3,5 dB."),
    "pon": ("PON/GPON : réseau optique passif point-multipoint : un OLT sert plusieurs ONU via des coupleurs, sans électronique active sur le trajet.",
            "GPON : réseau optique passif : OLT wa7ed yokhdem barcha ONU men tari9 splitters, mouch fih électronique fil trajet."),
    "onu": ("ONU/ONT : le terminal optique chez l'abonné. Sa sensibilité (≈ −27/−28 dBm en classe B+) fixe la puissance minimale reçue.",
            "ONU/ONT : el jihaz el optique 3and el abonné. Sensibilité mtaou (≈ −27/−28 dBm B+) hiya aqal puissance yeqbalha."),
    "olt": ("OLT : équipement de tête du réseau GPON, côté opérateur.", "OLT : équipement el ras mtaa GPON, jiha mtaa el opérateur."),
    "nap": ("NAP : point d'accès/boîtier de distribution proche des abonnés (ex. 1:8) ; le FDH est le point de brassage en amont (ex. 1:4).",
            "NAP : boîtier el distribution9rib men el abonnés (par ex. 1:8) ; el FDH howa no9ta el brassage 9bal (par ex. 1:4)."),
    "fdh": ("FDH : armoire/point de brassage fibre en amont des NAP.", "FDH : armoire fibre 9bal el NAP."),
    "rsrp": ("RSRP : puissance du signal de référence reçu en 4G/5G (dBm). Plus proche de 0, meilleur : −80 bon, −110 limite.",
             "RSRP : puissance el signal reçu fil 4G/5G (dBm). Kol ma 9rib men 0 ahsen : −80 behi, −110 limite."),
    "sinr": ("SINR : rapport signal sur interférences + bruit (dB) : détermine le débit radio.", "SINR : signal/interférences+bruit (dB) : howa li y7added el débit radio."),
    "imsi": ("IMSI-catcher : fausse antenne qui force les mobiles à s'y connecter pour les identifier ou les écouter. Signes : cellule inconnue, TAC inhabituel, repli forcé vers la 2G.",
             "IMSI-catcher : fausse antenne tkhallik el mobiles yetconnectaw biha. 3alamat : cellule mejhoula, TAC ghrib, repli lel 2G."),
    "bler": ("BLER : taux de blocs radio erronés ; élevé = liaison dégradée.", "BLER : nisba el blocs radio el ghaltin ; kbir = liaison mdeghrya."),
    "tscm": ("TSCM : contre-mesures de surveillance technique, c'est-à-dire détection d'écoutes et de dispositifs espions.", "TSCM : détection mtaa el écoutes w el appareils espions."),
}
PLAYBOOK = {
    "open_cut": (["Se rendre à la distance indiquée (en tenant compte de la marge d'erreur) et inspecter boîtiers, chambres et jonctions.", "Mesurer la continuité des deux fils pour confirmer.", "Réparer par jonction ou remplacer le tronçon coupé.", "Refaire un TDR après réparation pour valider."],
                 ["Imchi lel distance el mazkoura (m3a el marge d'erreur) w chouf el boîtiers w el chambres.", "Qis el continuité mtaa el fil zouz bech tta2akked.", "Sallah bel jonction wella badel el tronçon.", "3awed TDR ba3d el réparation."]),
    "short_circuit": (["Localiser à la distance indiquée, puis isoler les tronçons pour réduire la zone.", "Mesurer la résistance d'isolement ; chercher eau/humidité dans les chambres et boîtiers.", "Remplacer le câble ou la jonction abîmée et sécher/étanchéifier.", "Refaire TDR et mesure d'isolement."],
                      ["Localisi 3and el distance w 3azzel el tronçons bech tsaghghar el zone.", "Qis résistance d'isolement ; fattech 3la el ma/ routouba fil chambres.", "Badel el câble wella el jonction w 3azzel mli7.", "3awed TDR w isolement."]),
    "oxidation": (["Inspecter les connecteurs, répartiteur et boîtiers humides ; nettoyer ou remplacer les contacts oxydés.", "Remesurer SNR/atténuation après intervention.", "Surveiller la ligne 7 jours : si la tendance revient, remplacer le tronçon."],
                  ["Chouf el connecteurs, el répartiteur w el boîtiers el mblolin ; naddhef wella badel el contacts el moksidin.", "3awed qis SNR/atténuation ba3d el intervention.", "Raqeb el ligne 7 ayem : ken el tendance rja3it, badel el tronçon."]),
    "bad_splice": (["Ouvrir la jonction à la distance indiquée, vérifier serrage et sertissage.", "Refaire l'épissure (ou la jonction) proprement.", "Comparer le TDR avant/après : l'écho doit disparaître."],
                   ["Iftah el jonction 3and el distance, chouf serrage.", "3awed el épissure behi.", "9aren TDR 9bal/ba3d : lécho lazem yekhtafi."]),
    "wiretap": (["Ne pas toucher au dispositif suspect : photographier et documenter (position, heure, mesures).", "Inspecter physiquement le point suspect sans le déconnecter.", "Alerter le responsable sécurité de l'opérateur ; engager une inspection TSCM complète.", "Comparer avec les plans réseau pour vérifier qu'aucune dérivation n'est autorisée."],
                ["Ma tmassech el appareil el mach9ouk fih : sawwer w sajjel (blasa, wa9t, mesures).", "Fattech el blasa physiquement men 8ir ma t7ell el connexion.", "A3lem el responsable sécurité mtaa el opérateur ; lazem inspection TSCM kamla.", "9aren m3a el plans réseau ken famma dérivation morakhkhsa."]),
    "healthy": (["Aucune intervention nécessaire.", "Poursuivre la surveillance de routine (tendance SNR/CRC)."], ["Ma 3andek 7ata intervention.", "Kamel el surveillance el 3adiya (tendance SNR/CRC)."]),
    "fiber_break": (["Confirmer avec un OTDR depuis l'autre extrémité si possible.", "Aller à la position indiquée et inspecter le câble (travaux, rongeurs, chambre).", "Réparer par soudure par fusion ou remplacement de tronçon.", "Contrôler la perte de la soudure (< 0,1 dB visé) puis le budget de bout en bout."],
                    ["Ta2akked b OTDR men el taraf el okhra ken ymkin.", "Imchi lel position w chouf el câble (travaux, fi2ran, chambre).", "Sallah b soudure par fusion wella badel tronçon.", "Qis perte el soudure (< 0,1 dB) w ba3d el budget kamel."]),
    "macrobend": (["Aller à la position indiquée et repérer pli, pincement ou serrage excessif.", "Rétablir un rayon de courbure conforme (respecter le minimum du câble).", "Refaire l'OTDR : la perte ponctuelle doit disparaître."],
                  ["Imchi lel position w dawwer 3la pli wella pincement.", "Rajja3 rayon de courbure mnasb.", "3awed OTDR : el perte lazem tekhtafi."]),
    "dirty_connector": (["Inspecter le connecteur au microscope, le nettoyer (stylo ou cassette), le réinspecter.", "Remesurer : la réflexion et la perte doivent baisser.", "Si persistant, remplacer le jarretière ou le connecteur."],
                        ["Fattech el connecteur bel microscope w naddhefou, 3awed fattech.", "3awed qis : el réflexion w el perte lazem yn9sou.", "Ken ma ta7sanch, badel el jarretière wella el connecteur."]),
    "fiber_tap": (["Ne pas déconnecter : documenter et photographier la zone à la position indiquée.", "Rechercher un coupleur ou une courbure intentionnelle non référencée.", "Alerter le responsable sécurité et comparer aux plans."],
                  ["Ma t9at3ech : sawwer w sajjel el blasa 3and el position.", "Dawwer 3la coupleur wella courbure m9soda mouch msajjla.", "A3lem el responsable sécurité w 9aren m3a el plans."]),
    "interference": (["Analyser le spectre sur la bande concernée pour identifier la source.", "Vérifier PCI/fréquences voisines et réglages (tilt, puissance).", "Ajuster la planification ou coordonner avec le voisin concerné."],
                     ["7allel el spectre fil bande bech tla9a el source.", "Chouf PCI/fréquences el jar w el réglages (tilt, puissance).", "3addel el planification wella nassa9 m3a el jar."]),
    "coverage_hole": (["Contrôler tilt, azimut et puissance des cellules voisines.", "Mesurer sur site (drive test) pour confirmer l'étendue du trou.", "Si insuffisant, étudier un site ou un petit relais supplémentaire."],
                      ["Chouf tilt, azimut w puissance mtaa el cellules el jar.", "3amel drive test bech tta2akked men el étendue.", "Ken ma yekfich, edras site wella relais zeyed."]),
    "jamming": (["Localiser la source avec un analyseur de spectre et une antenne directive.", "Documenter (bande, niveau, horaire).", "Signaler aux autorités compétentes : le brouillage est illégal."],
                ["Dawwer 3la el source b analyseur de spectre w antenne directive.", "Sajjel (bande, niveau, wa9t).", "A3lem el autorités : el brouillage mamnou3."]),
    "rogue_cell": (["Relever PLMN/TAC/PCI de la cellule inconnue et les repliements 2G.", "Informer immédiatement l'équipe sécurité ; ne pas se connecter avec des équipements sensibles.", "Localiser par mesures multi-points."],
                   ["Sajjel PLMN/TAC/PCI mtaa el cellule el mejhoula w el repli lel 2G.", "A3lem ferra9 el sécurité fi nafs el wa9t ; ma tconnectich bel équipements el 7assassa.", "Localisi b mesures men barcha blayes."]),
}
URGENT = {"open_cut", "short_circuit", "fiber_break", "wiretap", "fiber_tap", "jamming", "rogue_cell"}


# ----------------------------------------------------------------------------- utilitaires
def detect_language(q: str) -> str:
    words = set(re.findall(r"[a-z0-9']+", q.lower()))
    return "tn" if len(words & DIALECT) - (1 if "el" in words and len(words & DIALECT) == 1 else 0) >= 1 else "fr"


def detect_domain(q: str, ctx: dict, recent: list[str]) -> str | None:
    for d, rx in DOMAIN_WORDS.items():
        if re.search(rx, q.lower()) and (d in ctx or (d == "fiber" and "fiber_history" in ctx)):
            return d
    for d in reversed(recent):
        if d in ctx:
            return d
    return next(iter(ctx), None)


def _fault(res: dict, dom: str) -> str | None:
    f = res.get("fault")
    if isinstance(f, dict):
        return f.get("label")
    return f if isinstance(f, str) else None


def _lab(label: str | None) -> str:
    return LABELS_FR.get(label or "", label or "—")


def compact_context(ctx: dict) -> dict:
    """Résumé JSON des résultats, sans courbes ni tableaux volumineux (pour le LLM et pour l'affichage)."""
    out = {}
    for d, r in ctx.items():
        c = {k: v for k, v in r.items() if k not in ("features", "events")}
        if "fault" in c and isinstance(c["fault"], dict):
            c["fault"] = {k: v for k, v in c["fault"].items() if k != "proba"}
        if "maintenance" in c and isinstance(c["maintenance"], dict):
            c["maintenance"] = {k: v for k, v in c["maintenance"].items() if k != "proba"}
        out[d] = c
    return out


# ----------------------------------------------------------------------------- réponses par règles
def _pick(lang: str, fr: str, tn: str) -> str:
    return tn if lang == "tn" else fr


def _summary(dom: str, r: dict, lang: str) -> str:
    L = lambda fr, tn: _pick(lang, fr, tn)
    if dom == "copper":
        f, d = r["fault"], r.get("distance_m")
        s = L(f"**Diagnostic cuivre : {f['label_fr']}** (confiance {f['confidence']:.0%}), santé de la ligne {r['health_score']:.0f}/100.",
              f"**Diagnostic cuivre : {f['label_fr']}** (thi9a {f['confidence']:.0%}), santé el ligne {r['health_score']:.0f}/100.")
        if d is not None:
            s += L(f" Défaut estimé à **{d:.0f} m** (± {r['distance_error_p90_m']:.0f} m).", f" El défaut 3and **{d:.0f} m** (± {r['distance_error_p90_m']:.0f} m).")
        s += L(f" Risque TSCM : {r['tscm']['risk_level']}. Profil conseillé : {r['modulation']['recommended']}.",
               f" Risque TSCM : {r['tscm']['risk_level']}. Profil li ynas7a bih : {r['modulation']['recommended']}.")
        dv = r.get("deep")
        if dv and not dv["agrees_with_rf"]:
            s += L(f" ⚠️ Le CNN n'est pas d'accord ({dv['fault']['label_fr']}) : refaire la mesure.", f" ⚠️ El CNN mouch m3a el Random Forest ({dv['fault']['label_fr']}) : 3awed el mesure.")
        return s
    if dom == "fiber":
        f, d = r["fault"], r.get("distance_m")
        s = L(f"**Diagnostic fibre : {f['label_fr']}** (confiance {f['confidence']:.0%}), santé {r['health_score']:.0f}/100, puissance ONU {r['rx_power_dbm']:.1f} dBm, marge {r['margin_db']:.1f} dB.",
              f"**Diagnostic fibre : {f['label_fr']}** (thi9a {f['confidence']:.0%}), santé {r['health_score']:.0f}/100, puissance ONU {r['rx_power_dbm']:.1f} dBm, marge {r['margin_db']:.1f} dB.")
        if d is not None:
            s += L(f" Événement à **{d:.0f} m**.", f" El événement 3and **{d:.0f} m**.")
        return s
    if dom == "fiber_history":
        return L(f"**Tendance fibre : {r['label_fr']}**, coupure estimée dans ~{r['ttf_days_ml']:.0f} jours (marge actuelle {r['margin_db']:.1f} dB, pente {r['rx_slope_db_per_day']:+.2f} dB/jour).",
                 f"**Tendance fibre : {r['label_fr']}**, el coupure 9rib ~{r['ttf_days_ml']:.0f} ayem (marge tawa {r['margin_db']:.1f} dB, pente {r['rx_slope_db_per_day']:+.2f} dB/jour).")
    if dom == "radio":
        f = r["fault"]
        return L(f"**Diagnostic radio ({r.get('tech', '')}) : {f['label_fr']}** (confiance {f['confidence']:.0%}).",
                 f"**Diagnostic radio ({r.get('tech', '')}) : {f['label_fr']}** (thi9a {f['confidence']:.0%}).")
    if dom == "budget":
        ds = r["downstream"]
        return L(f"**Budget optique {r['pon_class']}** : marge descendante {ds['margin_db']:.1f} dB, statut **{r['status']}**.",
                 f"**Budget optique {r['pon_class']}** : marge descendante {ds['margin_db']:.1f} dB, statut **{r['status']}**.")
    return ""


def _where(dom: str, r: dict, lang: str) -> str:
    L = lambda fr, tn: _pick(lang, fr, tn)
    if dom == "copper":
        d, ts = r.get("distance_m"), r["tscm"]
        if d is None:
            return L("Aucun défaut à localiser : la ligne est saine.", "Ma famech défaut bech nlocalisiweh : el ligne saine.")
        s = L(f"Le défaut est à environ **{d:.0f} m** de l'extrémité de mesure, avec une incertitude de ± {r['distance_error_p90_m']:.0f} m (P90). Cherchez dans cette fenêtre : {max(d - r['distance_error_p90_m'], 0):.0f} à {d + r['distance_error_p90_m']:.0f} m.",
              f"El défaut 9rib men **{d:.0f} m** men el taraf li 9isit menou, m3a incertitude ± {r['distance_error_p90_m']:.0f} m (P90). Dawwer fil fenêtre : {max(d - r['distance_error_p90_m'], 0):.0f} lel {d + r['distance_error_p90_m']:.0f} m.")
        if ts.get("suspected_position_m"):
            s += L(f" Position suspecte TSCM : ~{ts['suspected_position_m']:.0f} m.", f" Position mach9ouk fiha (TSCM) : ~{ts['suspected_position_m']:.0f} m.")
        return s
    if dom == "fiber":
        d = r.get("distance_m")
        return (L(f"L'événement est à environ **{d:.0f} m** de l'OLT/du point de mesure.", f"El événement 9rib men **{d:.0f} m** men el OLT/el point de mesure.") if d is not None
                else L("Aucun événement à localiser : la liaison est saine.", "Ma famech événement : el liaison saine."))
    if dom == "radio":
        return L("L'analyse radio donne un type de problème, pas une position : localisez par mesures multi-points (drive test).", "El analyse radio t3ati nou3 el mochkel mouch el position : localisi b drive test.")
    return L("Cette analyse ne fournit pas de position.", "Hedha el analyse ma y3atich position.")


def _actions(dom: str, r: dict, lang: str) -> str:
    label = _fault(r, dom)
    pb = PLAYBOOK.get(label or "")
    if not pb:
        return _pick(lang, "Pas de procédure spécifique pour ce résultat ; reprenez la mesure ou consultez le manuel de l'équipement.",
                     "Ma famech procédure mkhossoussa lel résultat hedha ; 3awed el mesure wella chouf el manuel.")
    steps = pb[1] if lang == "tn" else pb[0]
    head = _pick(lang, f"**Marche à suivre ({_lab(label)})** :", f"**El étapes ({_lab(label)})** :")
    return head + "\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))


def _urgency(dom: str, r: dict, lang: str) -> str:
    L = lambda fr, tn: _pick(lang, fr, tn)
    label = _fault(r, dom)
    if dom == "fiber_history":
        lvl = r["label"]
        return L(f"Tendance « {r['label_fr']} » : " + ("**urgent**, intervenir sous quelques jours." if lvl == "imminent" else "à planifier prochainement." if lvl == "degrading" else "pas d'urgence."),
                 f"Tendance « {r['label_fr']} » : " + ("**urgent**, interviena fi ayem 9lail." if lvl == "imminent" else "lazem tplanifiha 9rib." if lvl == "degrading" else "ma famech isti3jel."))
    if dom == "budget":
        return L(f"Statut du budget : **{r['status']}**.", f"Statut el budget : **{r['status']}**.")
    if label in URGENT:
        return L(f"**Priorité haute** : « {_lab(label)} » coupe le service ou relève de la sécurité.", f"**Priorité 3alia** : « {_lab(label)} » y9at3a el service wella mta3 sécurité.")
    if label == "healthy":
        return L("Aucune urgence : tout est normal.", "Ma famech isti3jel : kol chay normal.")
    return L(f"**Priorité moyenne** : « {_lab(label)} » dégrade la qualité sans couper le service ; à planifier.", f"**Priorité moyenne** : « {_lab(label)} » tn9es men el qualité ama el service mazel ; plannifih.")


def _security(dom: str, r: dict, lang: str) -> str:
    L = lambda fr, tn: _pick(lang, fr, tn)
    if dom == "copper":
        ts = r["tscm"]
        s = L(f"Risque TSCM **{ts['risk_level']}** (score {ts['risk_score']:.0%}, probabilité d'écoute {ts['wiretap_proba']:.0%}).",
              f"Risque TSCM **{ts['risk_level']}** (score {ts['risk_score']:.0%}, probabilité écoute {ts['wiretap_proba']:.0%}).")
        if ts["indicators"]:
            s += L(" Indices : ", " El indices : ") + " ; ".join(ts["indicators"]) + "."
        else:
            s += L(" Aucun indice de branchement pirate.", " Ma famech indice mta3 branchement pirate.")
        return s
    if dom == "radio":
        f = r["fault"]
        return L(f"Diagnostic radio : {f['label_fr']}. " + ("Suspicion d'IMSI-catcher ou de brouillage : alertez la sécurité." if f["label"] in ("rogue_cell", "jamming") else "Rien de suspect côté sécurité."),
                 f"Diagnostic radio : {f['label_fr']}. " + ("Fama chak fi IMSI-catcher wella brouillage : a3lem el sécurité." if f["label"] in ("rogue_cell", "jamming") else "Ma famech 7aja mach9ouk fiha."))
    if dom == "fiber":
        return L("Prélèvement optique suspecté : " + ("**oui**, à la position indiquée." if _fault(r, dom) == "fiber_tap" else "non détecté."),
                 "Prélèvement optique : " + ("**eh**, 3and el position." if _fault(r, dom) == "fiber_tap" else "ma tla9ach."))
    return L("Pas d'analyse de sécurité disponible pour ce résultat.", "Ma famech analyse sécurité lel résultat hedha.")


def _future(dom: str, r: dict, lang: str) -> str:
    L = lambda fr, tn: _pick(lang, fr, tn)
    if dom == "fiber_history":
        return _summary("fiber_history", r, lang)
    if dom == "copper":
        m = r.get("maintenance")
        if not m:
            return L("Aucun historique fourni : ajoutez 30 jours de SNR/atténuation/CRC pour une prédiction.", "Ma famech historique : zid 30 youm men SNR/atténuation/CRC bech ntanabba.")
        return L(f"Tendance **{m['label_fr']}**, coupure estimée dans ~{m['ttf_days_ml']:.0f} jours (extrapolation linéaire : {m['ttf_days_linear']:.0f} j). Pente SNR {m['snr_slope_db_per_day']:+.2f} dB/jour.",
                 f"Tendance **{m['label_fr']}**, el coupure 9rib ~{m['ttf_days_ml']:.0f} ayem (extrapolation linéaire : {m['ttf_days_linear']:.0f} j). Pente SNR {m['snr_slope_db_per_day']:+.2f} dB/jour.")
    return L("Pas de prédiction disponible pour cette analyse.", "Ma famech prédiction lel analyse hedhi.")


def _modulation(ctx: dict, lang: str) -> str | None:
    r = ctx.get("copper")
    if not r:
        return None
    m = r["modulation"]
    rates = ", ".join(f"{k} {v:.0f} Mbit/s" for k, v in m["rates_mbps"].items() if v > 0)
    return _pick(lang, f"Profil recommandé : **{m['recommended']}**. Débits descendants estimés : {rates or 'aucun'}." + (f" {m['note']}" if m.get("note") else ""),
                 f"El profil li ynas7a bih : **{m['recommended']}**. Débits descendants : {rates or 'ma famech'}." + (f" {m['note']}" if m.get("note") else ""))


def _budget(ctx: dict, lang: str) -> str | None:
    if "budget" in ctx:
        return _summary("budget", ctx["budget"], lang)
    r = ctx.get("fiber")
    if r:
        return _pick(lang, f"Puissance reçue à l'ONU : {r['rx_power_dbm']:.1f} dBm, marge {r['margin_db']:.1f} dB par rapport à la sensibilité B+ (−27 dBm).",
                     f"Puissance reçue fil ONU : {r['rx_power_dbm']:.1f} dBm, marge {r['margin_db']:.1f} dB men sensibilité B+ (−27 dBm).")
    return None


def _reliability(dom: str, r: dict, lang: str) -> str:
    f = r.get("fault") if isinstance(r.get("fault"), dict) else None
    conf = f["confidence"] if f else None
    s = _pick(lang, "Fiabilité : ", "El fiabilité : ")
    if conf is not None:
        s += _pick(lang, f"confiance du modèle {conf:.0%}. ", f"thi9et el modèle {conf:.0%}. ")
    s += _pick(lang, "Les modèles sont entraînés sur des données simulées : leurs scores sont optimistes. Confirmez par une 2e mesure avant d'intervenir, surtout pour une localisation ou un soupçon d'écoute.",
               "El modèles mdarrbin 3la données simulées : el scores mta3hom optimistes. Ta2akked b mesure théniya 9bal ma ttdakhkhal, khassatan fil localisation wella ichtibeh fi écoute.")
    return s


def _glossary(q: str, lang: str) -> str | None:
    ql = q.lower()
    aliases = {"atténuation": r"att[ée]nuation", "splitter": r"(splitter|coupleur)", "pon": r"\b(g?pon)\b", "onu": r"\b(onu|ont)\b", "tdr": r"\btdr\b|r[ée]flectom[ée]trie temp",
               "otdr": r"\botdr\b", "snr": r"\bsnr\b", "crc": r"\bcrc\b", "olt": r"\bolt\b", "nap": r"\bnap\b", "fdh": r"\bfdh\b", "rsrp": r"\brsrp\b", "sinr": r"\bsinr\b",
               "imsi": r"imsi", "bler": r"\bbler\b", "tscm": r"\btscm\b"}
    for k, rx in aliases.items():
        if re.search(rx, ql):
            return GLOSSARY[k][1 if lang == "tn" else 0]
    return None


SUGGESTIONS = {
    "fr": ["Fais un résumé du diagnostic", "Où se trouve le défaut ?", "Que faire sur le terrain ?", "Est-ce urgent ?", "Y a-t-il un risque d'écoute ?", "Quand la ligne va-t-elle tomber en panne ?", "C'est quoi le TDR ?"],
    "tn": ["Chnowa el diagnostic ?", "Ouin el défaut ?", "Chnowa ndir fil terrain ?", "Mouch isti3jel ?", "Fama risque écoute ?", "Wakt tetla3 el panne ?", "Chnowa houwa el TDR ?"],
}


def answer_rules(question: str, ctx: dict, recent: list[str] | None = None) -> dict:
    lang = detect_language(question)
    q = question.strip()
    L = lambda fr, tn: _pick(lang, fr, tn)
    gloss = _glossary(q, lang) if re.search(INTENTS["glossary"], q.lower()) else None
    if gloss:
        return {"answer": gloss, "mode": "règles", "lang": lang, "domains": []}
    if re.search(INTENTS["hello"], q.lower()):
        return {"answer": L("Bonjour ! Lancez une analyse (cuivre, fibre ou radio), puis posez-moi vos questions : diagnostic, position, conseils, urgence, sécurité, prévision.",
                            "Marhba ! A3mel analyse (cuivre, fibre wella radio) w ba3d es2elni : diagnostic, position, conseils, urgence, sécurité, prévision."),
                "mode": "règles", "lang": lang, "domains": []}
    if not ctx:
        return {"answer": L("Je n'ai encore aucun résultat d'analyse. Faites d'abord une analyse dans l'onglet Cuivre, Fibre ou Radio, puis revenez me poser la question.",
                            "Mazel ma 3andi 7ata résultat. A3mel analyse fil onglet Cuivre, Fibre wella Radio w erja3 es2elni."), "mode": "règles", "lang": lang, "domains": []}
    ql = q.lower()
    dom = detect_domain(ql, ctx, recent or [])
    r = ctx[dom]
    intent = next((k for k in ("urgency", "security", "where", "actions", "future", "modulation", "budget", "reliability", "summary") if re.search(INTENTS[k], ql)), "summary")
    if dom == "fiber_history" and intent == "summary":
        intent = "future"
    if intent == "urgency":
        a = _urgency(dom, r, lang)
    elif intent == "security":
        a = _security(dom, r, lang)
    elif intent == "where":
        a = _where(dom, r, lang)
    elif intent == "actions":
        a = _actions(dom, r, lang)
    elif intent == "future":
        a = _future(dom, r, lang)
    elif intent == "modulation":
        a = _modulation(ctx, lang) or L("Le profil de modulation ne concerne que la ligne cuivre : lancez une analyse cuivre.", "El profil mta3 modulation lel cuivre bark : a3mel analyse cuivre.")
    elif intent == "budget":
        a = _budget(ctx, lang) or L("Pas de budget optique disponible : lancez l'analyse fibre ou le calculateur.", "Ma famech budget optique : a3mel analyse fibre wella el calculateur.")
    elif intent == "reliability":
        a = _reliability(dom, r, lang)
    else:
        a = _summary(dom, r, lang)
        if _fault(r, dom) not in (None, "healthy") and dom != "fiber_history":
            a += "\n\n" + _actions(dom, r, lang)
    return {"answer": a, "mode": "règles", "lang": lang, "domains": [dom]}


# ----------------------------------------------------------------------------- mode LLM (optionnel)
SYSTEM_PROMPT = (
    "Tu es l'assistant d'AI Line Analyzer, outil d'analyse de lignes télécom (cuivre xDSL, fibre GPON, radio 4G/5G/6G) pour techniciens terrain et ingénieurs réseau. "
    "Réponds dans la langue et le registre de l'utilisateur : français, ou dialecte tunisien écrit en lettres latines s'il l'utilise. "
    "Appuie-toi UNIQUEMENT sur les résultats d'analyse fournis pour tout chiffre (distance, confiance, dB, jours) : n'invente aucune valeur, et dis-le si une information manque. "
    "Donne des conseils concrets, courts et ordonnés, rappelle l'incertitude (± P90, données d'entraînement simulées) et recommande de confirmer sur le terrain avant une intervention. "
    "Pour un soupçon d'écoute ou de prélèvement : ne jamais encourager à déconnecter ou manipuler, recommander de documenter et d'alerter la sécurité. Réponse de 8 lignes maximum.")


def llm_available() -> bool:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
        return True
    except Exception:
        return False


def answer_llm(question: str, ctx: dict, history: list[dict] | None = None) -> dict:
    import anthropic
    client = anthropic.Anthropic()
    msgs = [{"role": m["role"], "content": m["content"]} for m in (history or [])[-6:] if m["role"] in ("user", "assistant")]
    msgs.append({"role": "user", "content": f"Résultats d'analyse (JSON) :\n{json.dumps(compact_context(ctx), ensure_ascii=False, default=float)}\n\nQuestion : {question}"})
    r = client.messages.create(model=os.environ.get("ASSISTANT_MODEL", "claude-sonnet-5-5"), max_tokens=700, system=SYSTEM_PROMPT, messages=msgs)
    return {"answer": "".join(b.text for b in r.content if getattr(b, "type", "") == "text"), "mode": "LLM", "lang": detect_language(question), "domains": list(ctx)}


def ask(question: str, ctx: dict, recent: list[str] | None = None, history: list[dict] | None = None, use_llm: bool = False) -> dict:
    """Point d'entrée : LLM si demandé et disponible, sinon (ou en cas d'échec) mode règles."""
    if use_llm and ctx and llm_available():
        try:
            return answer_llm(question, ctx, history)
        except Exception as e:
            out = answer_rules(question, ctx, recent)
            out["note"] = f"LLM indisponible ({str(e)[:80]}) : réponse par règles."
            return out
    return answer_rules(question, ctx, recent)
