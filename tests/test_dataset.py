"""Le jeu de données est-il bien celui que le README décrit.

Ces tests ne vérifient pas que le code s'exécute : ils vérifient que les
affirmations publiées restent vraies. Un README qui annonce 463 exécutions et
un parseur qui en rend 460 après un refactor est un mensonge involontaire, et
c'est exactement le genre que personne ne remarque.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.utils import create_statistical_features, encode_labels, load_robot_data

# Les chiffres du README. Les modifier ici oblige à les modifier là-bas.
N_EXECUTIONS = 463
N_RAW_FEATURES = 90
N_STAT_FEATURES = 48
N_ORIGINAL_LABELS = 16
N_NORMAL = 129  # « normal » (109) + « ok » (20)
SUBSET_SIZES = {"LP1": 88, "LP2": 47, "LP3": 47, "LP4": 117, "LP5": 164}


@pytest.fixture(scope="module")
def frame():
    return load_robot_data()


def test_toutes_les_executions_sont_chargees(frame):
    assert len(frame) == N_EXECUTIONS


def test_chaque_sous_ensemble_a_sa_taille(frame):
    counts = frame["source"].value_counts().to_dict()
    assert counts == SUBSET_SIZES


def test_une_execution_est_six_capteurs_sur_quinze_pas(frame):
    """90 valeurs par exécution, et pas une de plus : 6 capteurs x 15 relevés."""
    features = [c for c in frame.columns if c.startswith("feature_")]
    assert len(features) == N_RAW_FEATURES
    assert frame[features].isna().sum().sum() == 0


def test_le_desequilibre_annonce_est_le_bon(frame):
    """Le README dit 72 % d'anomalies. Ce chiffre porte toute la lecture des
    scores : s'il bouge, la baseline bouge, et les résultats changent de sens."""
    encoded, _ = encode_labels(frame.copy(), binary=True)
    normal = int((encoded["label_binary"] == 0).sum())
    assert normal == N_NORMAL
    assert frame["label"].nunique() == N_ORIGINAL_LABELS
    part_anomalies = 1 - normal / len(frame)
    assert 0.71 < part_anomalies < 0.73


def test_les_features_statistiques_sont_bien_quarante_huit(frame):
    encoded, _ = encode_labels(frame.copy(), binary=True)
    stats = create_statistical_features(encoded)
    meta = {"label", "label_encoded", "label_binary", "label_original", "source"}
    columns = [c for c in stats.columns if c not in meta]
    assert len(columns) == N_STAT_FEATURES


def test_aucune_feature_statistique_ne_reproduit_la_cible(frame):
    """Le piège qui a donné un F1 de 1,0000 pendant l'écriture du benchmark.

    `create_statistical_features` renvoie `label_encoded` et `label_binary` à
    côté des features. Les deux sont numériques, donc une sélection par type
    les embarque et le modèle apprend la réponse. Ce test échoue si une colonne
    de features devient un jour parfaitement corrélée à la cible.
    """
    encoded, _ = encode_labels(frame.copy(), binary=True)
    stats = create_statistical_features(encoded)
    meta = {"label", "label_encoded", "label_binary", "label_original", "source"}
    target = stats["label_binary"].to_numpy(dtype=float)

    for column in stats.columns:
        if column in meta:
            continue
        values = stats[column].to_numpy(dtype=float)
        if np.std(values) == 0:
            continue
        correlation = abs(np.corrcoef(values, target)[0, 1])
        assert correlation < 0.999, f"{column} reproduit la cible"


def test_les_deux_colonnes_de_cible_sont_d_accord(frame):
    """Les notebooks utilisent `label_binary`, le benchmark `label_encoded`.
    S'ils divergeaient, les deux ne mesureraient plus la même chose."""
    encoded, _ = encode_labels(frame.copy(), binary=True)
    stats = create_statistical_features(encoded)
    assert (stats["label_encoded"].to_numpy() == stats["label_binary"].to_numpy()).all()
