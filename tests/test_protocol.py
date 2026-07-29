"""Le protocole d'évaluation tient-il ses promesses.

Le README publie des chiffres et une conclusion. Ces tests vérifient que le
protocole qui les produit ne dérive pas : la baseline, la taille du jeu de
test, et le fait que les deux familles de modèles soient bien jugées sur le
même échantillon. Ce sont les trois choses dont la fausseté rendrait tous les
scores incomparables sans qu'aucun ne paraisse suspect.
"""

from __future__ import annotations

import numpy as np
from scripts.benchmark import (
    RANDOM_STATE,
    TEST_SIZE,
    build_dataset,
    run_baselines,
    score,
)
from sklearn.model_selection import train_test_split

N_TEST = 93
N_TRAIN = 370


def split():
    X, y = build_dataset()
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def test_le_decoupage_est_celui_annonce():
    X_train, X_test, y_train, y_test = split()
    assert len(y_train) == N_TRAIN
    assert len(y_test) == N_TEST
    assert X_train.shape[1] == X_test.shape[1] == 48


def test_la_stratification_preserve_le_desequilibre():
    _, _, y_train, y_test = split()
    assert abs((y_train == 1).mean() - (y_test == 1).mean()) < 0.01


def test_la_baseline_constante_est_bien_celle_du_readme():
    """0,838 de F1 sans rien apprendre. C'est le chiffre qui donne son sens à
    tous les autres, et le seul que le README ne peut pas se permettre de rater."""
    _, _, _, y_test = split()
    baselines = run_baselines(y_test)
    assert round(baselines["always_anomaly"]["f1_anomaly"], 3) == 0.838
    assert round(baselines["always_anomaly"]["accuracy"], 3) == 0.720
    assert baselines["always_normal"]["f1_anomaly"] == 0.0


def test_un_modele_sur_etiquettes_melangees_ne_bat_pas_la_reponse_constante():
    """Le contrôle qui manquait, et celui qui rend crédible un score parfait.

    Random Forest atteint 1,0000 en test et 1,0000 ± 0,0000 en validation
    croisée. Sur 463 exécutions, un score parfait est d'abord un signal
    d'alerte : il faut prouver qu'il vient des données et non d'une fuite.

    Si le pipeline fuyait, le modèle resterait bon même avec des étiquettes
    tirées au hasard. Entraîné sur du bruit, il doit au contraire tomber SOUS
    la réponse constante, parce qu'il apprend des motifs qui n'existent pas.
    """
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    X, y = build_dataset()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    forest = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
    )

    reel = cross_val_score(forest, X, y, cv=cv, scoring="f1").mean()
    melange = cross_val_score(
        forest, X, np.random.default_rng(0).permutation(y), cv=cv, scoring="f1"
    ).mean()

    assert reel > 0.95, "les étiquettes réelles doivent rester apprenables"
    assert melange < 0.83, f"étiquettes mélangées à {melange:.4f} : le pipeline fuit"


def test_le_f1_de_reference_est_celui_de_la_classe_anomalie():
    """Le défaut central des notebooks : trois définitions du F1 selon l'endroit.

    Ici `f1_anomaly` porte sur la classe positive et `f1_weighted` sur la
    moyenne pondérée. Sur un jeu déséquilibré les deux diffèrent nettement, et
    ce test échouerait si quelqu'un les réunifiait par mégarde.
    """
    y_true = np.array([1] * 71 + [0] * 22)
    y_pred = np.array([1] * 93)
    result = score(y_true, y_pred, None)
    assert result["f1_anomaly"] > result["f1_weighted"]
    assert result["recall_anomaly"] == 1.0


def test_les_features_ne_contiennent_pas_la_cible():
    """Un modèle trivial doit rester loin de la perfection.

    Si une colonne de cible se glissait dans X, une régression logistique
    atteindrait 1,0000. Le seuil est haut exprès : il ne teste pas la qualité
    du modèle, il teste l'absence de fuite.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, y_train, y_test = split()
    model = Pipeline(
        [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000))]
    ).fit(X_train, y_train)
    assert model.score(X_test, y_test) < 0.999
