"""Comparatif honnête des modèles, sans fuite de données.

Pourquoi ce script existe
-------------------------

Les notebooks standardisent le jeu complet avant de le découper :

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)          # voit tout le jeu
    X_train, X_test, ... = train_test_split(X_scaled, ...)

Le scaler apprend donc la moyenne et l'écart-type du jeu de test avant de
l'évaluer. C'est une fuite : les métriques publiées sont optimistes. Sur un jeu
de 463 exécutions dont 93 en test, l'effet est faible, mais il est réel et il
n'est pas mesurable tant qu'on ne l'a pas mesuré.

Ce script rejoue le même protocole (mêmes features statistiques, même découpage
80/20 stratifié, même graine, mêmes grilles d'hyperparamètres, même validation
croisée à 5 plis) avec une seule différence : le scaler vit dans un Pipeline,
donc il est réajusté sur les seuls plis d'entraînement à chaque tour.

Il corrige au passage deux autres défauts des notebooks :

  - le F1 rapporté n'était pas toujours le même. Le notebook 03 imprime une
    moyenne pondérée, le 05 le F1 de la classe positive, et le 04 les deux pour
    le même modèle. Ici le F1 de la classe « anomalie » est la mesure de
    référence, et la moyenne pondérée est reportée à côté, nommée.
  - le supervisé était évalué sur 93 échantillons et le non supervisé sur 376,
    ce qui rendait les deux familles incomparables. Ici elles partagent le même
    jeu de test.

Usage
-----

    ./.venv/bin/python scripts/benchmark.py
    ./.venv/bin/python scripts/benchmark.py --output reports/benchmark.json

La sortie console est un tableau lisible ; le JSON sert à rejouer la
comparaison plus tard et à vérifier qu'un changement de code n'a pas déplacé
les chiffres sans qu'on s'en aperçoive.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, OneClassSVM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import create_statistical_features, encode_labels, load_robot_data  # noqa: E402

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Mêmes grilles que src/models.py, à deux détails près, signalés en commentaire.
GRIDS = {
    "logistic_regression": (
        LogisticRegression(random_state=RANDOM_STATE),
        {
            "model__C": [0.1, 1, 10, 100],
            "model__penalty": ["l1", "l2"],
            "model__solver": ["liblinear", "saga"],
            # Les notebooks descendaient à 100 itérations, ce qui produisait
            # 260 ConvergenceWarning. Le plancher est relevé : un modèle qui
            # n'a pas convergé n'est pas un résultat, c'est un incident.
            "model__max_iter": [500, 2000],
        },
    ),
    "random_forest": (
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [10, 20, None],
        },
    ),
    "svm_rbf": (
        SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        {
            "model__C": [0.1, 1, 10],
            "model__gamma": ["scale", "auto", 0.001, 0.01],
        },
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.1, 0.2],
            "model__max_depth": [3, 5, 7],
        },
    ),
}


#: Colonnes de service du frame statistique. Tout le reste est une feature.
#: `label_encoded` et `label_binary` sont numériques et valent la cible : les
#: exclure par leur nom est plus sûr que de filtrer sur le type.
META_COLUMNS = frozenset(
    {"label", "label_encoded", "label_binary", "label_original", "source"}
)


def build_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Reproduit la préparation des notebooks : features statistiques, cible binaire.

    Les colonnes de service sont exclues nommément et non par leur type. Le
    frame renvoyé par create_statistical_features contient `label_encoded` et
    `label_binary`, deux colonnes numériques qui SONT la cible : les laisser
    entrer donne un F1 de 1,0000, ce qui n'est pas un résultat mais une fuite.
    """
    frame = load_robot_data()
    frame, _ = encode_labels(frame, binary=True)
    features = create_statistical_features(frame)

    columns = [c for c in features.columns if c not in META_COLUMNS]
    if len(columns) != 48:
        raise RuntimeError(f"48 features statistiques attendues, {len(columns)} trouvées")

    return features[columns].to_numpy(dtype=float), frame["label_encoded"].to_numpy(dtype=int)


def score(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None) -> dict:
    """Un seul jeu de métriques, nommées sans ambiguïté.

    `f1_anomaly` est la mesure de référence : la classe positive est l'anomalie,
    et c'est elle qui compte dans un contexte de sécurité. `f1_weighted` est
    donnée à côté parce que c'est elle que les notebooks affichaient.
    """
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_anomaly": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_anomaly": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_anomaly": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    result["roc_auc"] = roc_auc_score(y_true, y_proba) if y_proba is not None else None
    return result


def run_baselines(y_test: np.ndarray) -> dict:
    """Les deux prédictions constantes, sans lesquelles aucun score ne se lit.

    Le jeu contient 76 % d'anomalies. Un modèle qui répond toujours « anomalie »
    obtient donc déjà un F1 élevé sur la classe positive sans rien apprendre.
    Publier 0,96 sans publier ce repère laisse croire à une performance qui est
    pour les trois quarts un effet du déséquilibre des classes.
    """
    always_anomaly = np.ones_like(y_test)
    always_normal = np.zeros_like(y_test)
    return {
        "always_anomaly": score(y_test, always_anomaly, None),
        "always_normal": score(y_test, always_normal, None),
    }


def run_supervised(X_train, X_test, y_train, y_test) -> dict:
    """Chaque modèle est cherché puis évalué avec le scaler à l'intérieur du pipeline."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for name, (estimator, grid) in GRIDS.items():
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
        search = GridSearchCV(pipeline, grid, cv=cv, scoring="f1", n_jobs=-1, refit=True)
        search.fit(X_train, y_train)

        y_pred = search.predict(X_test)
        y_proba = search.predict_proba(X_test)[:, 1]

        results[name] = {
            **score(y_test, y_pred, y_proba),
            "cv_f1_mean": search.best_score_,
            "cv_f1_std": search.cv_results_["std_test_score"][search.best_index_],
            "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
        }
        print(f"  {name:20s} f1(anomalie)={results[name]['f1_anomaly']:.4f}")

    return results


def run_unsupervised(X_train, X_test, y_train, y_test) -> dict:
    """Non supervisé, évalué sur le MÊME jeu de test que le supervisé.

    Les deux modèles n'apprennent que sur les exécutions normales du jeu
    d'entraînement, ce qui est le cadre d'emploi réel : on dispose d'exemples de
    fonctionnement sain, pas d'un catalogue de pannes.
    """
    scaler = StandardScaler().fit(X_train[y_train == 0])
    train_normal = scaler.transform(X_train[y_train == 0])
    test_scaled = scaler.transform(X_test)

    results = {}

    # `contamination` décrit la proportion d'anomalies dans le jeu D'ENTRAÎNEMENT,
    # pas dans le jeu de test. Ici l'entraînement ne contient que des exécutions
    # normales, donc la contamination attendue est proche de zéro et 'auto' est
    # le réglage correct. Y mettre la proportion d'anomalies du jeu complet, 76 %,
    # serait à la fois faux et refusé par scikit-learn, qui plafonne à 0,5.
    forest = IsolationForest(contamination="auto", random_state=RANDOM_STATE, n_jobs=-1)
    forest.fit(train_normal)
    results["isolation_forest"] = {
        **score(y_test, (forest.predict(test_scaled) == -1).astype(int), None),
        "n_train_normal": int(len(train_normal)),
    }

    oc_svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
    oc_svm.fit(train_normal)
    results["one_class_svm"] = {
        **score(y_test, (oc_svm.predict(test_scaled) == -1).astype(int), None),
        "n_train_normal": int(len(train_normal)),
    }

    for name, value in results.items():
        print(f"  {name:20s} f1(anomalie)={value['f1_anomaly']:.4f}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, default=None, help="Chemin du rapport JSON")
    args = parser.parse_args()

    X, y = build_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Jeu complet      : {X.shape[0]} exécutions, {X.shape[1]} features statistiques")
    print(f"Entraînement     : {len(y_train)} ({int((y_train == 1).sum())} anomalies)")
    print(f"Test             : {len(y_test)} ({int((y_test == 1).sum())} anomalies)")
    baselines = run_baselines(y_test)
    print("\nPrédictions constantes, le repère à battre")
    for name, value in baselines.items():
        print(f"  {name:20s} f1(anomalie)={value['f1_anomaly']:.4f} "
              f"accuracy={value['accuracy']:.4f}")

    print("\nSupervisé, scaler dans le pipeline")
    supervised = run_supervised(X_train, X_test, y_train, y_test)

    print("\nNon supervisé, entraîné sur les seules exécutions normales")
    unsupervised = run_unsupervised(X_train, X_test, y_train, y_test)

    report = {
        "protocol": {
            "n_total": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "test_size": TEST_SIZE,
            "cv_folds": CV_FOLDS,
            "random_state": RANDOM_STATE,
            "scaler_inside_cv": True,
            "anomaly_share_test": float((y_test == 1).mean()),
        },
        "baselines": baselines,
        "supervised": supervised,
        "unsupervised": unsupervised,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"\nRapport écrit dans {args.output}")

    best = max(supervised.items(), key=lambda kv: kv[1]["f1_anomaly"])
    print(f"\nMeilleur modèle supervisé : {best[0]} (f1 anomalie {best[1]['f1_anomaly']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
