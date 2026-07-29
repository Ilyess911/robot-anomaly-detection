"""Comparatif honnête des modèles : mêmes données, même protocole, vrais repères.

Ce que ce script établit
------------------------

Les notebooks publiaient des scores sans rien contre quoi les lire. Ce script
mesure d'abord ce qu'obtient un non-modèle, puis ce qu'obtient un modèle bête,
et seulement ensuite ce qu'obtiennent les modèles réglés. L'écart entre ces
trois niveaux est le seul résultat qui ait un sens.

Il corrige quatre défauts de la chaîne d'origine :

1. **Une erreur d'étiquetage.** LP3 nomme sa classe saine « ok » et n'emploie
   jamais « normal ». Le codage binaire ne reconnaissait que « normal », donc
   les 20 exécutions saines de LP3 étaient comptées comme des défaillances et
   ce sous-ensemble paraissait défaillant à 100 %. Corrigé dans
   `src/utils.py` : 129 exécutions saines sur 463, et non 109.

2. **Aucun repère.** On ajoute la réponse constante, la meilleure feature seule
   à un seul seuil, et un arbre de profondeur 2.

3. **Un F1 aux définitions changeantes.** Le notebook 03 imprime une moyenne
   pondérée, le 05 le F1 de la classe positive, et le 04 les deux pour le même
   modèle. Ici `f1_anomaly` est la référence et `f1_weighted` est nommée.

4. **Deux jeux de test différents.** Le supervisé était évalué sur 93
   échantillons, le non supervisé sur 376. Les deux partagent désormais le même.

La fuite de standardisation
---------------------------

Les notebooks standardisent avant de découper, donc le scaler voit le jeu de
test. Le script rejoue les deux protocoles dans le MÊME environnement, avec les
mêmes graines : c'est la seule façon d'attribuer un écart à la fuite plutôt
qu'à cinq ans d'évolution de scikit-learn.

Usage
-----

    make benchmark
    ./.venv/bin/python scripts/benchmark.py --output reports/benchmark.json

La sortie console est lisible ; le JSON est versionné pour que le README ne
demande jamais qu'on le croie sur parole.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import sklearn
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
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import create_statistical_features, encode_labels, load_robot_data

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
META_COLUMNS = frozenset({"label", "label_encoded", "label_binary", "label_original", "source"})

#: Noms des 48 features, dans l'ordre des colonnes de X. Rempli par
#: build_dataset, pour que le repère « un seul capteur » puisse nommer lequel.
FEATURE_NAMES: list[str] = []


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

    FEATURE_NAMES.clear()
    FEATURE_NAMES.extend(columns)
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


def run_baselines(y_test: np.ndarray, X_train=None, y_train=None, X_test=None) -> dict:
    """Les repères sans lesquels aucun score ne se lit.

    Trois niveaux de bêtise croissante, et c'est l'écart entre eux qui dit ce
    qu'un modèle apporte réellement :

      1. répondre toujours « anomalie ». Le jeu en contient 72 %, donc ce
         non-modèle obtient déjà un F1 élevé sur la classe positive ;
      2. un seul capteur, un seul seuil. C'est le repère que presque personne
         ne publie, et c'est le plus instructif : s'il suffit, la question
         n'était pas difficile ;
      3. un arbre de profondeur 2, soit trois décisions en tout.

    Un modèle réglé par recherche d'hyperparamètres ne se juge pas contre zéro,
    il se juge contre ces trois-là.
    """
    results = {
        "always_anomaly": score(y_test, np.ones_like(y_test), None),
        "always_normal": score(y_test, np.zeros_like(y_test), None),
    }

    if X_train is None:
        return results

    best = None
    for index in range(X_train.shape[1]):
        stump = DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE)
        stump.fit(X_train[:, [index]], y_train)
        current = score(y_test, stump.predict(X_test[:, [index]]), None)
        if best is None or current["f1_anomaly"] > best[0]["f1_anomaly"]:
            best = (current, index)

    name = FEATURE_NAMES[best[1]] if best[1] < len(FEATURE_NAMES) else str(best[1])
    results["best_single_feature"] = {**best[0], "feature": name}

    shallow = DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE)
    shallow.fit(X_train, y_train)
    results["depth_2_tree"] = score(y_test, shallow.predict(X_test), None)

    return results


def run_supervised(X_train, X_test, y_train, y_test, leaky: bool = False) -> dict:
    """Cherche puis évalue chaque modèle.

    `leaky=False` place le scaler dans le Pipeline : il est réajusté sur les
    seuls plis d'entraînement à chaque tour de validation croisée.

    `leaky=True` reproduit le défaut des notebooks : les données arrivent déjà
    standardisées sur le jeu complet, le scaler ayant vu le jeu de test. Le
    pipeline ne contient alors plus que le modèle.

    Les deux modes tournent dans le MÊME environnement, avec les mêmes graines
    et les mêmes grilles. C'est la seule façon d'attribuer un écart à la fuite
    plutôt qu'à une version de bibliothèque : comparer les chiffres d'un
    notebook exécuté en 2025 sous Python 3.9 à ceux d'un script exécuté
    aujourd'hui mélangerait deux variables et ne prouverait rien.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for name, (estimator, grid) in GRIDS.items():
        steps = (
            [("model", estimator)]
            if leaky
            else [("scaler", StandardScaler()), ("model", estimator)]
        )
        pipeline = Pipeline(steps)
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
        "n_train_normal": len(train_normal),
    }

    oc_svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
    oc_svm.fit(train_normal)
    results["one_class_svm"] = {
        **score(y_test, (oc_svm.predict(test_scaled) == -1).astype(int), None),
        "n_train_normal": len(train_normal),
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

    # Le protocole fuité des notebooks : standardiser AVANT de découper. Le
    # découpage porte alors sur des données que le scaler a toutes vues.
    X_leaky = StandardScaler().fit_transform(X)
    Xl_train, Xl_test, yl_train, yl_test = train_test_split(
        X_leaky, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Jeu complet      : {X.shape[0]} exécutions, {X.shape[1]} features statistiques")
    print(f"Entraînement     : {len(y_train)} ({int((y_train == 1).sum())} anomalies)")
    print(f"Test             : {len(y_test)} ({int((y_test == 1).sum())} anomalies)")
    baselines = run_baselines(y_test, X_train, y_train, X_test)
    print("\nLes repères à battre, du plus bête au moins bête")
    for name, value in baselines.items():
        print(
            f"  {name:22s} f1(anomalie)={value['f1_anomaly']:.4f} accuracy={value['accuracy']:.4f}"
        )

    print("\nSupervisé, scaler ajusté sur le jeu complet (le défaut des notebooks)")
    leaky = run_supervised(Xl_train, Xl_test, yl_train, yl_test, leaky=True)

    print("\nSupervisé, scaler dans le pipeline")
    supervised = run_supervised(X_train, X_test, y_train, y_test)

    print("\nCoût réel de la fuite, à environnement identique")
    print(f"  {'modèle':22s} {'fuité':>8s} {'corrigé':>9s} {'écart':>8s}")
    for name in supervised:
        before = leaky[name]["f1_anomaly"]
        after = supervised[name]["f1_anomaly"]
        print(f"  {name:22s} {before:8.4f} {after:9.4f} {after - before:+8.4f}")

    print("\nNon supervisé, entraîné sur les seules exécutions normales")
    unsupervised = run_unsupervised(X_train, X_test, y_train, y_test)

    report = {
        "protocol": {
            "n_total": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_train": len(y_train),
            "n_test": len(y_test),
            "test_size": TEST_SIZE,
            "cv_folds": CV_FOLDS,
            "random_state": RANDOM_STATE,
            "scaler_inside_cv": True,
            "anomaly_share_test": float((y_test == 1).mean()),
        },
        "environment": {
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "baselines": baselines,
        "supervised_leaky": leaky,
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
