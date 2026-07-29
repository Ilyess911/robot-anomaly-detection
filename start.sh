#!/usr/bin/env bash
#
# Prépare l'environnement et ouvre les notebooks.
#
#   ./start.sh
#
# Le script s'arrête à la première erreur plutôt que de continuer sur un
# environnement à moitié installé, ce qui produirait un échec plus loin et
# plus difficile à lire.

set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Création de l'environnement virtuel"
  python3 -m venv .venv
  ./.venv/bin/pip install --quiet --upgrade pip
  ./.venv/bin/pip install --quiet -r requirements.txt
  ./.venv/bin/python -m ipykernel install --user \
    --name robot-anomaly --display-name "Python (robot-anomaly)"
else
  echo "Environnement virtuel déjà présent"
fi

echo "Vérification des dépendances"
./.venv/bin/python -c "import numpy, pandas, scipy, sklearn, matplotlib, seaborn" || {
  echo "Dépendances manquantes, réinstallation"
  ./.venv/bin/pip install --quiet -r requirements.txt
}

echo
echo "Notebooks dans notebooks/, à exécuter dans l'ordre 01 à 05."
echo "Kernel à sélectionner : Python (robot-anomaly)"
echo

exec ./.venv/bin/jupyter notebook notebooks/
