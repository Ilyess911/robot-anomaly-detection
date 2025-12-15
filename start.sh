#!/bin/bash

# Script de démarrage pour le projet Robot Anomaly Detection

echo "🚀 Démarrage du projet Robot Anomaly Detection"
echo ""

# Activer l'environnement virtuel
if [ ! -d ".venv" ]; then
    echo "❌ Environnement virtuel non trouvé. Création..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    python -m ipykernel install --user --name=robot-anomaly --display-name="Python (robot-anomaly)"
else
    echo "✅ Activation de l'environnement virtuel..."
    source .venv/bin/activate
fi

# Vérifier que les dépendances sont installées
echo ""
echo "📦 Vérification des dépendances..."
python3 -c "import numpy, pandas, matplotlib, sklearn, jupyter; print('✅ Toutes les dépendances sont installées')" 2>/dev/null || {
    echo "⚠️  Installation des dépendances manquantes..."
    pip install -r requirements.txt
}

# Lancer Jupyter Notebook
echo ""
echo "📓 Lancement de Jupyter Notebook..."
echo "   Les notebooks sont disponibles dans le dossier 'notebooks/'"
echo "   Kernel disponible: Python (robot-anomaly)"
echo ""
jupyter notebook
