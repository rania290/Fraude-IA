# 🛡️ FraudGuard - Détection de Fraude Bancaire

FraudGuard est une application de détection de fraude bancaire utilisant le machine learning pour identifier les transactions frauduleuses en temps réel.

## 📋 Fonctionnalités

- 🔐 Authentification sécurisée des utilisateurs
- 🔍 Analyse en temps réel des transactions
- 📊 Visualisations et statistiques des transactions
- 📈 Historique des analyses avec export CSV
- 🤖 Modèle de machine learning entraîné sur le dataset Credit Card Fraud Detection

## 🚀 Installation

1. Cloner le dépôt :
```bash
git clone https://github.com/votre-username/fraudguard.git
cd fraudguard
```

2. Créer un environnement virtuel avec Python 3.10 :
```bash
conda create --name ml python=3.10
conda activate ml
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## 💻 Utilisation

1. Entraîner le modèle :
```bash
python model/train_model.py
```

2. Démarrer l'API FastAPI :
```bash
python api.py
```

3. Lancer l'interface Streamlit :
```bash
streamlit run app.py
```

4. Ouvrir votre navigateur à l'adresse : http://localhost:8501

## 📊 Structure du Projet

```
fraudguard/
├── api.py                 # API FastAPI
├── app.py                 # Interface Streamlit
├── requirements.txt       # Dépendances
├── data/                  # Données
│   └── creditcard.csv    # Dataset
├── model/                 # Modèles ML
│   ├── train_model.py    # Entraînement
│   ├── predict.py        # Prédictions
│   └── model.pkl         # Modèle entraîné
└── auth/                  # Authentification
    ├── auth_utils.py     # Utilitaires d'auth
    └── users.json        # Base utilisateurs
```

## 🔒 Sécurité

- Authentification par nom d'utilisateur et mot de passe
- Hachage sécurisé des mots de passe avec sel
- Validation des données d'entrée
- Gestion des erreurs et exceptions

## 📈 Visualisations

- Distribution des montants de transaction
- Répartition des transactions normales/frauduleuses
- Évolution temporelle des transactions
- Importance des caractéristiques du modèle

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
1. Fork le projet
2. Créer une branche pour votre fonctionnalité
3. Commiter vos changements
4. Pousser vers la branche
5. Ouvrir une Pull Request

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 🙏 Remerciements

- Dataset : [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Université Libre de Bruxelles (ULB) pour le dataset
- La communauté open source pour les bibliothèques utilisées 