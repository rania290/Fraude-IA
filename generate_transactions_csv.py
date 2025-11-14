import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import joblib
import os

def generate_transaction_data(num_samples=100, target_fraud_ratio=0.3):
    """
    Génère des données de transactions avec un ratio cible de transactions frauduleuses.
    
    Args:
        num_samples (int): Nombre total de transactions à générer
        target_fraud_ratio (float): Ratio cible de transactions frauduleuses (entre 0 et 1)
    
    Returns:
        pandas.DataFrame: DataFrame contenant les transactions générées
    """
    # Vérifier si les fichiers du modèle et du scaler existent
    model_path = 'model/logisticregression.pkl'
    scaler_path = 'model/scaler.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Erreur: Modèle ou scaler non trouvé. Vérifiez que {model_path} et {scaler_path} existent.")
        return None
    
    # Charger le modèle et le scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # On va générer plus de transactions que nécessaire pour pouvoir atteindre le ratio cible
    extra_factor = 3
    extra_samples = num_samples * extra_factor
    
    # Générer des dates sur les 30 derniers jours
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    dates = [start_date + timedelta(
        seconds=random.randint(0, int((end_date - start_date).total_seconds()))) 
        for _ in range(extra_samples)]
    
    # Créer un DataFrame vide pour les données extra
    extra_df = pd.DataFrame()
    extra_df['timestamp'] = [d.strftime("%Y-%m-%d %H:%M:%S") for d in dates]
    
    # Générer des montants avec plus de variation pour augmenter la chance d'obtenir des fraudes
    amounts = np.zeros(extra_samples)
    
    # Créer trois groupes de transactions: petites, moyennes et grandes
    small_tx = int(extra_samples * 0.5)  # 50% petites transactions
    medium_tx = int(extra_samples * 0.3)  # 30% transactions moyennes
    large_tx = extra_samples - small_tx - medium_tx  # 20% grandes transactions
    
    # Mélanger les indices
    indices = np.random.permutation(extra_samples)
    small_indices = indices[:small_tx]
    medium_indices = indices[small_tx:small_tx+medium_tx]
    large_indices = indices[small_tx+medium_tx:]
    
    # Générer des montants pour chaque groupe
    amounts[small_indices] = np.random.exponential(scale=50, size=small_tx)
    amounts[small_indices] = np.clip(amounts[small_indices], 1, 150)
    
    amounts[medium_indices] = np.random.exponential(scale=200, size=medium_tx)
    amounts[medium_indices] = np.clip(amounts[medium_indices], 100, 500)
    
    amounts[large_indices] = np.random.exponential(scale=1000, size=large_tx) 
    amounts[large_indices] = np.clip(amounts[large_indices], 500, 5000)
    
    extra_df['Amount'] = amounts.round(2)
    
    # Générer les caractéristiques V1-V28 avec des distributions variées
    # Pour maximiser les chances d'obtenir des transactions frauduleuses
    for i in range(1, 29):
        v_name = f'V{i}'
        
        # Distribution de base pour tous
        values = np.random.normal(loc=0, scale=1.0, size=extra_samples)
        
        # Pour les grandes transactions, modifier certaines caractéristiques pour augmenter la chance de fraude
        if i in [1, 3]:  # Valeurs fortement négatives typiques des fraudes
            values[large_indices] = np.random.normal(loc=-4.0, scale=2.0, size=len(large_indices))
        elif i in [4, 10, 12]:  # Valeurs fortement positives typiques des fraudes
            values[large_indices] = np.random.normal(loc=3.0, scale=2.0, size=len(large_indices))
        elif i in [14, 17, 9]:  # Autres caractéristiques importantes
            values[large_indices] = np.random.normal(loc=-2.5, scale=1.8, size=len(large_indices))
            
        extra_df[v_name] = values.round(6)
    
    # Préparer les caractéristiques pour la prédiction
    features = extra_df[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 
                  'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                  'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']].values
    
    # Normaliser les données
    X_scaled = scaler.transform(features)
    
    # Utiliser le modèle pour prédire si les transactions sont frauduleuses
    extra_df['is_fraud'] = model.predict(X_scaled).astype(int)
    
    # Calculer les probabilités si disponible
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(X_scaled)
        extra_df['fraud_probability'] = probas[:, 1]
    
    # Sélectionner maintenant les transactions pour atteindre notre ratio cible
    fraud_df = extra_df[extra_df['is_fraud'] == 1]
    legit_df = extra_df[extra_df['is_fraud'] == 0]
    
    print(f"Données générées: {len(fraud_df)} fraudes et {len(legit_df)} légitimes")
    
    # Calculer combien de transactions de chaque type nous avons besoin
    target_fraud_count = int(num_samples * target_fraud_ratio)
    target_legit_count = num_samples - target_fraud_count
    
    # Vérifier si nous avons assez de transactions frauduleuses
    if len(fraud_df) < target_fraud_count:
        print(f"ATTENTION: Impossible d'atteindre {target_fraud_ratio*100}% de fraudes. Seulement {len(fraud_df)} trouvées.")
        target_fraud_count = len(fraud_df)
        target_legit_count = num_samples - target_fraud_count
    
    # Si nous n'avons pas assez de transactions légitimes (peu probable)
    if len(legit_df) < target_legit_count:
        print(f"ATTENTION: Pas assez de transactions légitimes. Utilisation de {len(legit_df)} transactions légitimes.")
        target_legit_count = len(legit_df)
        target_fraud_count = min(target_fraud_count, num_samples - target_legit_count)
    
    # Échantillonner les transactions de chaque type
    selected_fraud = fraud_df.sample(n=target_fraud_count, random_state=42)
    selected_legit = legit_df.sample(n=target_legit_count, random_state=42)
    
    # Combiner en un seul DataFrame
    df = pd.concat([selected_fraud, selected_legit])
    
    # Mélanger les lignes
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Ajouter l'identifiant de transaction
    df['transaction_id'] = [f"TX-{i:06d}" for i in range(1, len(df) + 1)]
    
    # Ajouter des descriptions
    merchants = [
        'Amazon', 'SuperMarché', 'Station Service', 'Restaurant', 'Boutique', 
        'Pharmacie', 'Transfert Bancaire', 'Électronique', 'Vêtements', 
        'Voyage', 'Hôtel', 'Transport', 'Assurance', 'Abonnement'
    ]
    
    fraud_descriptions = [
        "Transaction inhabituelle de {amount}€",
        "Achat suspect en ligne de {amount}€",
        "Paiement international de {amount}€",
        "Transaction non reconnue de {amount}€",
        "Retrait ATM inhabituel de {amount}€",
        "Transaction e-commerce suspect de {amount}€"
    ]
    
    legitimate_descriptions = [
        "Achat chez {merchant} de {amount}€",
        "Paiement à {merchant} de {amount}€",
        "Transaction normale chez {merchant} de {amount}€",
        "Abonnement mensuel {merchant} de {amount}€",
        "Facture {merchant} de {amount}€"
    ]
    
    descriptions = []
    for i, row in df.iterrows():
        merchant = random.choice(merchants)
        amount = row['Amount']
        
        if row['is_fraud'] == 1:
            template = random.choice(fraud_descriptions)
            descriptions.append(template.format(amount=f"{amount:.2f}"))
        else:
            template = random.choice(legitimate_descriptions)
            descriptions.append(template.format(merchant=merchant, amount=f"{amount:.2f}"))
    
    df['description'] = descriptions
    
    # Réorganiser les colonnes
    base_cols = ['transaction_id', 'timestamp', 'Amount', 'description', 'is_fraud']
    prob_col = ['fraud_probability'] if 'fraud_probability' in df.columns else []
    feature_cols = [f'V{i}' for i in range(1, 29)]
    
    final_cols = base_cols + prob_col + feature_cols
    return df[final_cols]

if __name__ == "__main__":
    # Générer 100 transactions avec 30% de fraudes
    transactions_df = generate_transaction_data(num_samples=100, target_fraud_ratio=0.3)
    
    if transactions_df is not None:
        # Sauvegarder dans un fichier CSV
        transactions_df.to_csv("sample_transactions.csv", index=False)
        print(f"\nFichier CSV généré avec {len(transactions_df)} transactions")
        
        # Afficher quelques statistiques
        num_fraud = transactions_df['is_fraud'].sum()
        print(f"- Transactions frauduleuses: {num_fraud} ({num_fraud/len(transactions_df)*100:.1f}%)")
        print(f"- Transactions légitimes: {len(transactions_df)-num_fraud} ({(len(transactions_df)-num_fraud)/len(transactions_df)*100:.1f}%)")
        
        # Afficher quelques exemples
        print("\nExemples de transactions:")
        for i, (_, row) in enumerate(transactions_df.head(5).iterrows()):
            status = "FRAUDULEUSE" if row['is_fraud'] == 1 else "LÉGITIME"
            print(f"{i+1}. {row['transaction_id']} - {row['description']} - {status}")
    else:
        print("Impossible de générer les données de transaction. Vérifiez que les modèles sont entraînés.")