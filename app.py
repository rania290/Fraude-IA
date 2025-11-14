import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime
import json
from database import init_db, verify_user, hash_password, create_user, get_user
import logging
import time
import os

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="FraudGuard",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .css-1d391kg {
        padding: 1rem;
    }
    .transaction-card {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
        background-color: #f9f9f9;
        cursor: pointer;
    }
    .transaction-card:hover {
        background-color: #f0f0f0;
    }
    </style>
""", unsafe_allow_html=True)

# Configuration de l'API
API_URL = "http://localhost:8000"
PREDICT_ENDPOINT = f"{API_URL}/predict"
MODELS_ENDPOINT = f"{API_URL}/models"

# Chemin du fichier CSV des transactions
TRANSACTIONS_CSV = "sample_transactions.csv"

def get_available_models():
    """Récupère la liste des modèles disponibles depuis l'API."""
    try:
        response = requests.get(MODELS_ENDPOINT)
        if response.status_code == 200:
            return response.json().get("models", ["LogisticRegression"])
        logger.warning(f"Impossible de récupérer les modèles (Erreur {response.status_code})")
        return ["LogisticRegression"]
    except requests.exceptions.ConnectionError:
        logger.error("Erreur de connexion à l'API")
        st.error("Erreur de connexion à l'API. Vérifiez que l'API est en cours d'exécution.")
        return ["LogisticRegression"]
    except Exception as e:
        logger.error(f"Erreur inattendue: {str(e)}")
        return ["LogisticRegression"]

def load_transaction_examples():
    """Charge les exemples de transactions depuis le fichier CSV."""
    try:
        if os.path.exists(TRANSACTIONS_CSV):
            return pd.read_csv(TRANSACTIONS_CSV)
        else:
            logger.warning(f"Fichier CSV non trouvé: {TRANSACTIONS_CSV}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Erreur lors du chargement des transactions: {str(e)}")
        return pd.DataFrame()

def login_page():
    """Affiche la page de connexion."""
    st.title("🔐 Connexion")
    
    with st.form("login_form"):
        username = st.text_input("👤 Nom d'utilisateur")
        password = st.text_input("🔑 Mot de passe", type="password")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.form_submit_button("🚪 Se connecter", use_container_width=True):
                if not username or not password:
                    st.error("❌ Veuillez remplir tous les champs")
                    return
                
                try:
                    if verify_user(username, password):
                        st.session_state["authenticated"] = True
                        st.session_state["username"] = username
                        st.rerun()
                    else:
                        st.error("❌ Nom d'utilisateur ou mot de passe incorrect")
                except Exception as e:
                    logger.error(f"Erreur lors de la connexion: {str(e)}")
                    st.error("❌ Une erreur est survenue lors de la connexion")
        
        with col2:
            if st.form_submit_button("📝 S'inscrire", use_container_width=True):
                st.session_state["show_register"] = True
                st.rerun()

def register_page():
    """Affiche la page d'inscription."""
    st.title("📝 Inscription")
    
    if "registration_success" in st.session_state and st.session_state["registration_success"]:
        st.success("✅ Inscription réussie! Vous pouvez maintenant vous connecter.")
        time.sleep(2)
        st.session_state["registration_success"] = False
        st.session_state["show_register"] = False
        st.rerun()
    
    with st.form("register_form"):
        username = st.text_input("👤 Nom d'utilisateur")
        email = st.text_input("📧 Email")
        password = st.text_input("🔑 Mot de passe", type="password")
        confirm_password = st.text_input("🔄 Confirmer le mot de passe", type="password")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.form_submit_button("✅ S'inscrire", use_container_width=True):
                if not username or not email or not password or not confirm_password:
                    st.error("❌ Veuillez remplir tous les champs")
                    return
                
                if password != confirm_password:
                    st.error("❌ Les mots de passe ne correspondent pas")
                    return
                
                try:
                    if get_user(username):
                        st.error("❌ Ce nom d'utilisateur est déjà pris")
                        return
                    
                    if create_user(username, password, email):
                        st.session_state["registration_success"] = True
                        st.rerun()
                    else:
                        st.error("❌ Une erreur est survenue lors de l'inscription")
                except Exception as e:
                    logger.error(f"Erreur lors de l'inscription: {str(e)}")
                    st.error("❌ Une erreur est survenue lors de l'inscription")
        
        with col2:
            if st.form_submit_button("🔙 Retour à la connexion", use_container_width=True):
                st.session_state["show_register"] = False
                st.rerun()

def check_api_connection():
    """Vérifie si l'API est accessible."""
    try:
        response = requests.get(f"{API_URL}/health")
        return response.status_code == 200
    except:
        return False

def main():
    """Fonction principale de l'application."""
    try:
        init_db()
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation de la base de données: {str(e)}")
        st.error("❌ Erreur de connexion à la base de données")
        return

    # Initialize session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "show_register" not in st.session_state:
        st.session_state.show_register = False
    if "transaction_history" not in st.session_state:
        st.session_state.transaction_history = []
    if "available_models" not in st.session_state:
        st.session_state.available_models = ["LogisticRegression"]
    if "transaction_examples" not in st.session_state:
        st.session_state.transaction_examples = load_transaction_examples()
    if "selected_transaction" not in st.session_state:
        st.session_state.selected_transaction = None
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = None
    
    # Afficher la page appropriée
    if not st.session_state.authenticated:
        if st.session_state.show_register:
            register_page()
        else:
            login_page()
        return

    # Vérifier la connexion à l'API
    if not check_api_connection():
        st.error("❌ Impossible de se connecter à l'API. Veuillez vérifier que l'API est en cours d'exécution.")
        st.info("ℹ️ Pour démarrer l'API, ouvrez un terminal et exécutez la commande : `python api.py`")
        
        if st.button("🚪 Se déconnecter"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()
        return

    # Mettre à jour la liste des modèles disponibles
    st.session_state.available_models = get_available_models()

    # Interface principale
    st.title("🛡️ FraudGuard - Détection de Fraude")
    
    # Sidebar
    with st.sidebar:
        st.success(f"👋 Bienvenue, {st.session_state.username}!")
        if st.button("🚪 Se déconnecter", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()
        
        st.markdown("---")
        
        # Sélection du modèle
        selected_model = st.selectbox(
            "🤖 Choisir le modèle",
            st.session_state.available_models
        )
        
        st.markdown("---")
        st.markdown("### 📊 Statistiques")
        st.markdown(f"Transactions analysées: {len(st.session_state.transaction_history)}")
        
        if st.session_state.transaction_history:
            fraud_count = sum(1 for t in st.session_state.transaction_history if t['prediction'] == 1)
            st.markdown(f"Fraudes détectées: {fraud_count}")
            fraud_rate = (fraud_count / len(st.session_state.transaction_history)) * 100
            st.markdown(f"Taux de fraude: {fraud_rate:.2f}%")
        
        # Option d'upload de CSV
        st.markdown("---")
        st.markdown("### 📁 Importer des transactions")
        uploaded_file = st.file_uploader("Importer un fichier CSV", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.transaction_examples = df
                st.success(f"✅ {len(df)} transactions importées avec succès!")
            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")
    
    # Contenu principal
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Analyse", "📋 Exemples", "📈 Statistiques", "📋 Historique"])
    
    # Si une transaction a été sélectionnée, passer automatiquement à l'onglet Analyse
    if "active_tab" in st.session_state and st.session_state.active_tab is not None:
        st.info("✅ Transaction chargée! Utilisez le formulaire d'analyse ci-dessous.")
        st.session_state.active_tab = None
    
    with tab1:
        st.header("🔍 Analyse de Transaction")
        
        # Bouton pour effacer le formulaire doit être en DEHORS du formulaire
        if "selected_transaction" in st.session_state and st.session_state.selected_transaction is not None:
            if st.button("🧹 Effacer le formulaire"):
                st.session_state.selected_transaction = None
                st.rerun()
        
        with st.form("transaction_form"):
            # Récupérer les valeurs pré-remplies depuis la transaction sélectionnée si disponible
            prefill_data = {}
            if "selected_transaction" in st.session_state and st.session_state.selected_transaction is not None:
                prefill_data = st.session_state.selected_transaction
            
            amount = st.number_input(
                "💰 Montant", 
                min_value=0.0, 
                value=float(prefill_data.get("Amount", 100.0)), 
                step=10.0
            )
            
            st.subheader("Caractéristiques de la Transaction")
            cols = st.columns(4)
            features = {}
            
            # Remplir les champs V1-V28 avec les valeurs de la transaction sélectionnée si disponibles
            for i in range(1, 29):
                with cols[(i-1) % 4]:
                    features[f"V{i}"] = st.number_input(
                        f"V{i}", 
                        value=float(prefill_data.get(f"V{i}", 0.0)), 
                        format="%.6f"
                    )
            
            features["Amount"] = amount
            
            submitted = st.form_submit_button("🔍 Analyser")
            
            if submitted:
                try:
                    response = requests.post(
                        f"{PREDICT_ENDPOINT}?model_name={selected_model}",
                        json=features
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        prediction = result["prediction"]
                        probability = result["probability_fraud"]
                        
                        # Ajouter à l'historique
                        transaction = {
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'amount': amount,
                            'prediction': prediction,
                            'probability': probability,
                            'model': selected_model
                        }
                        st.session_state.transaction_history.append(transaction)
                        
                        # Effacer la transaction sélectionnée après analyse
                        if "selected_transaction" in st.session_state:
                            st.session_state.selected_transaction = None
                        
                        # Afficher le résultat
                        if prediction == 1:
                            st.error(f"⚠️ Transaction Frauduleuse! (Probabilité: {probability:.2%})")
                        else:
                            st.success(f"✅ Transaction Légitime (Probabilité de fraude: {probability:.2%})")
                    else:
                        st.error(f"❌ Erreur: {response.text}")
                except Exception as e:
                    logger.error(f"Erreur lors de l'analyse: {str(e)}")
                    st.error("❌ Une erreur est survenue lors de l'analyse")
    
    with tab2:
        st.header("📋 Exemples de Transactions")
        
        if st.session_state.transaction_examples.empty:
            st.info("ℹ️ Aucun exemple de transaction disponible. Importez un fichier CSV d'exemples.")
            if st.button("📥 Générer des exemples"):
                try:
                    from generate_transactions_csv import generate_transaction_data
                    df = generate_transaction_data(num_samples=50)
                    df.to_csv(TRANSACTIONS_CSV, index=False)
                    st.session_state.transaction_examples = df
                    st.success("✅ Exemples générés avec succès!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Erreur lors de la génération des exemples: {str(e)}")
        else:
            # Filtrage
            col1, col2 = st.columns(2)
            with col1:
                filter_type = st.selectbox(
                    "🔍 Filtrer par type",
                    ["Tous", "Frauduleux", "Légitimes"]
                )
            with col2:
                filter_amount = st.slider(
                    "💰 Montant maximum",
                    0.0,
                    st.session_state.transaction_examples["Amount"].max(),
                    st.session_state.transaction_examples["Amount"].max(),
                    step=50.0
                )
            
            # Appliquer les filtres
            filtered_df = st.session_state.transaction_examples.copy()
            if filter_type == "Frauduleux":
                filtered_df = filtered_df[filtered_df["is_fraud"] == 1]
            elif filter_type == "Légitimes":
                filtered_df = filtered_df[filtered_df["is_fraud"] == 0]
            filtered_df = filtered_df[filtered_df["Amount"] <= filter_amount]
            
            # Afficher les exemples
            st.write(f"Nombre d'exemples correspondants: {len(filtered_df)}")
            
            if not filtered_df.empty:
                # Sélecteur de transaction
                st.subheader("🔍 Sélectionner une transaction")
                transaction_options = [f"{row['transaction_id']} - {'⚠️ Frauduleuse' if row['is_fraud'] == 1 else '✅ Légitime'} - {row['Amount']:.2f}€ - {row['description']}" for _, row in filtered_df.iterrows()]
                selected_transaction_str = st.selectbox("Choisir une transaction à analyser", transaction_options)
                
                # Extraire l'ID de transaction
                selected_tx_id = selected_transaction_str.split(" - ")[0]
                
                # Bouton pour charger la transaction
                if st.button("📝 Charger cette transaction dans le formulaire d'analyse"):
                    selected_row = filtered_df[filtered_df["transaction_id"] == selected_tx_id].iloc[0]
                    
                    features = {}
                    for v in range(1, 29):
                        features[f"V{v}"] = float(selected_row[f"V{v}"])
                    features["Amount"] = float(selected_row["Amount"])
                    
                    st.session_state.selected_transaction = features
                    st.session_state.active_tab = 0
                    st.rerun()
                
                # Afficher les cartes de transaction
                st.subheader("📋 Détails des transactions")
                for i, row in filtered_df.head(10).iterrows():
                    fraud_status = "⚠️ Frauduleuse" if row["is_fraud"] == 1 else "✅ Légitime"
                    
                    expander = st.expander(f"{fraud_status} - {row['transaction_id']} - {row['description']}")
                    with expander:
                        st.write(f"**Montant:** {row['Amount']:.2f}€")
                        st.write(f"**Date:** {row['timestamp']}")
                        
                        if st.button(f"🔍 Analyser cette transaction", key=f"analyze_{i}"):
                            features = {}
                            for v in range(1, 29):
                                features[f"V{v}"] = float(row[f"V{v}"])
                            features["Amount"] = float(row["Amount"])
                            
                            st.session_state.selected_transaction = features
                            st.session_state.active_tab = 0
                            st.rerun()
                
                # Pagination
                if len(filtered_df) > 10:
                    st.info("ℹ️ Seuls les 10 premiers exemples sont affichés.")
                
                # Option d'export
                st.download_button(
                    "📥 Télécharger les exemples filtrés",
                    filtered_df.to_csv(index=False),
                    "transactions_filtrees.csv",
                    "text/csv"
                )
    
    with tab3:
        st.header("📈 Statistiques")
        if st.session_state.transaction_history:
            df = pd.DataFrame(st.session_state.transaction_history)
            
            # Distribution des montants
            fig1 = px.histogram(
                df, 
                x='amount',
                title='Distribution des Montants',
                labels={'amount': 'Montant', 'count': 'Nombre'}
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            # Répartition fraude/légitime
            fig2 = px.pie(
                df,
                names=df['prediction'].map({0: 'Légitime', 1: 'Frauduleuse'}),
                title='Répartition des Transactions'
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("ℹ️ Aucune transaction analysée")
    
    with tab4:
        st.header("📋 Historique des Transactions")
        if st.session_state.transaction_history:
            df = pd.DataFrame(st.session_state.transaction_history)
            df['Type'] = df['prediction'].map({0: '✅ Légitime', 1: '⚠️ Frauduleuse'})
            
            st.dataframe(
                df[['timestamp', 'amount', 'Type', 'probability', 'model']].rename(columns={
                    'timestamp': 'Date/Heure',
                    'amount': 'Montant',
                    'probability': 'Probabilité de fraude',
                    'model': 'Modèle'
                }),
                use_container_width=True
            )
            
            # Bouton de téléchargement
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Télécharger l'historique",
                csv,
                "historique_transactions.csv",
                "text/csv"
            )
        else:
            st.info("ℹ️ Aucune transaction dans l'historique")

if __name__ == "__main__":
    main()