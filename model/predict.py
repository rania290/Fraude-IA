import joblib
import numpy as np
import pandas as pd
import os

AVAILABLE_MODELS = {
    "LogisticRegression": "logisticregression.pkl",
    "LinearSVC": "linearsvc.pkl",
    "KNN": "knn.pkl"
}

def load_model_and_scaler(model_name: str, model_dir="model"):
    """Load the specified trained model and the scaler."""
    
    # Validate model name
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Modèle inconnu: {model_name}. Modèles disponibles: {list(AVAILABLE_MODELS.keys())}")
    
    model_filename = AVAILABLE_MODELS[model_name]
    model_path = os.path.join(model_dir, model_filename)
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    # Check if files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Fichier modèle introuvable: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Fichier scaler introuvable: {scaler_path}")

    # Load model and scaler
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement du modèle ou du scaler: {e}")

def predict_transaction(data: dict, model_name: str):
    """
    Make a prediction for a single transaction using the specified model.
    
    Args:
        data (dict): Dictionary containing transaction features (V1-V28, Amount).
        model_name (str): Name of the model to use (e.g., "LogisticRegression").
        
    Returns:
        dict: Prediction results including class and probabilities (if available).
    """
    try:
        # Load model and scaler
        model, scaler = load_model_and_scaler(model_name)
        
        # Ensure all expected features are present
        expected_features = [f'V{i}' for i in range(1, 29)] + ['Amount']
        if not all(feature in data for feature in expected_features):
            missing = [f for f in expected_features if f not in data]
            raise ValueError(f"Caractéristiques manquantes: {', '.join(missing)}")
            
        # Convert input data to DataFrame in the correct order
        df = pd.DataFrame([data])[expected_features]
        
        # Scale the features
        scaled_data = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(scaled_data)[0]
        
        # Get probabilities (if the model supports predict_proba)
        probabilities = [0.0, 0.0]  # Default if no probabilities
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(scaled_data)[0]
        
        return {
            "prediction": int(prediction),
            "probability_fraud": float(probabilities[1]),
            "probability_normal": float(probabilities[0]),
            "model_used": model_name
        }
        
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        return {"error": str(e)}
    except Exception as e:
        # Catch unexpected errors
        return {"error": f"Erreur inattendue: {str(e)}"}
