from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import joblib
import numpy as np
from typing import Dict, List, Optional
import os
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FraudGuard API",
    description="API pour la détection de fraude bancaire",
    version="1.0.0"
)

# Modèles disponibles
AVAILABLE_MODELS = ["LogisticRegression", "LinearSVC", "KNN"]

# Mapping des noms de modèles vers les fichiers
MODEL_FILES = {
    "LogisticRegression": "logisticregression.pkl",
    "LinearSVC": "linearsvc.pkl",
    "KNN": "knn.pkl"
}

class TransactionData(BaseModel):
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

def load_model(model_name: str):
    """Charge le modèle spécifié et son scaler."""
    try:
        if model_name not in MODEL_FILES:
            raise HTTPException(status_code=400, detail=f"Modèle {model_name} non disponible")
        
        model_file = MODEL_FILES[model_name]
        model_path = os.path.join('model', model_file)
        scaler_path = os.path.join('model', 'scaler.pkl')
        
        if not os.path.exists(model_path):
            logger.error(f"Fichier modèle non trouvé: {model_path}")
            raise HTTPException(status_code=404, detail=f"Modèle {model_name} non trouvé")
        if not os.path.exists(scaler_path):
            logger.error(f"Fichier scaler non trouvé: {scaler_path}")
            raise HTTPException(status_code=404, detail="Scaler non trouvé")
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors du chargement du modèle: {str(e)}")

@app.get("/health")
def health_check():
    """Vérifie l'état de santé de l'API."""
    return {"status": "ok"}

@app.get("/models")
def get_models():
    """Renvoie la liste des modèles disponibles."""
    return {"models": AVAILABLE_MODELS}

@app.post("/predict")
def predict(
    data: TransactionData,
    model_name: str = Query(
        "LogisticRegression",
        description="Nom du modèle à utiliser (LogisticRegression, LinearSVC, ou KNN)"
    )
):
    """Prédit si une transaction est frauduleuse."""
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Modèle {model_name} non disponible")
    
    try:
        # Charger le modèle et le scaler
        model, scaler = load_model(model_name)
        
        # Préparer les données
        features = [
            data.V1, data.V2, data.V3, data.V4, data.V5, data.V6, data.V7, data.V8,
            data.V9, data.V10, data.V11, data.V12, data.V13, data.V14, data.V15, data.V16,
            data.V17, data.V18, data.V19, data.V20, data.V21, data.V22, data.V23, data.V24,
            data.V25, data.V26, data.V27, data.V28, data.Amount
        ]
        
        # Mettre en forme et normaliser les données
        X = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)
        
        # Faire la prédiction
        prediction = int(model.predict(X_scaled)[0])
        
        # Obtenir la probabilité si disponible
        probability = 0.0
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(X_scaled)[0][1])
        
        return {
            "prediction": prediction,
            "probability_fraud": probability,
            "model_used": model_name
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)