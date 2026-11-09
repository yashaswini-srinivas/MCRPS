import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd

# Suppress sklearn/LightGBM feature-name mismatch warning (cosmetic only)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config


class PredictionTool:
    """Deterministic MCRPS prediction tool."""

    COMPLICATIONS = {
        "Neuropathy": "neuropathy",
        "Nephropathy": "nephropathy",
        "Retinopathy": "retinopathy",
        "Foot Ulcer": "foot_ulcer",
        "Cardiovascular": "cvd"
    }

    MODEL_TYPES = ["xgboost", "lightgbm", "rf"]

    def __init__(self):
        self.models_dir = os.path.join(config.BASE_DIR, "models")
        self.models = {}
        self.preprocessing = {}
        print("[Prediction Tool] Initializing...")
        self._load_models()

    def _load_models(self):
        for complication, target_name in self.COMPLICATIONS.items():
            self.models[complication] = {}
            for model_type in self.MODEL_TYPES:
                filename = f"{target_name}_{model_type}_model.pkl"
                model_path = os.path.join(self.models_dir, filename)

                if not os.path.exists(model_path):
                    continue

                try:
                    model = joblib.load(model_path)
                    self.models[complication][model_type] = model
                except Exception as e:
                    print(f"[Prediction Tool] Failed to load {filename}: {e}")

    def _find_artifact_path(self, target_name: str, artifact_type: str) -> str:
        """Helper to safely find artifact files regardless of prefix naming."""
        paths_to_check = [
            os.path.join(self.models_dir, f"{target_name}_{artifact_type}.pkl"),
            os.path.join(self.models_dir, f"label_{target_name}_{artifact_type}.pkl"),
        ]
        for p in paths_to_check:
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"Preprocessing artifact '{artifact_type}' for '{target_name}' not found.")

    def _load_preprocessing(self, target_name: str):
        if target_name in self.preprocessing:
            return self.preprocessing[target_name]

        imputer_path = self._find_artifact_path(target_name, "imputer")
        scaler_path = self._find_artifact_path(target_name, "scaler")
        features_path = self._find_artifact_path(target_name, "features")

        imputer = joblib.load(imputer_path)
        scaler = joblib.load(scaler_path)
        feature_names = joblib.load(features_path)

        self.preprocessing[target_name] = {
            "imputer": imputer,
            "scaler": scaler,
            "feature_names": feature_names
        }

        return self.preprocessing[target_name]

    def prepare_patient_data(self, patient_vector):
        """Convert patient vector to DataFrame with standard feature names."""
        feature_names = [
            "Age",
            "Sex",
            "Race",
            "Diabetes",
            "HbA1c",
            "SerumCreatinine",
            "ACR",
            "SBP",
            "DBP",
            "LDL",
            "Triglycerides",
            "HDL",
            "BMI",
            "FootUlcer",
            "CRP",
            "eGFR"
        ]

        if patient_vector is None:
            raise ValueError("Patient vector cannot be None.")

        if len(patient_vector) != len(feature_names):
            raise ValueError(
                f"Expected {len(feature_names)} features, received {len(patient_vector)}."
            )

        clean_vector = []
        for val in patient_vector:
            if val is None or (isinstance(val, str) and val.strip() == ""):
                clean_vector.append(0.0)
            else:
                clean_vector.append(float(val))

        patient_array = np.asarray(clean_vector, dtype=float).reshape(1, -1)
        return pd.DataFrame(patient_array, columns=feature_names)

    def preprocess_patient(self, patient_vector, target_name):
        patient_df = self.prepare_patient_data(patient_vector)
        preprocessing = self._load_preprocessing(target_name)
        feature_names = preprocessing["feature_names"]

        if "Creatinine" in feature_names and "SerumCreatinine" in patient_df.columns:
            patient_df = patient_df.rename(columns={"SerumCreatinine": "Creatinine"})

        X_patient = patient_df[feature_names]
        X_imputed = preprocessing["imputer"].transform(X_patient)
        X_scaled = preprocessing["scaler"].transform(X_imputed)

        return X_scaled

    def _predict_model(self, model, X_patient):
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_patient)
            probability = float(probabilities[0][1]) if probabilities.shape[1] >= 2 else float(probabilities[0][0])
        else:
            prediction = model.predict(X_patient)
            probability = float(np.asarray(prediction).ravel()[0])

        return max(0.0, min(1.0, probability))

    def predict_single(self, complication, patient_vector):
        if complication not in self.COMPLICATIONS:
            raise ValueError(f"Unknown complication: {complication}")

        target_name = self.COMPLICATIONS[complication]
        models = self.models.get(complication, {})

        if not models:
            raise FileNotFoundError(f"No trained models available for {complication}.")

        X_patient = self.preprocess_patient(patient_vector, target_name)
        individual_predictions = {}

        for model_type in self.MODEL_TYPES:
            if model_type not in models:
                continue

            probability = self._predict_model(models[model_type], X_patient)
            individual_predictions[model_type] = probability

        if not individual_predictions:
            raise RuntimeError(f"No usable models available for {complication}.")

        ensemble_probability = sum(individual_predictions.values()) / len(individual_predictions)

        return {
            "probability": round(ensemble_probability, 4),
            "model_predictions": {
                name: round(val, 4) for name, val in individual_predictions.items()
            }
        }

    def predict_all(self, patient_vector):
        predictions = {}
        for complication in self.COMPLICATIONS:
            try:
                result = self.predict_single(complication, patient_vector)
                predictions[complication] = result
            except Exception as e:
                predictions[complication] = {
                    "probability": None,
                    "model_predictions": {},
                    "error": str(e)
                }
        return predictions

    def calculate_risk_summary(self, predictions):
        valid_predictions = {
            comp: res["probability"]
            for comp, res in predictions.items()
            if res.get("probability") is not None
        }

        if not valid_predictions:
            return {
                "dcrs": 0.0,
                "risk_level": "UNKNOWN",
                "highest_risk_complication": "Unknown",
                "highest_risk_score": 0.0
            }

        highest_complication = max(valid_predictions, key=valid_predictions.get)
        highest_risk = valid_predictions[highest_complication]

        valid_vals = [v for v in valid_predictions.values() if v is not None]
        average_risk = sum(valid_vals) / len(valid_vals) if valid_vals else 0.0

        dcrs = round(0.6 * highest_risk + 0.4 * average_risk, 4) if highest_risk is not None else 0.0

        if dcrs is None:
            risk_level = "UNKNOWN"
            dcrs = 0.0
        elif dcrs >= 0.55:
            risk_level = "HIGH"
        elif dcrs >= 0.35:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        # Corrected Indentation: Unnested return statement
        return {
            "dcrs": dcrs,
            "risk_level": risk_level,
            "highest_risk_complication": highest_complication,
            "highest_risk_score": round(highest_risk, 4) if highest_risk is not None else 0.0
        }

    def predict(self, patient_vector):
        predictions = self.predict_all(patient_vector)
        summary = self.calculate_risk_summary(predictions)

        complication_risks = {
            comp: res["probability"] for comp, res in predictions.items()
        }

        return {
            "complication_risks": complication_risks,
            "model_details": predictions,
            "dcrs": summary["dcrs"],
            "risk_level": summary["risk_level"],
            "highest_risk_complication": summary["highest_risk_complication"],
            "highest_risk_score": summary["highest_risk_score"]
        }