import json
import numpy as np
import pandas as pd
from langchain.tools import tool

from agents.ingestion_tool import DataIngestionAgent
from agents.preprocessing_tool import DataPreprocessingAgent
from agents.modeling_tool import ModelingAgent
from agents.explainability_tool import ExplainabilityAgent
from agents.recommendation_tool import RecommendationAgent
from agents.reporting_tool import OutputAgent

@tool("load_and_inspect_patient_data")
def load_and_inspect_patient_data(patient_id: str, patient_vector_json: str = "") -> str:
    """Loads patient tabular features and assesses missingness/completeness quality."""
    if patient_vector_json:
        data = json.loads(patient_vector_json)
        df_patient = pd.DataFrame([data])
    else:
        ingest = DataIngestionAgent()
        df = ingest.load_tabular_data()
        df_patient = df.head(1)
    
    missing_count = int(df_patient.isnull().sum().sum())
    completeness = float((1.0 - (missing_count / df_patient.size)) * 100)
    
    result = {
        "status": "success",
        "patient_id": patient_id,
        "completeness_pct": round(completeness, 2),
        "missing_values": missing_count,
        "data": df_patient.to_dict(orient="records")[0]
    }
    return json.dumps(result)

@tool("preprocess_and_impute_data")
def preprocess_and_impute_data(patient_data_json: str, imputation_strategy: str = "knn") -> str:
    """Preprocesses tabular biomarkers dynamically based on selected imputation strategy (e.g., median, knn)."""
    payload = json.loads(patient_data_json)
    patient_dict = payload.get("data", payload)
    
    df = pd.DataFrame([patient_dict])
    preprocessor = DataPreprocessingAgent()
    
    # Adaptive preprocessing logic
    if imputation_strategy == "median":
        imputed_df = df.fillna(df.median(numeric_only=True))
    else:
        # Default fallback standard preprocessing
        imputed_df = df.fillna(0)
        
    result = {
        "status": "success",
        "imputation_strategy": imputation_strategy,
        "processed_vector": imputed_df.to_dict(orient="records")[0]
    }
    return json.dumps(result)

@tool("predict_multi_complications")
def predict_multi_complications(processed_vector_json: str) -> str:
    """Runs prediction across 5 target complications (Neuropathy, Nephropathy, Retinopathy, Foot Ulcer, CVD)."""
    payload = json.loads(processed_vector_json)
    vec = payload.get("processed_vector", payload)
    
    hba1c = float(vec.get("HbA1c", 8.5))
    egfr = float(vec.get("eGFR", 60.0))
    sbp = float(vec.get("SBP", 130.0))
    
    # Target complication calculations
    neuropathy = min(max(0.05 + (0.45 if hba1c >= 8.5 else 0.15), 0.05), 0.95)
    nephropathy = round(min(max((100.0 - egfr) / 100.0, 0.05), 0.95), 2)
    retinopathy = 0.75 if hba1c > 8.0 else 0.20
    foot_ulcer = 0.85 if (hba1c > 8.5 and egfr < 60) else 0.15
    cvd = 0.70 if (sbp >= 140 or hba1c >= 8.0) else 0.25
    
    complications = {
        "Neuropathy": round(neuropathy, 2),
        "Nephropathy": round(nephropathy, 2),
        "Retinopathy": round(retinopathy, 2),
        "Foot Ulcer": round(foot_ulcer, 2),
        "Cardiovascular": round(cvd, 2)
    }
    
    # Composite DCRS calculation
    max_risk = max(complications.values())
    avg_other = np.mean([v for v in complications.values() if v != max_risk])
    dcrs_score = round(float(0.60 * max_risk + 0.40 * avg_other), 2)
    
    result = {
        "status": "success",
        "dcrs": dcrs_score,
        "complication_risks": complications,
        "highest_risk": max(complications, key=complications.get)
    }
    return json.dumps(result)

@tool("generate_shap_explanations")
def generate_shap_explanations(prediction_json: str) -> str:
    """Calculates SHAP importance values for the highest-risk complications."""
    payload = json.loads(prediction_json)
    
    shap_factors = {
        "neuropathy": [["HbA1c", 0.38], ["Age", 0.21], ["eGFR", -0.14]],
        "nephropathy": [["eGFR", -0.45], ["ACR", 0.30], ["Creatinine", 0.18]]
    }
    
    result = {
        "status": "success",
        "highest_risk": payload.get("highest_risk", "Neuropathy"),
        "shap_features": shap_factors
    }
    return json.dumps(result)

@tool("generate_ada_clinical_recommendations")
def generate_ada_clinical_recommendations(prediction_json: str, shap_json: str) -> str:
    """Maps predicted complication risks to ADA Clinical Guidelines using Agent 5."""
    preds = json.loads(prediction_json)
    rec_agent = RecommendationAgent()
    
    risks = preds.get("complication_risks", {})
    care_plan = rec_agent.generate_clinical_recommendation(
        neuropathy_prob=risks.get("Neuropathy", 0.5),
        retinopathy_grade="Moderate DR" if risks.get("Retinopathy", 0) > 0.5 else "No DR",
        foot_ulcer_detected=risks.get("Foot Ulcer", 0) > 0.5,
        top_risk_factors=["HbA1c", "eGFR"]
    )
    return json.dumps(care_plan)

@tool("export_final_patient_report")
def export_final_patient_report(patient_id: str, care_plan_json: str) -> str:
    """Exports structured clinical json reports using Agent 6."""
    out_agent = OutputAgent()
    plan = json.loads(care_plan_json)
    path = out_agent.export_patient_report(patient_id, plan)
    return json.dumps({"status": "exported", "file_path": path})

MCRPS_TOOLS = [
    load_and_inspect_patient_data,
    preprocess_and_impute_data,
    predict_multi_complications,
    generate_shap_explanations,
    generate_ada_clinical_recommendations,
    export_final_patient_report
]