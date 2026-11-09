from crewai.tools import tool

# Corrected imports pointing to agents/
from agents.ingestion_tool import DataIngestionTool
from agents.preprocessing_tool import DataPreprocessingTool
from agents.prediction_tool import PredictionTool
from agents.explainability_tool import ExplainabilityTool
from agents.recommendation_tool import RecommendationTool
from agents.reporting_tool import ReportingTool

ingestion = DataIngestionTool()
preprocessing = DataPreprocessingTool()
prediction = PredictionTool()
explainability = ExplainabilityTool()
recommendation = RecommendationTool()
reporting = ReportingTool()

@tool("ingestion_tool")
def ingestion_tool() -> str:
    """Load and validate the MCRPS patient/tabular dataset."""
    try:
        df = ingestion.load_tabular_data()
        return f"Data ingestion successful. Rows: {df.shape[0]}, Columns: {df.shape[1]}."
    except Exception as e:
        return f"INGESTION_ERROR: {str(e)}"

@tool("preprocessing_tool")
def preprocessing_tool(target_col: str = "label_neuropathy") -> str:
    """Preprocess MCRPS tabular data using imputation and scaling."""
    try:
        df = ingestion.load_tabular_data()
        X_train, y_train = preprocessing.preprocess_tabular(df, target_col=target_col)
        return f"Preprocessing successful for target '{target_col}'. Train shape: {X_train.shape}."
    except Exception as e:
        return f"PREPROCESSING_ERROR: {str(e)}"

@tool("prediction_tool")
def prediction_tool(patient_vector: str) -> str:
    """Predict complication risks for a patient using the trained MCRPS models."""
    try:
        result = prediction.predict(patient_vector)
        return str(result)
    except Exception as e:
        return f"PREDICTION_ERROR: {str(e)}"

@tool("explainability_tool")
def explainability_tool(patient_vector: str, feature_names: str) -> str:
    """Generate explainability information for a patient's prediction using SHAP."""
    try:
        result = explainability.calculate_shap_contributions(patient_vector=patient_vector, feature_names=feature_names)
        return str(result)
    except Exception as e:
        return f"EXPLAINABILITY_ERROR: {str(e)}"

@tool("recommendation_tool")
def recommendation_tool(risks: str) -> str:
    """Generate guideline-based clinical recommendations from predicted complication risks."""
    try:
        result = recommendation.generate_clinical_recommendations(risks=risks)
        return str(result)
    except Exception as e:
        return f"RECOMMENDATION_ERROR: {str(e)}"

@tool("reporting_tool")
def reporting_tool(patient_id: str, care_plan: str) -> str:
    """Export the final MCRPS clinical decision-support report as a JSON file."""
    try:
        file_path = reporting.export_patient_report(patient_id=patient_id, care_plan=care_plan)
        return f"REPORT_SUCCESS: {file_path}"
    except Exception as e:
        return f"REPORTING_ERROR: {str(e)}"

MCRPS_TOOLS = [
    ingestion_tool,
    preprocessing_tool,
    prediction_tool,
    explainability_tool,
    recommendation_tool,
    reporting_tool,
]