"""
MCRPS Legacy Pipeline Compatibility Layer

This module preserves the previous run_inference() interface
for backward compatibility.

The Streamlit application should NOT use this function as the
main execution path.

The actual application workflow is now handled by:

    app.py
        ↓
    Orchestrator Agent
        ↓
    MCRPS Tools
"""

from typing import Optional

from agents.prediction_tool import PredictionTool
from agents.explainability_tool import ExplainabilityTool
from agents.recommendation_tool import RecommendationTool
from agents.reporting_tool import ReportingTool


FEATURE_COLS = [
    "Age",
    "Sex",
    "Race",
    "HbA1c",
    "Creatinine",
    "ACR",
    "SBP",
    "DBP",
    "LDL",
    "Triglycerides",
    "HDL",
    "BMI",
    "CRP",
    "eGFR"
]


def run_inference(
    patient_csv_row: list,
    patient_id: str = "P_93714",
    retinal_img_path: Optional[str] = None,
    foot_img_path: Optional[str] = None
):
    """
    Backward-compatible wrapper for the previous MCRPS pipeline.

    IMPORTANT:
        The Streamlit application should use the new
        run_agentic_mcrps() entry point instead.

    This wrapper simply executes the deterministic MCRPS
    tools directly for compatibility with older code.
    """

    # ---------------------------------------------------------
    # 1. Prediction
    # ---------------------------------------------------------

    prediction_tool = PredictionTool()

    prediction_result = prediction_tool.predict(
        patient_vector=patient_csv_row,
        retinal_img_path=retinal_img_path,
        foot_img_path=foot_img_path
    )


    # ---------------------------------------------------------
    # 2. Explainability
    # ---------------------------------------------------------

    explainability_tool = ExplainabilityTool()

    explanation_result = (
        explainability_tool.calculate_shap_contributions(
            patient_vector=patient_csv_row,
            feature_names=FEATURE_COLS
        )
    )


    # ---------------------------------------------------------
    # 3. Clinical Recommendations
    # ---------------------------------------------------------

    recommendation_tool = RecommendationTool()

    care_plan = (
        recommendation_tool.generate_clinical_recommendations(
            prediction_result["complication_risks"]
        )
    )


    # ---------------------------------------------------------
    # 4. Final Report
    # ---------------------------------------------------------

    report = {
        "dcrs": prediction_result["dcrs"],
        "risk_level": prediction_result["risk_level"],
        "highest_risk_complication": (
            prediction_result["highest_risk_complication"]
        ),
        "highest_risk_score": (
            prediction_result["highest_risk_score"]
        ),
        "complication_risks": (
            prediction_result["complication_risks"]
        ),
        "shap_top5_features": {
            "neuropathy": explanation_result
        },
        "clinical_recommendations": care_plan
    }


    # ---------------------------------------------------------
    # 5. Reporting
    # ---------------------------------------------------------

    reporting_tool = ReportingTool()

    report_path = reporting_tool.export_patient_report(
        patient_id=patient_id,
        care_plan=report
    )

    report["report_path"] = report_path

    return report