from typing import TypedDict, Annotated, List, Dict, Any
import operator

class AgentState(TypedDict):
    """Tracks state parameters across all 6 execution agents."""
    patient_id: str
    patient_data: Dict[str, Any]
    completeness_score: float
    imputation_applied: bool
    prediction_results: Dict[str, float]
    shap_explanations: Dict[str, Any]
    clinical_recommendations: Dict[str, Any]
    messages: Annotated[List[Dict[str, Any]], operator.add]