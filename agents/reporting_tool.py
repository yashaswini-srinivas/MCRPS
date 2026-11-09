import os
import sys
import json
from datetime import datetime

# Ensure parent directory imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config


class ReportingTool:
    """Tool that compiles and exports MCRPS clinical decision-support reports."""
    def __init__(self):
        self.output_dir = os.path.join(config.BASE_DIR, "outputs")
        os.makedirs(self.output_dir, exist_ok=True)

    def export_patient_report(self, patient_id: str, care_plan: dict) -> str:
        """Exports a formatted JSON report for clinical record integration."""
        print(f"[Reporting Tool] Compiling Clinical Decision Support Report for Patient: {patient_id}...")

        report = {
            "metadata": {
                "patient_id": patient_id,
                "timestamp": datetime.now().isoformat(),
                "system_version": "MCRPS-v1.0"
            },
            "clinical_assessment": care_plan
        }

        file_path = os.path.join(self.output_dir, f"patient_report_{patient_id}.json")
        with open(file_path, "w") as f:
            json.dump(report, f, indent=4)

        print(f"[Reporting Tool] Report successfully exported to: {file_path}")
        return file_path


if __name__ == "__main__":
    print("Testing Reporting Tool...")
    reporting_tool = ReportingTool()
    # Test dummy clinical plan output
    sample_plan = {
        "neuropathy_risk_score": 0.78,
        "retinopathy_grade": "Moderate DR",
        "foot_ulcer_detected": True,
        "key_driving_factors": ["HbA1c", "Age"],
        "clinical_action_items": [
            "High Neuropathy Risk: Initiate daily foot inspection protocol.",
            "Retinopathy Detected (Moderate DR): Immediate dilated eye exam recommended."
        ],
        "specialist_referrals": ["Urgent Referral to Ophthalmologist"]
    }

    reporting_tool.export_patient_report("P_TEST_101", sample_plan)