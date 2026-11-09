class RecommendationTool:
    """Tool that maps patient risk scores to guideline-based clinical recommendations."""
    def __init__(self):
        pass

    def generate_clinical_recommendations(self, risks: dict) -> dict:
        """
        Generates ADA guideline-based recommendations structured by complication.
        """
        recs = {}
        
        # Neuropathy Interventions
        if risks.get("Neuropathy", 0) >= 0.50:
            recs["neuropathy"] = {
                "priority": "URGENT" if risks["Neuropathy"] >= 0.75 else "HIGH",
                "recommendations": [
                    "Initiate daily foot inspection protocol and 10g monofilament testing.",
                    "Re-evaluate glycemic target (< 7.0%) and adjust anti-hyperglycemic regimens.",
                    "Referral to Neurologist and Podiatrist for NCV testing."
                ]
            }
        else:
            recs["neuropathy"] = {
                "priority": "LOW",
                "recommendations": ["Routine annual foot examination and sensory testing."]
            }
            
        # Nephropathy Interventions
        if risks.get("Nephropathy", 0) >= 0.50:
            recs["nephropathy"] = {
                "priority": "HIGH",
                "recommendations": [
                    "Evaluate urine albumin-to-creatinine ratio (ACR) every 3-6 months.",
                    "Consider initiating SGLT2 inhibitor or ACEi/ARB therapy for renal protection."
                ]
            }
        else:
            recs["nephropathy"] = {
                "priority": "LOW",
                "recommendations": ["Annual serum creatinine and eGFR monitoring."]
            }

        # Retinopathy Interventions
        if risks.get("Retinopathy", 0) >= 0.50:
            recs["retinopathy"] = {
                "priority": "HIGH",
                "recommendations": [
                    "Urgent referral to Ophthalmology for dilated fundus examination.",
                    "Screen for diabetic macular edema and evaluate for anti-VEGF therapy."
                ]
            }

        # CVD Interventions
        if risks.get("Cardiovascular", 0) >= 0.50:
            recs["cvd"] = {
                "priority": "HIGH",
                "recommendations": [
                    "Optimize lipid profile (Target LDL < 70 mg/dL) with high-intensity statin therapy.",
                    "Target blood pressure threshold < 130/80 mmHg."
                ]
            }

        return recs