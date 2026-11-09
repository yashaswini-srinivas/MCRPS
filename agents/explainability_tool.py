import os
import sys
import warnings

# Python 3.12 Matplotlib & Pydantic warning compatibility patch
_orig_warn = warnings.warn
def _safe_warn(message, category=None, stacklevel=1, source=None, **kwargs):
    try:
        _orig_warn(message, category=category, stacklevel=stacklevel, source=source)
    except TypeError:
        _orig_warn(message)
warnings.warn = _safe_warn

import numpy as np
import shap


class ExplainabilityTool:
    """
    Deterministic Explainable AI tool for MCRPS.

    Responsibilities:
    1. Calculate SHAP feature attributions for trained
       tree-based ML models.
    2. Rank features according to absolute SHAP impact.
    3. Return the most influential features for a patient.

    This component is NOT an agent.
    The Orchestrator Agent decides when this tool
    should be invoked.
    """

    def __init__(self):
        pass

    # ========================================================
    # TREE MODEL SHAP
    # ========================================================

    def calculate_shap_values(
        self,
        model,
        patient_vector,
        feature_names
    ):
        """
        Calculate SHAP feature contributions for one patient.

        Parameters
        ----------
        model:
            Trained tree-based model such as
            XGBoost, LightGBM or Random Forest.

        patient_vector:
            Patient feature vector AFTER applying the same
            preprocessing used during model training.

        feature_names:
            Feature names in exactly the same order as
            the trained model.

        Returns
        -------
        list:
            Top five features ranked by absolute SHAP impact.
        """

        # ----------------------------------------------------
        # Validate input
        # ----------------------------------------------------

        if model is None:
            raise ValueError(
                "A trained model is required for SHAP analysis."
            )

        if patient_vector is None:
            raise ValueError(
                "Patient vector cannot be None."
            )

        if feature_names is None:
            raise ValueError(
                "Feature names cannot be None."
            )

        # ----------------------------------------------------
        # Convert patient data to numpy
        # ----------------------------------------------------

        patient_array = np.asarray(
            patient_vector,
            dtype=float
        ).reshape(1, -1)

        # ----------------------------------------------------
        # Validate feature count
        # ----------------------------------------------------

        if patient_array.shape[1] != len(feature_names):

            raise ValueError(
                "Number of patient features does not match "
                "number of feature names. "
                f"Patient features: {patient_array.shape[1]}, "
                f"Feature names: {len(feature_names)}"
            )

        # ----------------------------------------------------
        # Create TreeExplainer
        # ----------------------------------------------------

        explainer = shap.TreeExplainer(
            model
        )

        # ----------------------------------------------------
        # Calculate SHAP values
        # ----------------------------------------------------

        shap_values = explainer.shap_values(
            patient_array
        )

        # ----------------------------------------------------
        # Normalize SHAP output format
        # ----------------------------------------------------

        values = self._extract_positive_class_values(
            shap_values
        )

        # ----------------------------------------------------
        # Validate SHAP output
        # ----------------------------------------------------

        if len(values) != len(feature_names):

            raise ValueError(
                "SHAP output size does not match "
                "the number of feature names."
            )

        # ----------------------------------------------------
        # Rank features by absolute contribution
        # ----------------------------------------------------

        ranked_indices = np.argsort(
            np.abs(values)
        )[::-1]

        # ----------------------------------------------------
        # Keep top five
        # ----------------------------------------------------

        top_k = min(
            5,
            len(feature_names)
        )

        top_features = []

        for index in ranked_indices[:top_k]:

            shap_value = float(
                values[index]
            )

            top_features.append(
                {
                    "feature":
                        feature_names[index],

                    "shap_value":
                        round(
                            shap_value,
                            4
                        ),

                    "absolute_impact":
                        round(
                            abs(shap_value),
                            4
                        ),

                    "direction":
                        (
                            "increases_risk"
                            if shap_value > 0
                            else
                            "decreases_risk"
                            if shap_value < 0
                            else
                            "neutral"
                        )
                }
            )

        return top_features

    # ========================================================
    # SHAP OUTPUT HANDLER
    # ========================================================

    def _extract_positive_class_values(
        self,
        shap_values
    ):
        """
        Normalize different SHAP output formats.

        Supports common binary and multiclass
        TreeExplainer outputs.
        """

        # ----------------------------------------------------
        # Older SHAP versions may return a list
        # ----------------------------------------------------

        if isinstance(
            shap_values,
            list
        ):

            if len(shap_values) == 0:

                raise ValueError(
                    "SHAP returned an empty result."
                )

            # Binary classification:
            # class 1 = positive/risk class
            values = np.asarray(
                shap_values[-1]
            )

            if values.ndim == 2:

                return values[0]

            return values

        # ----------------------------------------------------
        # Newer SHAP versions may return numpy arrays
        # ----------------------------------------------------

        values = np.asarray(
            shap_values
        )

        # ----------------------------------------------------
        # Standard binary output:
        # (samples, features)
        # ----------------------------------------------------

        if values.ndim == 2:

            return values[0]

        # ----------------------------------------------------
        # Possible multiclass output:
        # (samples, features, classes)
        # ----------------------------------------------------

        if values.ndim == 3:

            return values[0, :, -1]

        # ----------------------------------------------------
        # Unexpected format
        # ----------------------------------------------------

        raise ValueError(
            "Unsupported SHAP output shape: "
            f"{values.shape}"
        )

    # ========================================================
    # SIMPLE SUMMARY
    # ========================================================

    def summarize_shap(
        self,
        shap_results
    ):
        """
        Generate a simple human-readable summary from
        SHAP results.

        This is deterministic formatting only.
        """

        if not shap_results:

            return "No SHAP explanations available."

        summary = []

        for item in shap_results:

            feature = item["feature"]
            shap_value = item["shap_value"]
            direction = item["direction"]

            if direction == "increases_risk":

                text = (
                    f"{feature} contributed "
                    f"toward higher predicted risk "
                    f"(SHAP={shap_value})."
                )

            elif direction == "decreases_risk":

                text = (
                    f"{feature} contributed "
                    f"toward lower predicted risk "
                    f"(SHAP={shap_value})."
                )

            else:

                text = (
                    f"{feature} had approximately "
                    f"neutral contribution "
                    f"(SHAP={shap_value})."
                )

            summary.append(
                text
            )

        return summary