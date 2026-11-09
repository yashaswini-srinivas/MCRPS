import os
import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class DataPreprocessingTool:
    """
    Deterministic preprocessing tool for MCRPS.

    The same explicit feature schema is used during:
        - model training
        - patient prediction
        - SHAP explanation
    """

    FEATURE_COLS = [
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
        "eGFR",
    ]

    def __init__(self, models_dir="models"):
        self.models_dir = models_dir

        os.makedirs(
            self.models_dir,
            exist_ok=True
        )

    # ---------------------------------------------------------------
    # TRAINING
    # ---------------------------------------------------------------
    def preprocess_tabular(
        self,
        df: pd.DataFrame,
        target_col: str
    ):
        """
        Fit imputer and scaler using training data only.

        Returns:
            X_train_processed
            y
        """

        missing_features = [
            col
            for col in self.FEATURE_COLS
            if col not in df.columns
        ]

        if missing_features:
            raise ValueError(
                "Dataset is missing required features: "
                f"{missing_features}"
            )

        if target_col not in df.columns:
            raise ValueError(
                f"Target column '{target_col}' "
                "does not exist in the dataset."
            )

        X = df[self.FEATURE_COLS].copy()
        y = df[target_col].copy()

        # Convert all features to numeric
        X = X.apply(
            pd.to_numeric,
            errors="coerce"
        )

        # -----------------------------------------------------------
        # Imputation
        # -----------------------------------------------------------
        imputer = SimpleImputer(
            strategy="median"
        )

        X_imputed = imputer.fit_transform(X)

        # -----------------------------------------------------------
        # Scaling
        # -----------------------------------------------------------
        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(
            X_imputed
        )

        # -----------------------------------------------------------
        # Save preprocessing artifacts
        # -----------------------------------------------------------
        joblib.dump(
            imputer,
            os.path.join(
                self.models_dir,
                f"{target_col}_imputer.pkl"
            )
        )

        joblib.dump(
            scaler,
            os.path.join(
                self.models_dir,
                f"{target_col}_scaler.pkl"
            )
        )

        joblib.dump(
            self.FEATURE_COLS,
            os.path.join(
                self.models_dir,
                f"{target_col}_features.pkl"
            )
        )

        print(
            f"[Preprocessing] Saved artifacts for "
            f"{target_col}"
        )

        return X_scaled, y

    # ---------------------------------------------------------------
    # LOAD ARTIFACTS
    # ---------------------------------------------------------------
    def load_preprocessing_artifacts(
        self,
        target_col: str
    ):
        """
        Load the imputer, scaler and feature schema
        belonging to a target.
        """

        imputer_path = os.path.join(
            self.models_dir,
            f"{target_col}_imputer.pkl"
        )

        scaler_path = os.path.join(
            self.models_dir,
            f"{target_col}_scaler.pkl"
        )

        features_path = os.path.join(
            self.models_dir,
            f"{target_col}_features.pkl"
        )

        if not os.path.exists(imputer_path):
            raise FileNotFoundError(
                f"Preprocessing imputer not found: "
                f"{imputer_path}"
            )

        if not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Preprocessing scaler not found: "
                f"{scaler_path}"
            )

        if not os.path.exists(features_path):
            raise FileNotFoundError(
                f"Feature schema not found: "
                f"{features_path}"
            )

        imputer = joblib.load(
            imputer_path
        )

        scaler = joblib.load(
            scaler_path
        )

        feature_cols = joblib.load(
            features_path
        )

        return (
            imputer,
            scaler,
            feature_cols
        )

    # ---------------------------------------------------------------
    # PATIENT TRANSFORMATION
    # ---------------------------------------------------------------
    def transform_patient(
        self,
        patient_data,
        target_col: str
    ):
        """
        Transform one patient's clinical data
        using the exact preprocessing artifacts
        fitted during training.
        """

        imputer, scaler, feature_cols = (
            self.load_preprocessing_artifacts(
                target_col
            )
        )

        # -----------------------------------------------------------
        # Convert input to DataFrame
        # -----------------------------------------------------------
        if isinstance(
            patient_data,
            pd.DataFrame
        ):
            patient_df = patient_data.copy()

        elif isinstance(
            patient_data,
            dict
        ):
            patient_df = pd.DataFrame(
                [patient_data]
            )

        elif isinstance(
            patient_data,
            (list, tuple, np.ndarray)
        ):
            if len(patient_data) != len(
                feature_cols
            ):
                raise ValueError(
                    f"Expected "
                    f"{len(feature_cols)} features, "
                    f"received "
                    f"{len(patient_data)}."
                )

            patient_df = pd.DataFrame(
                [patient_data],
                columns=feature_cols
            )

        else:
            raise TypeError(
                "patient_data must be a DataFrame, "
                "dictionary, list, tuple or numpy array."
            )

        # -----------------------------------------------------------
        # Validate columns
        # -----------------------------------------------------------
        missing_features = [
            col
            for col in feature_cols
            if col not in patient_df.columns
        ]

        if missing_features:
            raise ValueError(
                "Patient data is missing required "
                f"features: {missing_features}"
            )

        patient_df = patient_df[
            feature_cols
        ].copy()

        # -----------------------------------------------------------
        # Numeric conversion
        # -----------------------------------------------------------
        patient_df = patient_df.apply(
            pd.to_numeric,
            errors="coerce"
        )

        # -----------------------------------------------------------
        # Transform
        # -----------------------------------------------------------
        X_imputed = imputer.transform(
            patient_df
        )

        X_scaled = scaler.transform(
            X_imputed
        )

        return X_scaled