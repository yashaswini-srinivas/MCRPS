import os
import sys
import joblib

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


# ============================================================
# Ensure parent directory imports work
# ============================================================

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import config


class ModelingTool:
    """
    Deterministic MCRPS modeling tool.

    Responsibilities:
    1. Train XGBoost, LightGBM and Random Forest models.
    2. Train models for a specified complication target.
    3. Save trained model artifacts.
    4. Build the ResNet50 vision architecture.

    This component is NOT an agent.
    The Orchestrator Agent does not perform model training
    autonomously during patient inference.
    """

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self):

        self.models_dir = os.path.join(
            config.BASE_DIR,
            "models"
        )

        os.makedirs(
            self.models_dir,
            exist_ok=True
        )

    # ========================================================
    # TABULAR MODEL TRAINING
    # ========================================================

    def train_tabular_models(
        self,
        X_train,
        y_train,
        target_name: str
    ):
        """
        Train XGBoost, LightGBM and Random Forest models
        for one complication target.

        Example:

            target_name = "neuropathy"

        Generated files:

            models/
            ├── neuropathy_xgboost_model.pkl
            ├── neuropathy_lightgbm_model.pkl
            └── neuropathy_rf_model.pkl

        Parameters
        ----------
        X_train:
            Preprocessed training features.

        y_train:
            Target labels.

        target_name:
            Name of the complication being modeled.

        Returns
        -------
        dict
            Dictionary containing trained models.
        """

        if X_train is None:
            raise ValueError(
                "X_train cannot be None."
            )

        if y_train is None:
            raise ValueError(
                "y_train cannot be None."
            )

        if not target_name:
            raise ValueError(
                "target_name must be provided."
            )

        print(
            f"\n[Modeling Tool] "
            f"Training models for: {target_name}"
        )

        # ----------------------------------------------------
        # Define models
        # ----------------------------------------------------

        models = {

            "xgboost": XGBClassifier(
                n_estimators=100,
                max_depth=4,
                random_state=42,
                eval_metric="logloss"
            ),

            "lightgbm": LGBMClassifier(
                n_estimators=100,
                max_depth=4,
                random_state=42,
                verbose=-1
            ),

            "rf": RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
        }

        trained_models = {}

        # ----------------------------------------------------
        # Train each model
        # ----------------------------------------------------

        for model_name, model in models.items():

            print(
                f"[Modeling Tool] "
                f"Training {model_name} "
                f"for {target_name}..."
            )

            model.fit(
                X_train,
                y_train
            )

            trained_models[
                model_name
            ] = model

            # ------------------------------------------------
            # Save model
            # ------------------------------------------------

            model_filename = (
                f"{target_name}_"
                f"{model_name}_model.pkl"
            )

            model_path = os.path.join(
                self.models_dir,
                model_filename
            )

            joblib.dump(
                model,
                model_path
            )

            print(
                f"[Modeling Tool] "
                f"Saved: {model_path}"
            )

        print(
            f"[Modeling Tool] "
            f"Completed training for {target_name}."
        )

        return trained_models

    # ========================================================
    # TRAIN ALL TABULAR COMPLICATION MODELS
    # ========================================================

    def train_all_tabular_models(
        self,
        df,
        preprocessing_tool,
        target_columns
    ):
        """
        Train models for multiple complication targets.

        Parameters
        ----------
        df:
            Complete clinical dataset.

        preprocessing_tool:
            Instance of DataPreprocessingTool.

        target_columns:
            Dictionary such as:

            {
                "neuropathy": "label_neuropathy",
                "nephropathy": "label_nephropathy",
                "retinopathy": "label_retinopathy",
                "foot_ulcer": "label_foot_ulcer",
                "cardiovascular": "label_cvd"
            }

        Returns
        -------
        dict
            Trained models organized by complication.
        """

        all_models = {}

        for target_name, target_column in (
            target_columns.items()
        ):

            print(
                "\n"
                + "=" * 60
            )

            print(
                f"[Modeling Tool] "
                f"Processing target: {target_name}"
            )

            print(
                f"[Modeling Tool] "
                f"Label column: {target_column}"
            )

            print(
                "=" * 60
            )

            # ------------------------------------------------
            # Preprocess this specific target
            # ------------------------------------------------

            (
                X_train,
                y_train
            ) = preprocessing_tool.preprocess_tabular(
                df,
                target_col=target_column
            )

            # ------------------------------------------------
            # Train models
            # ------------------------------------------------

            trained_models = (
                self.train_tabular_models(
                    X_train=X_train,
                    y_train=y_train,
                    target_name=target_name
                )
            )

            all_models[target_name] = {
                "models": trained_models,
            }

        return all_models

    # ========================================================
    # RESNET50 MODEL
    # ========================================================

    def build_resnet_model(
        self,
        num_classes: int = 5
    ) -> Model:
        """
        Build the ResNet50 vision architecture.

        The ImageNet backbone is frozen and a custom
        classification head is added.

        NOTE:
        This method builds the architecture.
        It does NOT train the model or save trained weights.
        """

        print(
            "[Modeling Tool] "
            "Initializing ResNet50..."
        )

        base_model = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3)
        )

        base_model.trainable = False

        x = base_model.output

        x = GlobalAveragePooling2D()(x)

        x = Dense(
            128,
            activation="relu"
        )(x)

        outputs = Dense(
            num_classes,
            activation=(
                "softmax"
                if num_classes > 1
                else "sigmoid"
            )
        )(x)

        model = Model(
            inputs=base_model.input,
            outputs=outputs
        )

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        print(
            "[Modeling Tool] "
            "ResNet50 initialized."
        )

        return model


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        + "=" * 60
    )

    print(
        "Testing Modeling Tool"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # Import tools
    # --------------------------------------------------------

    from agents.ingestion_tool import (
        DataIngestionTool
    )

    from agents.preprocessing_tool import (
        DataPreprocessingTool
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    ingestion = DataIngestionTool()

    preprocessor = DataPreprocessingTool()

    modeling_tool = ModelingTool()

    df = ingestion.load_tabular_data()

    # --------------------------------------------------------
    # Test one target
    # --------------------------------------------------------

    target_column = "label_neuropathy"

    (
        X_train,
        X_test,
        y_train,
        y_test,
        features
    ) = preprocessor.preprocess_tabular(
        df,
        target_col=target_column
    )

    trained_models = (
        modeling_tool.train_tabular_models(
            X_train=X_train,
            y_train=y_train,
            target_name="neuropathy"
        )
    )

    print(
        "\n[Modeling Tool] "
        "Tabular model test completed."
    )

    # --------------------------------------------------------
    # Test ResNet architecture
    # --------------------------------------------------------

    resnet_model = (
        modeling_tool.build_resnet_model(
            num_classes=5
        )
    )

    print(
        "\n[Modeling Tool] "
        "ResNet50 architecture test completed."
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "Modeling Tool Test Completed"
    )

    print(
        "=" * 60
    )