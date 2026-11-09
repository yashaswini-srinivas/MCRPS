import os
import sys
import pandas as pd

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import config

from agents.preprocessing_tool import DataPreprocessingTool
from agents.modeling_tool import ModelingTool


# ============================================================
# TARGET CONFIGURATION
# ============================================================

TARGET_COLUMNS = {
    "nephropathy": "label_nephropathy",
    "cardiovascular": "label_cvd",
    "neuropathy": "label_neuropathy",
    "foot_ulcer": "label_foot_ulcer",
    "retinopathy": "label_retinopathy"
}


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("MCRPS MODEL TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\n[1] Loading NHANES dataset...")

    df = pd.read_csv(
        config.CSV_DATA_PATH
    )

    print(
        f"[Training] Dataset shape: {df.shape}"
    )

    print(
        "[Training] Available columns:"
    )

    print(
        df.columns.tolist()
    )

    # --------------------------------------------------------
    # Initialize tools
    # --------------------------------------------------------

    preprocessing_tool = (
        DataPreprocessingTool()
    )

    modeling_tool = (
        ModelingTool()
    )

    # --------------------------------------------------------
    # Train models for every complication
    # --------------------------------------------------------

    print("\n[2] Starting model training...")

    modeling_tool.train_all_tabular_models(
        df=df,
        preprocessing_tool=preprocessing_tool,
        target_columns=TARGET_COLUMNS
    )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MCRPS MODEL TRAINING COMPLETED")
    print("=" * 70)

    print("\nModels should now be available in:")

    print(
        os.path.join(
            config.BASE_DIR,
            "models"
        )
    )


if __name__ == "__main__":

    main()