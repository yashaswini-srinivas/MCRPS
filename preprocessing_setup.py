from agents.ingestion_tool import DataIngestionTool
from agents.preprocessing_tool import DataPreprocessingTool

print("Starting preprocessing...")

ingestion = DataIngestionTool()
df = ingestion.load_tabular_data()

preprocessing = DataPreprocessingTool()

targets = ["label_neuropathy", "label_cvd", "label_retinopathy", "label_foot_ulcer", "label_nephropathy"]

for target in targets:
    try:
        X_scaled, y = preprocessing.preprocess_tabular(df, target_col=target)
        print(f"✓ {target} preprocessed successfully")
    except Exception as e:
        print(f"✗ {target} error: {e}")

print("\n✓ All preprocessing complete!")
print("Check models/ folder for .pkl files")