from agents.ingestion_tool import DataIngestionTool
from agents.preprocessing_tool import DataPreprocessingTool

# Step 1: Preprocess training data (this generates the .pkl artifacts)
ingestion = DataIngestionTool()
df = ingestion.load_tabular_data()

preprocessing = DataPreprocessingTool()
X_scaled, y = preprocessing.preprocess_tabular(df, target_col="label_neuropathy")
print("✓ Preprocessing complete - artifacts saved")