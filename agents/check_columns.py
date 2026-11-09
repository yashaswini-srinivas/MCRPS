from agents.ingestion_tool import DataIngestionTool
import pandas as pd

ingestion = DataIngestionTool()
df = ingestion.load_tabular_data()

print("=" * 60)
print("ACTUAL COLUMN NAMES IN CSV:")
print("=" * 60)
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. '{col}'")

print("\n" + "=" * 60)
print("LOOKING FOR THESE PATTERNS:")
print("=" * 60)
print(f"SerumCreatinine columns: {[col for col in df.columns if 'serumcreatinine' in col.lower()]}")
print(f"Triglycerides columns: {[col for col in df.columns if 'triglycerides' in col.lower()]}")