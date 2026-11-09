import os

# Define project base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset paths configured from your system
CSV_DATA_PATH = r"C:\Users\devik\Desktop\dataset_csv\nhanes_combined.csv"
RETINO_DATA_DIR = r"C:\Users\devik\Desktop\dataset_images\kaggle_retino"
DFU_DATA_DIR = r"C:\Users\devik\Desktop\dataset_images\kaggle_foot\DFU"

def verify_paths():
    """Diagnostic check to verify data folder access on Windows."""
    paths = {
        "Tabular CSV Path": CSV_DATA_PATH,
        "Retinopathy Image Folder": RETINO_DATA_DIR,
        "Foot Ulcer Image Folder": DFU_DATA_DIR,
    }
    
    print("--- PATH DIAGNOSTIC CHECK ---")
    all_valid = True
    for key, path in paths.items():
        exists = os.path.exists(path)
        status = "FOUND" if exists else "NOT FOUND"
        print(f"[{status}] {key}: {path}")
        if not exists:
            all_valid = False
    return all_valid

if __name__ == "__main__":
    verify_paths()