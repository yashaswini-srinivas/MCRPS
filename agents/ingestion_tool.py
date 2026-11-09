import os
import pandas as pd
import numpy as np
from PIL import Image
import sys

# Ensure config can be imported from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config


def load_img(path, target_size=None):
    """Replacement for tf.keras.preprocessing.image.load_img using PIL."""
    img = Image.open(path)
    if target_size:
        img = img.resize(target_size)
    return img


def img_to_array(img):
    """Replacement for tf.keras.preprocessing.image.img_to_array using NumPy."""
    return np.asarray(img, dtype=np.float32)


class DataIngestionTool:
    """Agent 1: Ingests tabular data and image files from local Windows directories."""

    def __init__(self):
        self.csv_path = config.CSV_DATA_PATH
        self.retino_dir = config.RETINO_DATA_DIR
        self.dfu_dir = config.DFU_DATA_DIR

    def load_tabular_data(self) -> pd.DataFrame:
        """Loads the NHANES combined tabular CSV dataset."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV path not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        print(f"[Ingestion Tool] Ingested Tabular Data successfully...")
        return df

    def load_single_image(self, image_path: str, target_size=(224, 224)) -> np.ndarray:
        """Loads and resizes a single image file into a numpy array."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image path not found: {image_path}")

        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        print(f"[Ingestion Tool] Loaded image '{os.path.basename(image_path)}' with shape: {img_array.shape}")
        return img_array

    def inspect_directory(self, dir_path: str) -> dict:
        """Scans image directory and counts supported image formats."""
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        summary = {"total_files": 0, "subfolders": []}

        if not os.path.exists(dir_path):
            print(f"[Ingestion Tool Warning] Directory not found: {dir_path}")
            return summary

        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    summary["total_files"] += 1

        summary["subfolders"] = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
        print(f"[Ingestion Tool] Inspected '{dir_path}': Found {summary['total_files']} images across {len(summary['subfolders'])} subfolders.")
        return summary