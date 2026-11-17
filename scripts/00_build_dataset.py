""" Build the processed datasets from the raw NASA .mat files.
Reads : data/raw/B0005.mat, B0006.mat, B0007.mat, B0018.mat
Writes: data/processed/*.csv and outputs/eda/matlab_structure.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.battery_pipeline import run_dataset_pipeline

def main():
    run_dataset_pipeline()
if __name__ == "__main__":
    main()
