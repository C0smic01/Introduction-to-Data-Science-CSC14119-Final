# Data Pre-processing

**Purpose**

This folder is dedicated to cleaning, merging, and transforming raw data into a consolidated dataset ready for modeling. It contains scripts and notebooks that standardize columns, handle missing values, and produce the final dataset used in `Data Modeling`.

**Contents**

- `DataPreProcessing.ipynb` - main notebook for pre-processing steps and producing `football_players_full_dataset.csv`.
- `input/` - raw CSVs for different leagues.
- `output/` - processed datasets and intermediate artifacts.

**Requirements & Installation**

Install the dependencies in this folder with:

```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

**How to run**

- Open `DataPreProcessing.ipynb` in Jupyter and run cells sequentially to reproduce the pre-processing pipeline.
- Use the `input/` data files as the source; the notebook or scripts will write the consolidated dataset to `output/`.

**Notes**

- The pre-processing notebook includes common steps: parsing, type conversion, filling missing values, feature scaling, and saving the final dataset.
- Keep a copy of `football_players_full_dataset.csv` (or similar) for modeling reproducibility.
