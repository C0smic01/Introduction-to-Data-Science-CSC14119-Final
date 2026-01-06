# Data Exploration

**Purpose**

This folder contains exploratory data analysis (EDA) notebooks and scripts used to understand and visualize the football players dataset. It is focused on cleaning, visualizing distributions, and generating insights to guide feature engineering and modeling.

**Contents**

- `DataExploration.ipynb` - main EDA notebook with descriptive statistics, missing value analysis, and visualizations.
- `MeaningfulQuestion.ipynb` - notebook for forming and investigating meaningful analysis questions.
- `input/` - data inputs (e.g. `football_players_dataset.csv`).

**Requirements & Installation**

Use the included `requirements.txt` to install dependencies:

```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

**How to run**

Launch Jupyter and open the notebooks:

```bash
jupyter notebook
# or
jupyter lab
```

Open `DataExploration.ipynb` and run cells interactively to reproduce the analysis.

**Notes**

- The notebooks are designed for interactive exploration; ensure `input/football_players_dataset.csv` is present.
- For reproducibility, use the pinned dependency versions in `requirements.txt`.
