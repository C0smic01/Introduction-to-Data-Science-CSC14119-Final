# Football Player Market Value Predictor - Streamlit UI

## Overview
Interactive web application for predicting football player market values using Machine Learning models.

## Features
- **3 ML Models**: Random Forest, XGBoost, LightGBM
- **Model Selection**: Choose any model via sidebar
- **4 Prediction Modes**:
  1. Upload CSV for batch prediction
  2. Select player from database
  3. Manual input with key features
  4. Compare all 3 models side-by-side

## Quick Start

### Run the application:
```bash
cd "source/Data Modeling"
streamlit run streamlit-gui.py
```

The app will open in your browser at `http://localhost:8501`

## File Structure
```
source/Data Modeling/
├── streamlit-gui.py              # Main Streamlit application
├── pkl/
│   ├── RF_final_model.pkl        # Random Forest model
│   ├── XGB_final_model.pkl       # XGBoost model
│   ├── LGB_final_model.pkl       # LightGBM model
│   └── selected_features.pkl     # Feature list (62 features)
├── football_players_dataset.csv  # Player database
└── README.md                     # This file
```

## Usage Guide

### 1. Model Selection (Sidebar)
- Choose between Random Forest, XGBoost, or LightGBM
- View model description and statistics

### 2. Tab 1: Upload CSV File
- Download CSV template
- Upload file with player statistics
- Get batch predictions for all players
- Download results

### 3. Tab 2: Select from Database
- Filter by league, club, position
- Select any player from database
- Compare predicted vs actual market value

### 4. Tab 3: Manual Input
- Enter 10 key features manually
- Get instant prediction
- See player tier classification

### 5. Tab 4: Model Comparison
- Compare all 3 models on same player
- View ensemble average prediction
- See individual model differences
- Get accuracy metrics (if actual value available)

## Key Features Used
The models use 62 features including:
- Basic stats: age, appearances, minutes_played, goals, assists
- Per-90 metrics: goals_per_90, shots_per90, progressive_passes_per90
- Advanced stats: xG, xA, SCA, GCA, key passes
- Defensive: interceptions, blocks, tackles
- Engineered features: log transforms, ratios, interactions

## Model Performance
All models predict log-transformed market values and convert back to EUR (millions).

## Requirements
- Python 3.8+
- streamlit
- pandas
- numpy
- joblib
- scikit-learn (for model loading)
- xgboost
- lightgbm

## Notes
- Predictions are in millions of Euros (€M)
- The ensemble average (Tab 4) typically provides the most robust prediction
- Feature engineering is applied automatically to all inputs
