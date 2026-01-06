"""
Football Player Market Value Predictor - Streamlit GUI
========================================================
File: app.py
Run: streamlit run app.py

Features:
- Tab 1: Upload CSV file for batch prediction
- Tab 2: Select player from existing dataset
- Tab 3: Manual input (key features only)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
# ============================================================
# CONFIGURATION
# ============================================================
# Model paths for all 3 models
MODELS_CONFIG = {
    'Random Forest': 'pkl/RF_final_model.pkl',
    'XGBoost': 'pkl/XGB_final_model.pkl',
    'LightGBM': 'pkl/LGB_final_model.pkl'
}
FEATURES_PATH = "pkl/selected_features.pkl"
DATASET_PATH = "football_players_dataset.csv"  # Original dataset

# Top 10 most important features (from feature importance analysis)
KEY_FEATURES = [
    'age', 'minutes_played', 'appearances',
    'goals', 'assists', 'goals_per_90',
    'progressive_carries_per90', 'progressive_passes_per90',
    'passes_completed_per90', 'sca_per90'
]

# ============================================================
# LOAD RESOURCES
# ============================================================
@st.cache_resource
def load_all_models():
    """Load all 3 trained models."""
    models = {}
    for model_name, model_path in MODELS_CONFIG.items():
        try:
            models[model_name] = joblib.load(model_path)
        except Exception as e:
            st.warning(f"Failed to load {model_name}: {str(e)}")
            models[model_name] = None
    return models

@st.cache_resource
def load_features(_models):
    """Load required features list from the first available model."""
    # Try to get features from model itself (most reliable)
    for model_name, model in _models.items():
        if model is not None and hasattr(model, 'feature_names_in_'):
            return list(model.feature_names_in_)

    # Fallback to pkl file
    try:
        return joblib.load(FEATURES_PATH)
    except:
        return None

@st.cache_data
def load_dataset():
    """Load original dataset for player selection."""
    try:
        df = pd.read_csv(DATASET_PATH)
        return df
    except:
        return None

# ============================================================
# FEATURE ENGINEERING (same as training)
# ============================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering pipeline."""
    df = df.copy()
    
    # 1. Log transformations
    for col in ['minutes_played', 'goals', 'assists', 'appearances']:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col])
    
    # 2. Performance ratios
    if 'goals_per_90' in df.columns and 'shots_per90' in df.columns:
        df['conversion_rate'] = df['goals_per_90'] / df['shots_per90'].replace(0, 0.01)
    
    if 'key_passes_per90' in df.columns and 'passes_completed_per90' in df.columns:
        df['key_pass_ratio'] = df['key_passes_per90'] / df['passes_completed_per90'].replace(0, 0.01)
    
    if 'interceptions_per90' in df.columns and 'blocks_per90' in df.columns:
        df['defensive_contribution'] = df['interceptions_per90'] + df['blocks_per90']
    
    if 'progressive_passes_per90' in df.columns and 'progressive_carries_per90' in df.columns:
        df['total_progressive'] = df['progressive_passes_per90'] + df['progressive_carries_per90']
    
    # 3. Interaction features
    if 'age' in df.columns and 'minutes_played' in df.columns:
        df['age_experience'] = df['age'] * np.log1p(df['minutes_played'])
    
    if 'minutes_played' in df.columns and 'appearances' in df.columns:
        df['minutes_per_game'] = df['minutes_played'] / df['appearances'].replace(0, 1)
    
    # 4. Polynomial features
    for col in ['goals', 'assists', 'minutes_played']:
        if col in df.columns:
            df[f'{col}_squared'] = df[col] ** 2
    
    # 5. Categorical encoding (frequency) - skip if columns don't exist
    for col in ['nationality', 'current_club', 'league', 'position']:
        if col in df.columns:
            freq = df[col].map(df[col].value_counts())
            df[f'{col}_freq'] = freq.fillna(1)
        else:
            # Create default frequency encoding if column doesn't exist
            df[f'{col}_freq'] = 1

    # 6. League label encoding
    if 'league' in df.columns:
        league_map = {league: i for i, league in enumerate(df['league'].unique())}
        df['league_label_enc'] = df['league'].map(league_map).fillna(0)
    else:
        df['league_label_enc'] = 0
    
    return df

def prepare_for_prediction(df: pd.DataFrame, required_features: list,
                           training_data: pd.DataFrame = None) -> pd.DataFrame:
    """Prepare data for model prediction."""

    df = df.copy()

    # Calculate target encoding from training data if available
    nationality_target_mean = 1.5  # Default value
    club_target_mean = 1.5  # Default value

    if training_data is not None and 'market_value' in training_data.columns:
        # Calculate target encoding maps from training data
        if 'nationality' in training_data.columns and 'nationality' in df.columns:
            nationality_map = training_data.groupby('nationality')['market_value'].mean().to_dict()
            df['nationality_target_enc'] = df['nationality'].map(nationality_map).fillna(nationality_target_mean)
        elif 'nationality' not in df.columns:
            df['nationality_target_enc'] = nationality_target_mean

        if 'current_club' in training_data.columns and 'current_club' in df.columns:
            club_map = training_data.groupby('current_club')['market_value'].mean().to_dict()
            df['current_club_target_enc'] = df['current_club'].map(club_map).fillna(club_target_mean)
        elif 'current_club' not in df.columns:
            df['current_club_target_enc'] = club_target_mean
    else:
        # No training data, use defaults
        if 'nationality_target_enc' not in df.columns:
            df['nationality_target_enc'] = nationality_target_mean
        if 'current_club_target_enc' not in df.columns:
            df['current_club_target_enc'] = club_target_mean

    # Set default values for categorical columns if not present (for frequency encoding)
    if 'nationality' not in df.columns:
        df['nationality'] = 'Unknown'
    if 'current_club' not in df.columns:
        df['current_club'] = 'Unknown'
    if 'league' not in df.columns:
        df['league'] = 'Unknown'
    if 'position' not in df.columns:
        df['position'] = 'Unknown'

    # Apply feature engineering
    df_eng = engineer_features(df)

    # Ensure all required features exist with default values
    for feat in required_features:
        if feat not in df_eng.columns:
            # Set reasonable defaults based on feature name
            if feat == 'nationality_target_enc':
                df_eng[feat] = nationality_target_mean
            elif feat == 'current_club_target_enc':
                df_eng[feat] = club_target_mean
            elif 'freq' in feat:
                df_eng[feat] = 1  # Frequency default
            elif 'label_enc' in feat:
                df_eng[feat] = 0  # Encoding default
            else:
                df_eng[feat] = 0  # Numeric default

    # Select only required features in the correct order
    df_final = df_eng[required_features].copy()

    # Fill NaN with appropriate defaults
    df_final = df_final.fillna(0)

    return df_final

def predict_values(model, X: pd.DataFrame) -> np.ndarray:
    """Make predictions and convert from log scale."""
    log_preds = model.predict(X)
    predictions = np.expm1(log_preds)
    return np.maximum(0, predictions)  # Ensure non-negative

# ============================================================
# STREAMLIT UI
# ============================================================
def main():
    # Page config
    st.set_page_config(
        page_title="Player Value Predictor",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("Football Player Market Value Predictor")
    st.markdown("Predict player market value using Machine Learning")
    st.markdown("---")

    # Load resources
    models = load_all_models()
    required_features = load_features(models)
    dataset = load_dataset()

    # Check if resources loaded
    if required_features is None:
        st.error("Features list not found! Please ensure 'pkl/selected_features.pkl' exists.")
        st.stop()

    # Count successfully loaded models
    loaded_models = {name: model for name, model in models.items() if model is not None}

    if len(loaded_models) == 0:
        st.error("No models found! Please ensure model files exist in 'pkl/' folder.")
        st.info("Run the training notebooks first to generate the model files.")
        st.stop()

    # Sidebar - Model Selection
    st.sidebar.header("Model Selection")
    selected_model_name = st.sidebar.selectbox(
        "Choose a prediction model:",
        options=list(loaded_models.keys()),
        help="Select which machine learning model to use for predictions"
    )
    selected_model = loaded_models[selected_model_name]

    # Model info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Info")
    model_descriptions = {
        'Random Forest': "Ensemble of decision trees. Robust and interpretable.",
        'XGBoost': "Gradient boosting. High performance and accuracy.",
        'LightGBM': "Fast gradient boosting. Efficient for large datasets."
    }
    st.sidebar.info(model_descriptions.get(selected_model_name, ""))

    # Success message
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success(f"{len(loaded_models)}/{len(MODELS_CONFIG)} models loaded")
    with col2:
        st.info(f"{len(required_features)} features")
    with col3:
        if dataset is not None:
            st.info(f"{len(dataset):,} players in database")

    st.info(f"**Selected Model:** {selected_model_name}")
    st.markdown("---")

    # ========================================
    # TABS
    # ========================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "Upload CSV File",
        "Select from Database",
        "Manual Input",
        "Model Comparison"
    ])
    
    # ========================================
    # TAB 1: UPLOAD CSV
    # ========================================
    with tab1:
        st.header("Batch Prediction from CSV")
        st.markdown("Upload a CSV file with player statistics to predict market values.")
        
        # Download template
        if dataset is not None:
            template_df = dataset.head(3).drop(columns=['market_value'], errors='ignore')
            csv_template = template_df.to_csv(index=False)
            st.download_button(
                label="Download CSV Template",
                data=csv_template,
                file_name="player_template.csv",
                mime="text/csv"
            )
        
        # File uploader
        uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Read uploaded file
                upload_df = pd.read_csv(uploaded_file)
                st.write(f"Loaded {len(upload_df)} players")
                
                # Show preview
                with st.expander("Preview uploaded data"):
                    st.dataframe(upload_df.head(10))
                
                # Predict button
                if st.button("Predict Market Values", key="predict_csv"):
                    with st.spinner(f"Processing with {selected_model_name}..."):
                        # Prepare data
                        X = prepare_for_prediction(upload_df, required_features, dataset)

                        # Predict
                        predictions = predict_values(selected_model, X)

                        # Add predictions to dataframe (keep original order)
                        result_df = upload_df.copy()
                        result_df['Predicted_Value_EUR_M'] = predictions.round(2)

                        # Display results
                        st.success("Prediction complete!")

                        # Show predictions in original CSV order
                        st.subheader("Prediction Results")
                        display_cols = ['Predicted_Value_EUR_M']
                        if 'current_club' in result_df.columns:
                            display_cols = ['current_club'] + display_cols
                        if 'age' in result_df.columns:
                            display_cols = ['age'] + display_cols

                        st.dataframe(result_df[display_cols], width='stretch')
                        
                        # Statistics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Average Value", f"€{predictions.mean():.2f}M")
                        with col2:
                            st.metric("Median Value", f"€{np.median(predictions):.2f}M")
                        with col3:
                            st.metric("Max Value", f"€{predictions.max():.2f}M")
                        with col4:
                            st.metric("Min Value", f"€{predictions.min():.2f}M")
                        
                        # Download results
                        csv_result = result_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results CSV",
                            data=csv_result,
                            file_name="predictions_result.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # ========================================
    # TAB 2: SELECT FROM DATABASE
    # ========================================
    with tab2:
        st.header("Select Player from Database")
        
        if dataset is None:
            st.warning("Dataset not found. Please ensure 'input/football_players_dataset.csv' exists.")
        else:
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # League filter
                leagues = ['All'] + sorted(dataset['league'].unique().tolist())
                selected_league = st.selectbox("League", leagues)
            
            with col2:
                # Club filter
                if selected_league == 'All':
                    clubs = ['All'] + sorted(dataset['current_club'].unique().tolist())
                else:
                    clubs = ['All'] + sorted(dataset[dataset['league'] == selected_league]['current_club'].unique().tolist())
                selected_club = st.selectbox("Club", clubs)
            
            with col3:
                # Position filter
                positions = ['All', 'Defender', 'Midfielder', 'Forward']
                selected_position = st.selectbox("Position", positions)
            
            # Apply filters
            filtered_df = dataset.copy()
            if selected_league != 'All':
                filtered_df = filtered_df[filtered_df['league'] == selected_league]
            if selected_club != 'All':
                filtered_df = filtered_df[filtered_df['current_club'] == selected_club]
            if selected_position != 'All':
                pos_map = {'Defender': 'is_DF', 'Midfielder': 'is_MF', 'Forward': 'is_FW'}
                filtered_df = filtered_df[filtered_df[pos_map[selected_position]] == 1]
            
            st.write(f"Found {len(filtered_df)} players")
            
            # Player selection
            if len(filtered_df) > 0:
                # Create display name (make a copy to avoid Arrow conversion issues)
                filtered_df = filtered_df.copy()
                filtered_df['display_name'] = filtered_df['current_club'].astype(str) + " - Age " + filtered_df['age'].astype(str)

                # Select player by index
                player_idx = st.selectbox(
                    "Select Player",
                    options=filtered_df.index.tolist(),
                    format_func=lambda x: filtered_df.loc[x, 'display_name']
                )
                
                selected_player = filtered_df.loc[[player_idx]]
                
                # Show player info
                st.subheader("Player Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Age", int(selected_player['age'].values[0]))
                with col2:
                    st.metric("Appearances", int(selected_player['appearances'].values[0]))
                with col3:
                    st.metric("Goals", int(selected_player['goals'].values[0]))
                with col4:
                    st.metric("Assists", int(selected_player['assists'].values[0]))
                
                # Show more stats
                with st.expander("View All Statistics"):
                    # Convert to display-friendly format to avoid Arrow serialization issues
                    display_stats = selected_player.T.copy()
                    display_stats.columns = ['Value']
                    # Convert all values to string to avoid type conflicts
                    display_stats['Value'] = display_stats['Value'].astype(str)
                    st.dataframe(display_stats)
                
                # Actual vs Predicted
                actual_value = selected_player['market_value'].values[0]
                
                # Predict
                if st.button("Predict Market Value", key="predict_select"):
                    with st.spinner(f"Calculating with {selected_model_name}..."):
                        X = prepare_for_prediction(selected_player, required_features, dataset)
                        prediction = predict_values(selected_model, X)[0]
                        
                        st.markdown("---")
                        st.subheader("Prediction Result")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Predicted Value", f"€{prediction:.2f}M")
                        with col2:
                            st.metric("Actual Value", f"€{actual_value:.2f}M")
                        with col3:
                            diff = prediction - actual_value
                            diff_pct = (diff / actual_value) * 100 if actual_value > 0 else 0
                            st.metric("Difference", f"€{diff:.2f}M", f"{diff_pct:+.1f}%")
    
    # ========================================
    # TAB 3: MANUAL INPUT
    # ========================================
    with tab3:
        st.header("Manual Input")
        st.markdown("Enter player information for market value prediction.")

        st.info("**Important:** League and Club heavily influence predictions (Club = 58% importance!)")

        # Initialize session state for league selection if not exists
        if 'selected_manual_league' not in st.session_state:
            st.session_state['selected_manual_league'] = "Premier League" if dataset is not None else "Premier League"

        # League selection OUTSIDE form to allow dynamic club update
        col1, col2 = st.columns([1, 3])
        with col1:
            if dataset is not None:
                available_leagues = sorted(dataset['league'].unique().tolist())
                temp_league = st.selectbox("1. Select League:", available_leagues,
                                           index=available_leagues.index(st.session_state['selected_manual_league'])
                                                 if st.session_state['selected_manual_league'] in available_leagues else 0,
                                           help="Select league first, then clubs will update below",
                                           key="temp_manual_league")
                if temp_league != st.session_state['selected_manual_league']:
                    st.session_state['selected_manual_league'] = temp_league
                    st.rerun()

        st.markdown("---")

        # Create input form (with all other fields including club based on selected league)
        with st.form("manual_input_form"):
            st.subheader("2. Player & Club Information")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                position = st.selectbox("Position", ["Defender", "Midfielder", "Forward"])

            with col2:
                # Get clubs from selected league
                selected_league = st.session_state['selected_manual_league']
                if dataset is not None and selected_league:
                    league_clubs = sorted(dataset[dataset['league'] == selected_league]['current_club'].unique().tolist())
                    if len(league_clubs) > 0:
                        selected_club = st.selectbox("Current Club", league_clubs,
                                                    help=f"Clubs in {selected_league}")
                    else:
                        selected_club = "Unknown"
                else:
                    selected_club = "Unknown"

            with col3:
                # Nationality
                if dataset is not None:
                    available_nationalities = sorted(dataset['nationality'].unique().tolist())
                    nationality = st.selectbox("Nationality", available_nationalities, index=0)
                else:
                    nationality = "Unknown"

            with col4:
                age = st.number_input("Age", min_value=16, max_value=45, value=25)

            st.subheader("Performance Stats")
            col1, col2, col3 = st.columns(3)

            with col1:
                appearances = st.number_input("Appearances", min_value=0, max_value=60, value=30)
            with col2:
                minutes = st.number_input("Minutes Played", min_value=0, max_value=5000, value=2500)
            with col3:
                goals = st.number_input("Total Goals", min_value=0, max_value=50, value=5)

            col1, col2, col3 = st.columns(3)
            with col1:
                assists = st.number_input("Total Assists", min_value=0, max_value=30, value=5)
            with col2:
                goals_per_90 = st.number_input("Goals per 90 min", min_value=0.0, max_value=2.0, value=0.2, step=0.05)

            st.subheader("Advanced Stats (per 90 minutes)")
            col1, col2, col3 = st.columns(3)

            with col1:
                progressive_carries = st.number_input("Progressive Carries", min_value=0.0, max_value=15.0, value=3.0, step=0.5)
            with col2:
                progressive_passes = st.number_input("Progressive Passes", min_value=0.0, max_value=15.0, value=5.0, step=0.5)
            with col3:
                sca = st.number_input("Shot-Creating Actions", min_value=0.0, max_value=10.0, value=3.0, step=0.5)

            passes_completed = st.slider("Passes Completed per 90", min_value=10.0, max_value=100.0, value=45.0)
            
            # Submit button
            submitted = st.form_submit_button("Predict Market Value", width='stretch')
        
        if submitted:
            # Create input dictionary
            input_data = {
                'age': age,
                'nationality': nationality,
                'current_club': selected_club,
                'league': selected_league,
                'appearances': appearances,
                'minutes_played': minutes,
                'is_DF': 1 if position == "Defender" else 0,
                'is_MF': 1 if position == "Midfielder" else 0,
                'is_FW': 1 if position == "Forward" else 0,
                'goals': goals,
                'assists': assists,
                'goals_per_90': goals_per_90,
                'progressive_carries_per90': progressive_carries,
                'progressive_passes_per90': progressive_passes,
                'sca_per90': sca,
                'passes_completed_per90': passes_completed,
                # Default values for other features
                'npg_per90': goals_per_90 * 0.85,
                'xg_per90': goals_per_90 * 0.9,
                'xag_per90': assists / max(appearances, 1) * 90 / max(minutes / max(appearances, 1), 1) * 0.8,
                'shots_per90': goals_per_90 * 4,
                'shots_on_target_per90': goals_per_90 * 2,
                'shots_on_target_pct': 40.0,
                'avg_shot_distance': 16.0,
                'gca_per90': sca * 0.3,
                'key_passes_per90': sca * 0.5,
                'pass_completion_pct': 80.0,
                'passes_into_final_third_per90': passes_completed * 0.15,
                'passes_into_penalty_area_per90': passes_completed * 0.05,
                'progressive_passes_rec_per90': progressive_passes * 0.8,
                'take_ons_per90': progressive_carries * 0.5,
                'carries_into_final_third_per90': progressive_carries * 0.4,
                'touches_att_third_per90': 20.0,
                'touches_att_pen_per90': 5.0,
                'passes_received_per90': passes_completed * 0.8,
                'interceptions_per90': 1.5 if position == "Defender" else 1.0,
                'blocks_per90': 1.5 if position == "Defender" else 0.8,
                'ball_recoveries_per90': 5.0,
                'aerials_won_per90': 2.0 if position == "Defender" else 1.0,
                'yellow_cards_per90': 0.15,
                'fouls_committed_per90': 1.0,
            }

            # Create DataFrame
            input_df = pd.DataFrame([input_data])
            
            with st.spinner(f"Calculating market value with {selected_model_name}..."):
                # Prepare and predict
                X = prepare_for_prediction(input_df, required_features, dataset)
                prediction = predict_values(selected_model, X)[0]
            
            # Display result
            st.markdown("---")
            st.subheader("Predicted Market Value")
            
            # Big number display
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); border-radius: 15px; margin: 20px 0;">
                <h1 style="color: white; font-size: 48px; margin: 0;">€{prediction:.2f}M</h1>
                <p style="color: #ccc; font-size: 18px;">Estimated Market Value</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Value range interpretation
            if prediction < 1:
                tier = "Lower League / Youth Player"
            elif prediction < 5:
                tier = "Solid Squad Player"
            elif prediction < 15:
                tier = "First Team Regular"
            elif prediction < 40:
                tier = "Key Player / Star"
            elif prediction < 80:
                tier = "Elite Player"
            else:
                tier = "World Class"
            
            st.info(f"**Player Tier:** {tier}")
            
            # Show input summary
            with st.expander("Input Summary"):
                summary_df = pd.DataFrame({
                    'Feature': ['Age', 'Position', 'Appearances', 'Minutes', 'Goals', 'Assists',
                               'Goals/90', 'Progressive Carries/90', 'Progressive Passes/90', 'SCA/90'],
                    'Value': [str(age), position, str(appearances), str(minutes), str(goals), str(assists),
                             str(goals_per_90), str(progressive_carries), str(progressive_passes), str(sca)]
                })
                st.table(summary_df)

    # ========================================
    # TAB 4: MODEL COMPARISON
    # ========================================
    with tab4:
        st.header("Compare All Models")
        st.markdown("Compare predictions from all available models on the same input data.")

        if len(loaded_models) < 2:
            st.warning("Need at least 2 models to compare. Only {} model(s) loaded.".format(len(loaded_models)))
        else:
            st.info(f"Comparing {len(loaded_models)} models: {', '.join(loaded_models.keys())}")

            # ========================================
            # Comparison Input Options
            # ========================================
            comparison_method = st.radio(
                "Choose input method:",
                ["Select Player from Database", "Manual Input"],
                horizontal=True
            )

            player_data = None

            # ========================================
            # METHOD 1: SELECT FROM DATABASE
            # ========================================
            if comparison_method == "Select Player from Database":
                if dataset is None:
                    st.warning("Dataset not found.")
                else:
                    # Simplified player selection
                    col1, col2 = st.columns(2)

                    with col1:
                        leagues = ['All'] + sorted(dataset['league'].unique().tolist())
                        selected_league = st.selectbox("League", leagues, key="comp_league")

                    with col2:
                        if selected_league == 'All':
                            clubs = ['All'] + sorted(dataset['current_club'].unique().tolist())
                        else:
                            clubs = ['All'] + sorted(dataset[dataset['league'] == selected_league]['current_club'].unique().tolist())
                        selected_club = st.selectbox("Club", clubs, key="comp_club")

                    # Apply filters
                    filtered_df = dataset.copy()
                    if selected_league != 'All':
                        filtered_df = filtered_df[filtered_df['league'] == selected_league]
                    if selected_club != 'All':
                        filtered_df = filtered_df[filtered_df['current_club'] == selected_club]

                    if len(filtered_df) > 0:
                        # Create display name (make a copy to avoid Arrow conversion issues)
                        filtered_df = filtered_df.copy()
                        filtered_df['display_name'] = filtered_df['current_club'].astype(str) + " - Age " + filtered_df['age'].astype(str)

                        # Select player
                        player_idx = st.selectbox(
                            "Select Player",
                            options=filtered_df.index.tolist(),
                            format_func=lambda x: filtered_df.loc[x, 'display_name'],
                            key="comp_player"
                        )

                        player_data = filtered_df.loc[[player_idx]]

                        # Show player info
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Age", int(player_data['age'].values[0]))
                        with col2:
                            st.metric("Appearances", int(player_data['appearances'].values[0]))
                        with col3:
                            st.metric("Goals", int(player_data['goals'].values[0]))
                        with col4:
                            st.metric("Assists", int(player_data['assists'].values[0]))

            # ========================================
            # METHOD 2: MANUAL INPUT
            # ========================================
            else:
                with st.form("comparison_manual_form"):
                    st.subheader("Basic Information")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        age = st.number_input("Age", min_value=16, max_value=45, value=25, key="comp_age")
                    with col2:
                        appearances = st.number_input("Appearances", min_value=0, max_value=60, value=30, key="comp_app")
                    with col3:
                        minutes = st.number_input("Minutes Played", min_value=0, max_value=5000, value=2500, key="comp_min")

                    st.subheader("Position & Club Information")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        position = st.selectbox("Position", ["Defender", "Midfielder", "Forward"], key="comp_pos")

                    with col2:
                        if dataset is not None:
                            available_leagues = sorted(dataset['league'].unique().tolist())
                            comp_league = st.selectbox("League", available_leagues,
                                                       index=available_leagues.index("Premier League") if "Premier League" in available_leagues else 0,
                                                       key="comp_league_select")
                        else:
                            comp_league = "Premier League"

                    with col3:
                        if dataset is not None and comp_league:
                            league_clubs = sorted(dataset[dataset['league'] == comp_league]['current_club'].unique().tolist())
                            if len(league_clubs) > 0:
                                comp_club = st.selectbox("Club", league_clubs, key="comp_club_select")
                            else:
                                comp_club = "Unknown"
                        else:
                            comp_club = "Unknown"

                    with col4:
                        if dataset is not None:
                            available_nationalities = sorted(dataset['nationality'].unique().tolist())
                            comp_nationality = st.selectbox("Nationality", available_nationalities, index=0, key="comp_nationality")
                        else:
                            comp_nationality = "Unknown"

                    st.subheader("Performance Stats")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        goals = st.number_input("Goals", min_value=0, max_value=50, value=5, key="comp_goals")
                    with col2:
                        assists = st.number_input("Assists", min_value=0, max_value=30, value=5, key="comp_assists")
                    with col3:
                        goals_per_90 = st.number_input("Goals per 90", min_value=0.0, max_value=2.0, value=0.2, step=0.05, key="comp_g90")

                    submitted = st.form_submit_button("Load Data for Comparison", width='stretch')

                if submitted:
                    # Create manual input dataframe
                    input_data = {
                        'age': age,
                        'nationality': comp_nationality,
                        'current_club': comp_club,
                        'league': comp_league,
                        'appearances': appearances,
                        'minutes_played': minutes,
                        'is_DF': 1 if position == "Defender" else 0,
                        'is_MF': 1 if position == "Midfielder" else 0,
                        'is_FW': 1 if position == "Forward" else 0,
                        'goals': goals,
                        'assists': assists,
                        'goals_per_90': goals_per_90,
                        'progressive_carries_per90': 3.0,
                        'progressive_passes_per90': 5.0,
                        'sca_per90': 3.0,
                        'passes_completed_per90': 45.0,
                        'npg_per90': goals_per_90 * 0.85,
                        'xg_per90': goals_per_90 * 0.9,
                        'xag_per90': assists / max(appearances, 1) * 90 / max(minutes / max(appearances, 1), 1) * 0.8,
                        'shots_per90': goals_per_90 * 4,
                        'shots_on_target_per90': goals_per_90 * 2,
                        'shots_on_target_pct': 40.0,
                        'avg_shot_distance': 16.0,
                        'gca_per90': 1.0,
                        'key_passes_per90': 2.0,
                        'pass_completion_pct': 80.0,
                        'passes_into_final_third_per90': 7.0,
                        'passes_into_penalty_area_per90': 2.0,
                        'progressive_passes_rec_per90': 4.0,
                        'take_ons_per90': 1.5,
                        'carries_into_final_third_per90': 1.5,
                        'touches_att_third_per90': 20.0,
                        'touches_att_pen_per90': 5.0,
                        'passes_received_per90': 35.0,
                        'interceptions_per90': 1.5 if position == "Defender" else 1.0,
                        'blocks_per90': 1.5 if position == "Defender" else 0.8,
                        'ball_recoveries_per90': 5.0,
                        'aerials_won_per90': 2.0 if position == "Defender" else 1.0,
                        'yellow_cards_per90': 0.15,
                        'fouls_committed_per90': 1.0,
                    }
                    player_data = pd.DataFrame([input_data])

                    # Save to session state so it persists after form submit
                    st.session_state['comparison_player_data'] = player_data
                    st.success("Data loaded! Click 'Compare All Models' button below to see predictions.")

                # Check if we have data in session state
                if 'comparison_player_data' in st.session_state:
                    player_data = st.session_state['comparison_player_data']

                    # Show loaded data summary
                    st.markdown("---")
                    st.subheader("Loaded Player Data")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Age", int(player_data['age'].values[0]))
                    with col2:
                        st.metric("Position", "Defender" if player_data['is_DF'].values[0] == 1 else ("Midfielder" if player_data['is_MF'].values[0] == 1 else "Forward"))
                    with col3:
                        st.metric("Club", player_data['current_club'].values[0])
                    with col4:
                        st.metric("League", player_data['league'].values[0])

            # ========================================
            # COMPARISON RESULTS
            # ========================================
            if player_data is not None:
                if st.button("Compare All Models", key="compare_btn", width='stretch'):
                    with st.spinner("Running predictions on all models..."):
                        # Prepare data
                        X = prepare_for_prediction(player_data, required_features, dataset)

                        # Get predictions from all models
                        predictions = {}
                        for model_name, model in loaded_models.items():
                            pred = predict_values(model, X)[0]
                            predictions[model_name] = pred

                        # Display results
                        st.markdown("---")
                        st.subheader("Prediction Results")

                        # Show predictions in columns
                        cols = st.columns(len(predictions))
                        for idx, (model_name, pred_value) in enumerate(predictions.items()):
                            with cols[idx]:
                                st.metric(
                                    label=f"**{model_name}**",
                                    value=f"€{pred_value:.2f}M"
                                )

                        # Calculate statistics
                        pred_values = list(predictions.values())
                        avg_prediction = np.mean(pred_values)
                        min_prediction = np.min(pred_values)
                        max_prediction = np.max(pred_values)
                        std_prediction = np.std(pred_values)

                        st.markdown("---")
                        st.subheader("Ensemble Statistics")

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Average (Ensemble)", f"€{avg_prediction:.2f}M")
                        with col2:
                            st.metric("Min Prediction", f"€{min_prediction:.2f}M")
                        with col3:
                            st.metric("Max Prediction", f"€{max_prediction:.2f}M")
                        with col4:
                            st.metric("Std Deviation", f"€{std_prediction:.2f}M")

                        # Comparison table
                        st.markdown("---")
                        st.subheader("Detailed Comparison")

                        comparison_df = pd.DataFrame({
                            'Model': list(predictions.keys()),
                            'Prediction (€M)': [f"€{v:.2f}M" for v in predictions.values()],
                            'Diff from Avg': [f"{v - avg_prediction:+.2f}M" for v in predictions.values()],
                            'Diff from Avg (%)': [f"{((v - avg_prediction) / avg_prediction * 100):+.1f}%" if avg_prediction > 0 else "N/A" for v in predictions.values()]
                        })

                        st.dataframe(comparison_df, width='stretch', hide_index=True)

                        # Show actual value if available
                        if 'market_value' in player_data.columns:
                            actual_value = player_data['market_value'].values[0]
                            st.markdown("---")
                            st.subheader("Comparison with Actual Value")

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Actual Market Value", f"€{actual_value:.2f}M")
                            with col2:
                                ensemble_diff = avg_prediction - actual_value
                                ensemble_diff_pct = (ensemble_diff / actual_value) * 100 if actual_value > 0 else 0
                                st.metric("Ensemble vs Actual", f"€{ensemble_diff:.2f}M", f"{ensemble_diff_pct:+.1f}%")

                            # Model accuracy comparison
                            accuracy_df = pd.DataFrame({
                                'Model': list(predictions.keys()),
                                'Prediction': [f"€{v:.2f}M" for v in predictions.values()],
                                'Error': [f"{v - actual_value:+.2f}M" for v in predictions.values()],
                                'Error (%)': [f"{((v - actual_value) / actual_value * 100):+.1f}%" if actual_value > 0 else "N/A" for v in predictions.values()],
                                'Absolute Error': [f"€{abs(v - actual_value):.2f}M" for v in predictions.values()]
                            })

                            st.dataframe(accuracy_df, width='stretch', hide_index=True)

                        # Recommendation
                        st.markdown("---")
                        st.info(f"**Recommendation:** The ensemble average (€{avg_prediction:.2f}M) combines all models and typically provides the most robust prediction.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 14px;">
        <p>Football Player Market Value Predictor | Built with Streamlit</p>
        <p>Model: Random Forest / XGBoost / LightGBM | Data: Football Statistics</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
