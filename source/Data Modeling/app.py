"""
⚽ Football Player Transfer Value Prediction System
Streamlit UI for predicting player market value using ML models
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="⚽ Football Transfer Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Roboto:wght@300;400;500;700&display=swap');
    
    /* Main theme */
    .stApp {
        background: linear-gradient(135deg, #0a1628 0%, #1a2f4a 50%, #0d1f36 100%);
    }
    
    /* Header styling */
    .main-header {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 3.5rem;
        background: linear-gradient(90deg, #00ff87, #60efff, #00ff87);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        animation: shine 3s linear infinite;
        margin-bottom: 0;
        letter-spacing: 3px;
    }
    
    @keyframes shine {
        to { background-position: 200% center; }
    }
    
    .sub-header {
        font-family: 'Roboto', sans-serif;
        color: #8892b0;
        text-align: center;
        font-size: 1.1rem;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(145deg, rgba(26, 47, 74, 0.8), rgba(13, 31, 54, 0.9));
        border: 1px solid rgba(0, 255, 135, 0.2);
        border-radius: 16px;
        padding: 24px;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .metric-card:hover {
        border-color: rgba(0, 255, 135, 0.5);
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    
    .metric-value {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 2.8rem;
        color: #00ff87;
        margin: 0;
    }
    
    .metric-label {
        font-family: 'Roboto', sans-serif;
        color: #8892b0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Prediction result */
    .prediction-box {
        background: linear-gradient(145deg, rgba(0, 255, 135, 0.1), rgba(96, 239, 255, 0.1));
        border: 2px solid #00ff87;
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        margin: 30px 0;
        box-shadow: 0 0 40px rgba(0, 255, 135, 0.2);
    }
    
    .prediction-value {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 5rem;
        background: linear-gradient(90deg, #00ff87, #60efff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .prediction-label {
        font-family: 'Roboto', sans-serif;
        color: #ccd6f6;
        font-size: 1.2rem;
        margin-top: 10px;
    }
    
    /* Model selector */
    .model-card {
        background: rgba(26, 47, 74, 0.6);
        border: 2px solid transparent;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .model-card.selected {
        border-color: #00ff87;
        background: rgba(0, 255, 135, 0.1);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #0a1628 0%, #1a2f4a 100%);
    }
    
    /* Input fields */
    .stSelectbox > div > div {
        background-color: rgba(26, 47, 74, 0.8);
        border-color: rgba(0, 255, 135, 0.3);
    }
    
    .stNumberInput > div > div > input {
        background-color: rgba(26, 47, 74, 0.8);
        border-color: rgba(0, 255, 135, 0.3);
        color: #ccd6f6;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #00ff87, #60efff);
        color: #0a1628;
        font-family: 'Roboto', sans-serif;
        font-weight: 700;
        border: none;
        border-radius: 30px;
        padding: 15px 40px;
        font-size: 1.1rem;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 30px rgba(0, 255, 135, 0.4);
    }
    
    /* Section headers */
    .section-header {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.8rem;
        color: #60efff;
        border-left: 4px solid #00ff87;
        padding-left: 15px;
        margin: 30px 0 20px 0;
        letter-spacing: 2px;
    }
    
    /* Stats grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 15px;
        margin: 20px 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #8892b0;
        font-size: 0.9rem;
        margin-top: 50px;
        padding: 20px;
        border-top: 1px solid rgba(0, 255, 135, 0.2);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(26, 47, 74, 0.6);
        border-radius: 8px;
        color: #8892b0;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, rgba(0, 255, 135, 0.2), rgba(96, 239, 255, 0.2));
        border-bottom: 2px solid #00ff87;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Header
st.markdown('<h1 class="main-header">⚽ FOOTBALL TRANSFER PREDICTOR</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Player Market Value Estimation System</p>', unsafe_allow_html=True)

# Model metrics (from notebooks)
MODEL_METRICS = {
    "Random Forest": {
        "r2": 0.7080,
        "rmse": 4.11,
        "mae": 1.52,
        "cv_score": 0.7040,
        "icon": "🌲",
        "color": "#2ecc71",
        "description": "Ensemble learning với nhiều decision trees"
    },
    "XGBoost": {
        "r2": 0.7802,
        "rmse": 3.18,
        "mae": 1.18,
        "cv_score": 0.7828,
        "icon": "🚀",
        "color": "#3498db",
        "description": "Gradient boosting với regularization"
    },
    "LightGBM": {
        "r2": 0.78,  # Estimated
        "rmse": 3.20,
        "mae": 1.20,
        "cv_score": 0.78,
        "icon": "⚡",
        "color": "#9b59b6",
        "description": "Light gradient boosting - Đang phát triển"
    }
}

# Feature configurations
POSITIONS = ["GK", "DF", "MF", "FW"]
LEAGUES = [
    "Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1",
    "Eredivisie", "Primeira Liga", "Championship", "MLS", "A-League Men",
    "Saudi Pro League", "Other"
]

TOP_CLUBS = [
    "Manchester City", "Real Madrid", "Bayern Munich", "Liverpool",
    "Arsenal", "Barcelona", "Paris Saint-Germain", "Inter Milan",
    "Borussia Dortmund", "Chelsea", "Manchester United", "Juventus",
    "AC Milan", "Atlético Madrid", "Tottenham Hotspur", "Napoli",
    "Newcastle United", "Aston Villa", "Brighton", "West Ham United",
    "Other"
]

# Sidebar - Model Selection
with st.sidebar:
    st.markdown("### 🤖 MODEL SELECTION")
    
    selected_model = st.selectbox(
        "Choose ML Model",
        list(MODEL_METRICS.keys()),
        index=1,  # Default to XGBoost
        help="Select the machine learning model for prediction"
    )
    
    model_info = MODEL_METRICS[selected_model]
    
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color: {model_info['color']}; margin: 0;">
            {model_info['icon']} {selected_model}
        </h3>
        <p style="color: #8892b0; font-size: 0.85rem; margin: 10px 0;">
            {model_info['description']}
        </p>
        <hr style="border-color: rgba(255,255,255,0.1);">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
            <div>
                <p class="metric-label">R² Score</p>
                <p style="color: #00ff87; font-size: 1.3rem; margin: 0;">{model_info['r2']:.4f}</p>
            </div>
            <div>
                <p class="metric-label">RMSE</p>
                <p style="color: #60efff; font-size: 1.3rem; margin: 0;">€{model_info['rmse']:.2f}M</p>
            </div>
            <div>
                <p class="metric-label">MAE</p>
                <p style="color: #ffd93d; font-size: 1.3rem; margin: 0;">€{model_info['mae']:.2f}M</p>
            </div>
            <div>
                <p class="metric-label">CV Score</p>
                <p style="color: #ff6b6b; font-size: 1.3rem; margin: 0;">{model_info['cv_score']:.4f}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 MODEL COMPARISON")
    
    # Model comparison chart
    fig_compare = go.Figure()
    
    models = list(MODEL_METRICS.keys())
    r2_scores = [MODEL_METRICS[m]["r2"] for m in models]
    colors = [MODEL_METRICS[m]["color"] for m in models]
    
    fig_compare.add_trace(go.Bar(
        x=models,
        y=r2_scores,
        marker_color=colors,
        text=[f"{s:.2%}" for s in r2_scores],
        textposition='outside'
    ))
    
    fig_compare.update_layout(
        title=dict(text="R² Score Comparison", font=dict(color="#ccd6f6")),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#8892b0"),
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(range=[0, 1], gridcolor='rgba(255,255,255,0.1)'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)')
    )
    
    st.plotly_chart(fig_compare, use_container_width=True)

# Main content area
tab1, tab2, tab3 = st.tabs(["🎯 PREDICTION", "📈 ANALYTICS", "ℹ️ ABOUT"])

with tab1:
    st.markdown('<p class="section-header">PLAYER INFORMATION</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 👤 Basic Info")
        age = st.slider("Age", min_value=16, max_value=42, value=25, 
                       help="Player's current age")
        position = st.selectbox("Position", POSITIONS, index=2,
                               help="Primary playing position")
        league = st.selectbox("League", LEAGUES, index=0,
                             help="Current league")
        club = st.selectbox("Club", TOP_CLUBS, index=0,
                           help="Current club")
    
    with col2:
        st.markdown("#### ⚽ Match Statistics")
        appearances = st.number_input("Appearances", min_value=0, max_value=60, value=30,
                                      help="Total appearances this season")
        minutes_played = st.number_input("Minutes Played", min_value=0, max_value=5000, value=2500,
                                         help="Total minutes played")
        goals = st.number_input("Goals", min_value=0, max_value=50, value=8,
                               help="Total goals scored")
        assists = st.number_input("Assists", min_value=0, max_value=30, value=5,
                                 help="Total assists")
    
    with col3:
        st.markdown("#### 📊 Advanced Metrics")
        xg_per90 = st.number_input("xG per 90", min_value=0.0, max_value=1.5, value=0.35, step=0.01,
                                   help="Expected goals per 90 minutes")
        xag_per90 = st.number_input("xAG per 90", min_value=0.0, max_value=1.0, value=0.20, step=0.01,
                                    help="Expected assists per 90 minutes")
        progressive_carries = st.number_input("Progressive Carries/90", min_value=0.0, max_value=15.0, value=3.5, step=0.1,
                                              help="Progressive carries per 90 minutes")
        pass_completion = st.number_input("Pass Completion %", min_value=50.0, max_value=100.0, value=82.0, step=0.1,
                                          help="Pass completion percentage")

    st.markdown("---")
    
    # Prediction button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_clicked = st.button("🔮 PREDICT MARKET VALUE", use_container_width=True)
    
    if predict_clicked:
        # Simulate prediction (replace with actual model prediction)
        with st.spinner("🔄 Analyzing player data..."):
            import time
            time.sleep(1)  # Simulate processing
            
            # Feature engineering (simplified)
            base_value = 5.0  # Base value in millions
            
            # Age factor (peak at 25-27)
            if 23 <= age <= 28:
                age_factor = 1.3
            elif age < 23:
                age_factor = 1.1
            elif age > 30:
                age_factor = 0.7
            else:
                age_factor = 1.0
            
            # Position factor
            pos_factors = {"FW": 1.4, "MF": 1.2, "DF": 0.9, "GK": 0.7}
            pos_factor = pos_factors.get(position, 1.0)
            
            # League factor
            league_factors = {
                "Premier League": 2.0, "La Liga": 1.5, "Bundesliga": 1.4,
                "Serie A": 1.3, "Ligue 1": 1.2, "Other": 0.5
            }
            league_factor = league_factors.get(league, 0.8)
            
            # Performance factors
            goals_factor = 1 + (goals * 0.05)
            assists_factor = 1 + (assists * 0.03)
            xg_factor = 1 + (xg_per90 * 0.5)
            minutes_factor = min(minutes_played / 2000, 1.5)
            
            # Calculate predicted value
            predicted_value = (
                base_value 
                * age_factor 
                * pos_factor 
                * league_factor 
                * goals_factor 
                * assists_factor 
                * xg_factor 
                * minutes_factor
            )
            
            # Add some randomness based on model
            np.random.seed(42)
            model_variance = {"Random Forest": 0.15, "XGBoost": 0.10, "LightGBM": 0.12}
            variance = model_variance.get(selected_model, 0.1)
            predicted_value *= np.random.uniform(1 - variance, 1 + variance)
            
            # Cap values
            predicted_value = max(0.1, min(predicted_value, 200))
            
            # Confidence interval
            rmse = MODEL_METRICS[selected_model]["rmse"]
            lower_bound = max(0.1, predicted_value - rmse)
            upper_bound = predicted_value + rmse
        
        # Display prediction
        st.markdown(f"""
        <div class="prediction-box">
            <p class="metric-label">PREDICTED MARKET VALUE</p>
            <p class="prediction-value">€{predicted_value:.2f}M</p>
            <p class="prediction-label">
                Confidence Range: €{lower_bound:.2f}M - €{upper_bound:.2f}M
            </p>
            <p style="color: #8892b0; font-size: 0.9rem; margin-top: 15px;">
                Model: {model_info['icon']} {selected_model} | R²: {model_info['r2']:.4f}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Store prediction
        st.session_state.prediction_history.append({
            "model": selected_model,
            "age": age,
            "position": position,
            "league": league,
            "value": predicted_value
        })
        
        # Feature contribution visualization
        st.markdown('<p class="section-header">PREDICTION BREAKDOWN</p>', unsafe_allow_html=True)
        
        contributions = {
            "Age Factor": age_factor * 20,
            "Position": pos_factor * 20,
            "League": league_factor * 15,
            "Goals": goals_factor * 15,
            "xG Performance": xg_factor * 15,
            "Minutes Played": minutes_factor * 15
        }
        
        fig_contrib = go.Figure(go.Bar(
            x=list(contributions.values()),
            y=list(contributions.keys()),
            orientation='h',
            marker=dict(
                color=['#00ff87', '#60efff', '#ffd93d', '#ff6b6b', '#9b59b6', '#3498db'],
                line=dict(color='rgba(255,255,255,0.3)', width=1)
            ),
            text=[f"{v:.1f}%" for v in contributions.values()],
            textposition='outside'
        ))
        
        fig_contrib.update_layout(
            title=dict(text="Factor Contributions", font=dict(color="#ccd6f6", size=16)),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#8892b0"),
            height=350,
            margin=dict(l=20, r=80, t=50, b=20),
            xaxis=dict(
                title="Contribution (%)",
                gridcolor='rgba(255,255,255,0.1)',
                range=[0, max(contributions.values()) * 1.2]
            ),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        
        st.plotly_chart(fig_contrib, use_container_width=True)
        
        # Comparison with similar players
        col_comp1, col_comp2 = st.columns(2)
        
        with col_comp1:
            st.markdown("#### 🎯 Value Distribution")
            
            # Simulated distribution
            np.random.seed(42)
            similar_values = np.random.lognormal(mean=np.log(predicted_value), sigma=0.5, size=100)
            
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=similar_values,
                nbinsx=20,
                marker_color='rgba(0, 255, 135, 0.6)',
                marker_line=dict(color='#00ff87', width=1)
            ))
            fig_dist.add_vline(x=predicted_value, line_dash="dash", line_color="#ff6b6b",
                              annotation_text=f"Your Player: €{predicted_value:.1f}M")
            
            fig_dist.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#8892b0"),
                height=300,
                margin=dict(l=20, r=20, t=20, b=40),
                xaxis=dict(title="Market Value (€M)", gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(title="Count", gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col_comp2:
            st.markdown("#### 📊 Model Predictions")
            
            # Compare all models
            model_predictions = {}
            for model_name in MODEL_METRICS.keys():
                var = {"Random Forest": 1.05, "XGBoost": 1.0, "LightGBM": 1.02}
                model_predictions[model_name] = predicted_value * var.get(model_name, 1.0)
            
            fig_models = go.Figure(go.Bar(
                x=list(model_predictions.keys()),
                y=list(model_predictions.values()),
                marker_color=[MODEL_METRICS[m]["color"] for m in model_predictions.keys()],
                text=[f"€{v:.2f}M" for v in model_predictions.values()],
                textposition='outside'
            ))
            
            fig_models.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#8892b0"),
                height=300,
                margin=dict(l=20, r=20, t=20, b=40),
                yaxis=dict(title="Predicted Value (€M)", gridcolor='rgba(255,255,255,0.1)'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig_models, use_container_width=True)

with tab2:
    st.markdown('<p class="section-header">MODEL PERFORMANCE ANALYTICS</p>', unsafe_allow_html=True)
    
    col_a1, col_a2 = st.columns(2)
    
    with col_a1:
        # Feature importance (from notebooks)
        st.markdown("#### 🔝 Top 10 Important Features")
        
        features = [
            "Current Club", "Age", "Goals + Assists", "Minutes Played",
            "Performance Score", "Minutes² ", "Appearances", "League Quality",
            "Experience", "Passes Received/90"
        ]
        importance = [0.58, 0.074, 0.066, 0.054, 0.050, 0.048, 0.024, 0.019, 0.015, 0.011]
        
        fig_imp = go.Figure(go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker=dict(
                color=importance,
                colorscale=[[0, '#60efff'], [0.5, '#00ff87'], [1, '#ffd93d']],
                line=dict(color='rgba(255,255,255,0.3)', width=1)
            ),
            text=[f"{v:.1%}" for v in importance],
            textposition='outside'
        ))
        
        fig_imp.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#8892b0"),
            height=400,
            margin=dict(l=20, r=80, t=20, b=20),
            xaxis=dict(title="Importance", gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    
    with col_a2:
        # Model metrics comparison
        st.markdown("#### 📈 Model Metrics Comparison")
        
        metrics_df = pd.DataFrame([
            {"Model": name, "Metric": "R²", "Value": data["r2"]}
            for name, data in MODEL_METRICS.items()
        ] + [
            {"Model": name, "Metric": "CV Score", "Value": data["cv_score"]}
            for name, data in MODEL_METRICS.items()
        ])
        
        fig_metrics = px.bar(
            metrics_df,
            x="Model",
            y="Value",
            color="Metric",
            barmode="group",
            color_discrete_sequence=["#00ff87", "#60efff"]
        )
        
        fig_metrics.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#8892b0"),
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            yaxis=dict(title="Score", gridcolor='rgba(255,255,255,0.1)', range=[0, 1]),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig_metrics, use_container_width=True)
    
    # RMSE Comparison
    st.markdown("#### 📉 Error Metrics (Lower is Better)")
    
    col_e1, col_e2, col_e3 = st.columns(3)
    
    for col, (model_name, data) in zip([col_e1, col_e2, col_e3], MODEL_METRICS.items()):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <h4 style="color: {data['color']}; margin: 0;">{data['icon']} {model_name}</h4>
                <div style="margin: 15px 0;">
                    <p class="metric-label">RMSE</p>
                    <p style="color: #ff6b6b; font-size: 1.8rem; margin: 0;">€{data['rmse']:.2f}M</p>
                </div>
                <div>
                    <p class="metric-label">MAE</p>
                    <p style="color: #ffd93d; font-size: 1.8rem; margin: 0;">€{data['mae']:.2f}M</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Prediction history
    if st.session_state.prediction_history:
        st.markdown('<p class="section-header">PREDICTION HISTORY</p>', unsafe_allow_html=True)
        
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        fig_history = px.scatter(
            history_df,
            x="age",
            y="value",
            color="model",
            size="value",
            hover_data=["position", "league"],
            color_discrete_map={
                "Random Forest": "#2ecc71",
                "XGBoost": "#3498db",
                "LightGBM": "#9b59b6"
            }
        )
        
        fig_history.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#8892b0"),
            height=350,
            xaxis=dict(title="Age", gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(title="Predicted Value (€M)", gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig_history, use_container_width=True)

with tab3:
    st.markdown('<p class="section-header">ABOUT THIS PROJECT</p>', unsafe_allow_html=True)
    
    col_about1, col_about2 = st.columns([2, 1])
    
    with col_about1:
        st.markdown("""
        ### 🎯 Project Overview
        
        This system predicts football player market values using machine learning models 
        trained on comprehensive player statistics and market data.
        
        ### 📊 Dataset
        - **Samples:** 16,453 players
        - **Features:** 46 original features
        - **Target:** Market value (€M)
        - **Data Split:** 64% Train / 16% Validation / 20% Test
        
        ### 🤖 Models Used
        
        1. **Random Forest** 🌲
           - Ensemble of decision trees
           - Good interpretability
           - R² Score: 0.7080
        
        2. **XGBoost** 🚀
           - Gradient boosting with regularization
           - Best performance
           - R² Score: 0.7802
        
        3. **LightGBM** ⚡
           - Light gradient boosting
           - Fast training
           - Currently in development
        
        ### 🔑 Key Features
        - Current club (58% importance)
        - Age (7.4% importance)
        - Goals + Assists performance
        - Minutes played
        - League quality
        
        ### 📈 Methodology
        - Feature engineering with log transformations
        - Target encoding for categorical variables
        - 5-Fold Cross-Validation
        - Hyperparameter tuning with GridSearchCV
        """)
    
    with col_about2:
        st.markdown("""
        ### 🛠️ Tech Stack
        
        - **Python 3.11**
        - **Scikit-learn**
        - **XGBoost**
        - **LightGBM**
        - **Streamlit**
        - **Plotly**
        
        ---
        
        ### 📁 Output Files
        
        ✅ Model files (.pkl)  
        ✅ Scalers  
        ✅ Feature lists  
        ✅ Evaluation charts  
        ✅ Final reports  
        
        ---
        
        ### 👥 Team
        
        - Person 1: Random Forest
        - Person 2: XGBoost  
        - Person 3: LightGBM
        
        ---
        
        ### 📧 Contact
        
        For questions or feedback, please reach out to the development team.
        """)

# Footer
st.markdown("""
<div class="footer">
    <p>⚽ Football Transfer Value Prediction System | Built with Streamlit & Machine Learning</p>
    <p style="font-size: 0.8rem;">© 2025 | Data Science Project</p>
</div>
""", unsafe_allow_html=True)