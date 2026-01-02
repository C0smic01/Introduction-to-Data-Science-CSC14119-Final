
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TỐI ƯU HÓA MODEL DỰ ĐOÁN GIÁ TRỊ CẦU THỦ")
print("="*80)

# =====================================================
# BƯỚC 1: ĐỌC VÀ PHÂN TÍCH DỮ LIỆU
# =====================================================
df = pd.read_csv(r'D:\DataSciences\Introduction-to-Data-Science-CSC14119-Final\source\Data Modeling\football_players_dataset.csv')

print(f"\nShape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")

# Phân tích target distribution
print("\n" + "="*80)
print("PHÂN TÍCH TARGET (MARKET_VALUE)")
print("="*80)

print("\nStatistics:")
print(df['market_value'].describe())

print("\nDistribution:")
print(f"Mean: {df['market_value'].mean():.2f}M")
print(f"Median: {df['market_value'].median():.2f}M")
print(f"Std: {df['market_value'].std():.2f}M")
print(f"Skewness: {df['market_value'].skew():.2f}")

# Visualize distribution
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Original distribution
axes[0].hist(df['market_value'], bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Market Value (M€)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Market Value Distribution (Original)')
axes[0].axvline(df['market_value'].mean(), color='red', linestyle='--', label='Mean')
axes[0].axvline(df['market_value'].median(), color='green', linestyle='--', label='Median')
axes[0].legend()

# Log-transformed distribution
log_values = np.log1p(df['market_value'])
axes[1].hist(log_values, bins=50, edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Log(Market Value + 1)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Market Value Distribution (Log-transformed)')
axes[1].axvline(log_values.mean(), color='red', linestyle='--', label='Mean')
axes[1].axvline(log_values.median(), color='green', linestyle='--', label='Median')
axes[1].legend()

plt.tight_layout()
plt.savefig('target_distribution.png', dpi=300)
print("\n✓ Đã lưu: target_distribution.png")
plt.close()

# =====================================================
# BƯỚC 2: FEATURE ENGINEERING NÂNG CAO
# =====================================================
print("\n" + "="*80)
print("FEATURE ENGINEERING NÂNG CAO")
print("="*80)

df_features = df.copy()

# 1. LOG TRANSFORM cho các features có skewness cao
numeric_cols = df_features.select_dtypes(include=[np.number]).columns
skewed_features = []

for col in numeric_cols:
    if col not in ['market_value', 'is_GK', 'is_DF', 'is_MF', 'is_FW']:
        skewness = df_features[col].skew()
        if abs(skewness) > 1.0:  # Highly skewed
            skewed_features.append(col)
            # Log transform (adding 1 to handle zeros)
            df_features[f'{col}_log'] = np.log1p(df_features[col])

print(f"✓ Log-transformed {len(skewed_features)} skewed features")

# 2. RATIO FEATURES (tạo các chỉ số hiệu quả)
print("\nTạo ratio features:")

# Goals per shot (hiệu suất ghi bàn)
if 'goals_per_90' in df_features.columns and 'shots_per90' in df_features.columns:
    df_features['goals_per_shot'] = (
        df_features['goals_per_90'] / 
        df_features['shots_per90'].replace(0, 0.01)
    )
    print("   ✓ goals_per_shot")

# Assists per key pass (hiệu suất kiến tạo)
if 'assists_per_90' in df_features.columns and 'key_passes_per90' in df_features.columns:
    df_features['assists_per_keypass'] = (
        df_features['assists_per_90'] / 
        df_features['key_passes_per90'].replace(0, 0.01)
    )
    print("   ✓ assists_per_keypass")

# Take-on success × frequency
if 'take_ons_per90' in df_features.columns and 'take_on_success_pct' in df_features.columns:
    df_features['effective_dribbles'] = (
        df_features['take_ons_per90'] * 
        df_features['take_on_success_pct'] / 100
    )
    print("   ✓ effective_dribbles")

# Progressive actions (tổng hợp)
if 'progressive_passes_per90' in df_features.columns and 'progressive_carries_per90' in df_features.columns:
    df_features['total_progressive_actions'] = (
        df_features['progressive_passes_per90'] + 
        df_features['progressive_carries_per90']
    )
    print("   ✓ total_progressive_actions")

# Defensive contribution
if all(col in df_features.columns for col in ['tackles_per90', 'interceptions_per90', 'blocks_per90']):
    df_features['total_defensive_actions'] = (
        df_features['tackles_per90'] + 
        df_features['interceptions_per90'] + 
        df_features['blocks_per90']
    )
    print("   ✓ total_defensive_actions")

# 3. INTERACTION FEATURES (tương tác giữa các features)
print("\nTạo interaction features:")

# Age × Minutes (kinh nghiệm thực tế)
df_features['age_experience'] = df_features['age'] * np.log1p(df_features['minutes_played'])
print("   ✓ age_experience")

# Position-specific performance
for pos in ['GK', 'DF', 'MF', 'FW']:
    if f'is_{pos}' in df_features.columns:
        # Position × key stat
        if pos == 'FW' and 'goals_per_90' in df_features.columns:
            df_features[f'{pos}_performance'] = (
                df_features[f'is_{pos}'] * df_features['goals_per_90']
            )
        elif pos == 'MF' and 'assists_per_90' in df_features.columns:
            df_features[f'{pos}_performance'] = (
                df_features[f'is_{pos}'] * df_features['assists_per_90']
            )
        elif pos == 'DF' and 'tackles_per90' in df_features.columns:
            df_features[f'{pos}_performance'] = (
                df_features[f'is_{pos}'] * df_features['tackles_per90']
            )

print("   ✓ Position-specific performance features")

# 4. ENCODING CATEGORICAL VARIABLES
print("\nEncoding categorical variables:")

categorical_cols = ['nationality', 'position', 'current_club', 'league']

# Target encoding cho high cardinality features (nationality, club)
for col in ['nationality', 'current_club']:
    if col in df_features.columns:
        # Tính mean market_value cho mỗi category
        target_mean = df_features.groupby(col)['market_value'].mean()
        df_features[f'{col}_encoded'] = df_features[col].map(target_mean)
        
        # Fill missing với overall mean
        df_features[f'{col}_encoded'].fillna(df_features['market_value'].mean(), inplace=True)
        print(f"   ✓ {col}: target-encoded")

# Label encoding cho low cardinality (position, league)
le = LabelEncoder()
for col in ['position', 'league']:
    if col in df_features.columns:
        df_features[f'{col}_encoded'] = le.fit_transform(df_features[col].astype(str))
        print(f"   ✓ {col}: label-encoded")

# Frequency encoding (số lần xuất hiện)
for col in categorical_cols:
    if col in df_features.columns:
        freq = df_features[col].value_counts()
        df_features[f'{col}_frequency'] = df_features[col].map(freq)
        print(f"   ✓ {col}: frequency-encoded")

print(f"\n✓ Total features after engineering: {len(df_features.columns)}")

# =====================================================
# BƯỚC 3: FEATURE SELECTION
# =====================================================
print("\n" + "="*80)
print("FEATURE SELECTION")
print("="*80)

# Chọn features để train
exclude_cols = ['market_value', 'position_category', 'age_group', 'experience_level'] + categorical_cols

feature_cols = [col for col in df_features.columns 
                if col not in exclude_cols 
                and df_features[col].dtype in ['int64', 'float64']]

print(f"Số features để train: {len(feature_cols)}")

# Loại bỏ features có correlation thấp với target
print("\nPhân tích correlation với target:")

X_temp = df_features[feature_cols].fillna(0)
y = df_features['market_value']

correlations = {}
for col in feature_cols:
    correlations[col] = abs(X_temp[col].corr(y))

# Sort và hiển thị top features
top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:30]

print("\nTop 30 features:")
for i, (feature, corr) in enumerate(top_features, 1):
    print(f"   {i:2d}. {feature:50s}: {corr:.4f}")

# Chọn features có correlation > threshold
corr_threshold = 0.05
selected_features = [feat for feat, corr in correlations.items() if corr > corr_threshold]
print(f"\n✓ Selected {len(selected_features)} features (correlation > {corr_threshold})")

# =====================================================
# BƯỚC 4: CHUẨN BỊ DỮ LIỆU
# =====================================================
print("\n" + "="*80)
print("CHUẨN BỊ DỮ LIỆU CHO TRAINING")
print("="*80)

# Loại bỏ outliers trong target (optional)
Q1 = df_features['market_value'].quantile(0.01)
Q3 = df_features['market_value'].quantile(0.99)
df_clean = df_features[(df_features['market_value'] >= Q1) & 
                        (df_features['market_value'] <= Q3)].copy()

print(f"Dữ liệu sau khi loại outliers: {len(df_clean)} ({len(df_clean)/len(df_features)*100:.1f}%)")

# Prepare X, y
X = df_clean[selected_features].fillna(0)
y = df_clean['market_value']

# LOG TRANSFORM TARGET (quan trọng!)
y_log = np.log1p(y)

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Target mean: {y.mean():.2f}M")
print(f"Target (log) mean: {y_log.mean():.4f}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

print(f"\nTrain set: {X_train.shape[0]}")
print(f"Test set: {X_test.shape[0]}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Features đã được scaled")

# =====================================================
# BƯỚC 5: TRAIN MULTIPLE MODELS
# =====================================================
print("\n" + "="*80)
print("TRAIN VÀ SO SÁNH MODELS")
print("="*80)

models = {}
results = {}

# 1. Random Forest
print("\n1. Random Forest Regressor")
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
models['Random Forest'] = rf

y_pred_rf = rf.predict(X_test)
y_pred_rf_original = np.expm1(y_pred_rf)
y_test_original = np.expm1(y_test)

r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test_original, y_pred_rf_original)
rmse_rf = np.sqrt(mean_squared_error(y_test_original, y_pred_rf_original))

results['Random Forest'] = {'R2': r2_rf, 'MAE': mae_rf, 'RMSE': rmse_rf}
print(f"   R² Score: {r2_rf:.4f}")
print(f"   MAE: {mae_rf:.4f}M")
print(f"   RMSE: {rmse_rf:.4f}M")

# 2. Gradient Boosting
print("\n2. Gradient Boosting Regressor")
gb = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=7,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train, y_train)
models['Gradient Boosting'] = gb

y_pred_gb = gb.predict(X_test)
y_pred_gb_original = np.expm1(y_pred_gb)

r2_gb = r2_score(y_test, y_pred_gb)
mae_gb = mean_absolute_error(y_test_original, y_pred_gb_original)
rmse_gb = np.sqrt(mean_squared_error(y_test_original, y_pred_gb_original))

results['Gradient Boosting'] = {'R2': r2_gb, 'MAE': mae_gb, 'RMSE': rmse_gb}
print(f"   R² Score: {r2_gb:.4f}")
print(f"   MAE: {mae_gb:.4f}M")
print(f"   RMSE: {rmse_gb:.4f}M")

# 3. XGBoost
print("\n3. XGBoost Regressor")
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=7,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
models['XGBoost'] = xgb_model

y_pred_xgb = xgb_model.predict(X_test)
y_pred_xgb_original = np.expm1(y_pred_xgb)

r2_xgb = r2_score(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test_original, y_pred_xgb_original)
rmse_xgb = np.sqrt(mean_squared_error(y_test_original, y_pred_xgb_original))

results['XGBoost'] = {'R2': r2_xgb, 'MAE': mae_xgb, 'RMSE': rmse_xgb}
print(f"   R² Score: {r2_xgb:.4f}")
print(f"   MAE: {mae_xgb:.4f}M")
print(f"   RMSE: {rmse_xgb:.4f}M")

# =====================================================
# BƯỚC 6: SO SÁNH MODELS
# =====================================================
print("\n" + "="*80)
print("SO SÁNH PERFORMANCE")
print("="*80)

results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('R2', ascending=False)

print("\n", results_df.to_string())

# Visualize comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

metrics = ['R2', 'MAE', 'RMSE']
colors = ['skyblue', 'lightcoral', 'lightgreen']

for idx, (metric, color) in enumerate(zip(metrics, colors)):
    values = results_df[metric].values
    models_names = results_df.index.tolist()
    
    axes[idx].bar(models_names, values, color=color, alpha=0.7, edgecolor='black')
    axes[idx].set_ylabel(metric, fontsize=12)
    axes[idx].set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
    axes[idx].tick_params(axis='x', rotation=45)
    axes[idx].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(values):
        axes[idx].text(i, v + 0.01, f'{v:.4f}' if metric == 'R2' else f'{v:.2f}', 
                      ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300)
print("\n✓ Đã lưu: model_comparison.png")
plt.close()

# =====================================================
# BƯỚC 7: FEATURE IMPORTANCE
# =====================================================
print("\n" + "="*80)
print("FEATURE IMPORTANCE (Best Model)")
print("="*80)

best_model_name = results_df.index[0]
best_model = models[best_model_name]

print(f"Best model: {best_model_name}")

# Get feature importance
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 Most Important Features:")
    print(feature_importance.head(20).to_string(index=False))
    
    # Visualize
    plt.figure(figsize=(12, 8))
    top_20 = feature_importance.head(20)
    plt.barh(range(len(top_20)), top_20['importance'], alpha=0.7)
    plt.yticks(range(len(top_20)), top_20['feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top 20 Feature Importance ({best_model_name})', 
             fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    print("\n✓ Đã lưu: feature_importance.png")
    plt.close()

# =====================================================
# BƯỚC 8: RESIDUALS ANALYSIS
# =====================================================
print("\n" + "="*80)
print("PHÂN TÍCH RESIDUALS")
print("="*80)

# Chọn best model predictions
best_pred = None
if best_model_name == 'Random Forest':
    best_pred = y_pred_rf_original
elif best_model_name == 'Gradient Boosting':
    best_pred = y_pred_gb_original
else:
    best_pred = y_pred_xgb_original

residuals = y_test_original - best_pred

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Predicted vs Actual
axes[0, 0].scatter(y_test_original, best_pred, alpha=0.5)
axes[0, 0].plot([y_test_original.min(), y_test_original.max()], 
                [y_test_original.min(), y_test_original.max()], 
                'r--', lw=2)
axes[0, 0].set_xlabel('Actual Market Value (M€)')
axes[0, 0].set_ylabel('Predicted Market Value (M€)')
axes[0, 0].set_title('Predicted vs Actual')
axes[0, 0].grid(alpha=0.3)

# 2. Residuals distribution
axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].axvline(0, color='red', linestyle='--', label='Zero')
axes[0, 1].set_xlabel('Residuals (M€)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Residuals Distribution')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 3. Residuals vs Predicted
axes[1, 0].scatter(best_pred, residuals, alpha=0.5)
axes[1, 0].axhline(0, color='red', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Predicted Market Value (M€)')
axes[1, 0].set_ylabel('Residuals (M€)')
axes[1, 0].set_title('Residuals vs Predicted')
axes[1, 0].grid(alpha=0.3)

# 4. Q-Q plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('residuals_analysis.png', dpi=300)
print("✓ Đã lưu: residuals_analysis.png")
plt.close()

# Statistics
print(f"\nResiduals Statistics:")
print(f"   Mean: {residuals.mean():.4f}M")
print(f"   Std: {residuals.std():.4f}M")
print(f"   Min: {residuals.min():.4f}M")
print(f"   Max: {residuals.max():.4f}M")

# =====================================================
# BƯỚC 9: SAVE BEST MODEL
# =====================================================
print("\n" + "="*80)
print("LƯU MODEL")
print("="*80)

import joblib

# Save model, scaler, và feature names
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(selected_features, 'selected_features.pkl')

print(f"✓ Đã lưu: best_model.pkl ({best_model_name})")
print("✓ Đã lưu: scaler.pkl")
print("✓ Đã lưu: selected_features.pkl")

# =====================================================
# BƯỚC 10: SUMMARY REPORT
# =====================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

summary = f"""
📊 MODEL PERFORMANCE SUMMARY
{'='*80}

Dataset:
   - Total samples: {len(df_clean):,}
   - Training samples: {len(X_train):,}
   - Test samples: {len(X_test):,}
   - Features used: {len(selected_features)}

Best Model: {best_model_name}
   - R² Score: {results_df.iloc[0]['R2']:.4f}
   - MAE: {results_df.iloc[0]['MAE']:.4f}M
   - RMSE: {results_df.iloc[0]['RMSE']:.4f}M

Improvement from baseline:
   - Original R²: 0.2915
   - New R²: {results_df.iloc[0]['R2']:.4f}
   - Improvement: {(results_df.iloc[0]['R2'] - 0.2915) / 0.2915 * 100:+.1f}%

Files created:
   ✓ target_distribution.png
   ✓ model_comparison.png
   ✓ feature_importance.png
   ✓ residuals_analysis.png
   ✓ best_model.pkl
   ✓ scaler.pkl
   ✓ selected_features.pkl
"""

print(summary)

# Save report
with open('model_report.txt', 'w', encoding='utf-8') as f:
    f.write(summary)

print("✓ Đã lưu: model_report.txt")

print("\n" + "="*80)
print("HOÀN TẤT! 🎉")
print("="*80)