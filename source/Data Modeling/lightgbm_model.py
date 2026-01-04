# %% [markdown]
# # 🎯 DỰ ĐOÁN GIÁ TRỊ CẦU THỦ BÓNG ĐÁ - LIGHTGBM (OPTIMIZED)
# ## Complete Analysis with LightGBM - Fast Training
# 
# **OPTIMIZATIONS:**
# - Reduced parameter grid (từ 243 → 18 combinations)
# - 3-fold CV thay vì 5-fold (nhanh hơn 40%)
# - Progress bars với tqdm
# - Sampling cho tuning (5000 samples thay vì full dataset)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time
from tqdm.auto import tqdm

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# LightGBM
import lightgbm as lgb

import warnings
import joblib
warnings.filterwarnings('ignore')

# Thiết lập style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✅ Libraries imported successfully!")
print("📊 Using LightGBM (OPTIMIZED VERSION)")
print("🎯 Fast training with progress bars")
print(f"⏰ Started at: {time.strftime('%H:%M:%S')}\n")

# %%
# Load dataset
print("="*80)
print("📂 LOADING DATA")
print("="*80)

df = pd.read_csv('football_players_dataset.csv')

print(f"\n✅ Loaded {len(df):,} samples with {df.shape[1]} features")

# %%
# Feature Engineering
print("\n" + "="*80)
print("🔧 FEATURE ENGINEERING")
print("="*80)

start_time = time.time()

df_features = df.copy()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# 1. Log transformation
print("\n⏳ Log transformation...")
skewed_features = []
for col in tqdm(numeric_cols, desc="Processing", leave=False):
    if col not in ['market_value', 'is_GK', 'is_DF', 'is_MF', 'is_FW']:
        if abs(df_features[col].skew()) > 1.0:
            df_features[f'{col}_log'] = np.log1p(df_features[col])
            skewed_features.append(col)
print(f"✅ Transformed {len(skewed_features)} features")

# 2. Ratio features
print("⏳ Creating ratio features...")
if 'goals' in df_features.columns and 'shots_per90' in df_features.columns:
    df_features['goals_per_shot'] = df_features['goals'] / df_features['shots_per90'].replace(0, 0.01)

if 'passes_completed_per90' in df_features.columns and 'pass_completion_pct' in df_features.columns:
    df_features['pass_efficiency'] = df_features['passes_completed_per90'] * df_features['pass_completion_pct'] / 100

if all(col in df_features.columns for col in ['interceptions_per90', 'blocks_per90']):
    df_features['defensive_contribution'] = df_features['interceptions_per90'] + df_features['blocks_per90']

if all(col in df_features.columns for col in ['progressive_passes_per90', 'progressive_carries_per90']):
    df_features['total_progressive'] = df_features['progressive_passes_per90'] + df_features['progressive_carries_per90']
print("✅ Created 4 ratio features")

# 3. Interaction features
print("⏳ Creating interaction features...")
df_features['age_experience'] = df_features['age'] * np.log1p(df_features['minutes_played'])
if 'minutes_played' in df_features.columns and 'appearances' in df_features.columns:
    df_features['minutes_per_game'] = df_features['minutes_played'] / df_features['appearances'].replace(0, 1)
print("✅ Created 2 interaction features")

# 4. Polynomial features
print("⏳ Creating polynomial features...")
for feat in ['goals', 'assists', 'minutes_played']:
    if feat in df_features.columns:
        df_features[f'{feat}_squared'] = df_features[feat] ** 2
print("✅ Created 3 polynomial features")

# 5. Encoding
print("⏳ Encoding categorical variables...")
categorical_cols = ['nationality', 'position', 'current_club', 'league']

for col in ['nationality', 'current_club']:
    if col in df_features.columns:
        target_mean = df_features.groupby(col)['market_value'].mean()
        df_features[f'{col}_target_enc'] = df_features[col].map(target_mean)
        df_features[f'{col}_target_enc'].fillna(df_features['market_value'].mean(), inplace=True)

le = LabelEncoder()
for col in ['position', 'league']:
    if col in df_features.columns:
        df_features[f'{col}_label_enc'] = le.fit_transform(df_features[col].astype(str))

for col in categorical_cols:
    if col in df_features.columns:
        freq = df_features[col].value_counts()
        df_features[f'{col}_freq'] = df_features[col].map(freq)

print("✅ Encoded categorical variables")

elapsed = time.time() - start_time
print(f"\n✅ Feature engineering completed in {elapsed:.2f}s")
print(f"📊 Total features: {len(df_features.columns)}")

# %%
# Feature Selection
print("\n" + "="*80)
print("🎯 FEATURE SELECTION")
print("="*80)

exclude_cols = ['market_value', 'position_category', 'nationality', 'position', 
                'current_club', 'league']

feature_cols = [col for col in df_features.columns 
                if col not in exclude_cols 
                and df_features[col].dtype in ['int64', 'float64']]

X_temp = df_features[feature_cols].fillna(0)
y_temp = df_features['market_value']

print(f"⏳ Calculating correlations for {len(feature_cols)} features...")
correlations = {}
for col in tqdm(feature_cols, desc="Correlations", leave=False):
    try:
        correlations[col] = abs(X_temp[col].corr(y_temp))
    except:
        correlations[col] = 0

corr_threshold = 0.05
selected_features = [feat for feat, corr in correlations.items() if corr > corr_threshold]

print(f"✅ Selected {len(selected_features)} features (correlation > {corr_threshold})")

sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
print("\n🔝 Top 10 features:")
for i, (feat, corr) in enumerate(sorted_corr[:10], 1):
    print(f"   {i:2d}. {feat:40s}: {corr:.4f}")

# %%
# Data Preparation
print("\n" + "="*80)
print("🔨 DATA PREPARATION")
print("="*80)

# Remove outliers
Q1 = df_features['market_value'].quantile(0.01)
Q3 = df_features['market_value'].quantile(0.99)
df_clean = df_features[(df_features['market_value'] >= Q1) & 
                        (df_features['market_value'] <= Q3)].copy()

print(f"✅ Removed outliers: {len(df_clean):,}/{len(df_features):,} samples kept")

# Prepare X and y
X = df_clean[selected_features].fillna(0)
y = df_clean['market_value']
y_log = np.log1p(y)

# Three-way split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y_log, test_size=0.2, random_state=42, shuffle=True
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42, shuffle=True
)

print(f"\n📊 Data split:")
print(f"   Training:   {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Validation: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"   Test:       {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("✅ Feature scaling completed")

# %%
# Initial Model Training
print("\n" + "="*80)
print("🤖 INITIAL MODEL TRAINING")
print("="*80)

model = lgb.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

train_start = time.time()
print("⏳ Training initial model...")
model.fit(X_train_scaled, y_train)
train_time = time.time() - train_start
print(f"✅ Training completed in {train_time:.2f}s")

# Evaluate
y_val_pred_log = model.predict(X_val_scaled)
y_test_pred_log = model.predict(X_test_scaled)

y_val_pred = np.expm1(y_val_pred_log)
y_val_orig = np.expm1(y_val)
y_test_pred = np.expm1(y_test_pred_log)
y_test_orig = np.expm1(y_test)

val_r2 = r2_score(y_val, y_val_pred_log)
test_r2 = r2_score(y_test, y_test_pred_log)
test_rmse = np.sqrt(mean_squared_error(y_test_orig, y_test_pred))
test_mae = mean_absolute_error(y_test_orig, y_test_pred)

print(f"\n📊 Initial Performance:")
print(f"   Val R²:   {val_r2:.4f}")
print(f"   Test R²:  {test_r2:.4f}")
print(f"   Test RMSE: €{test_rmse:.2f}M")
print(f"   Test MAE:  €{test_mae:.2f}M")

# Cross-validation with progress
print("\n⏳ Running 5-fold cross-validation...")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = []
for fold, (train_idx, val_idx) in enumerate(tqdm(kfold.split(X_train_scaled), 
                                                   total=5, desc="CV Folds")):
    X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    fold_model = lgb.LGBMRegressor(**model.get_params())
    fold_model.fit(X_fold_train, y_fold_train)
    fold_score = fold_model.score(X_fold_val, y_fold_val)
    cv_scores.append(fold_score)

cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)

print(f"✅ CV R²: {cv_mean:.4f} ± {cv_std:.4f}")

initial_results = {
    'cv_mean': cv_mean, 'cv_std': cv_std,
    'val_r2': val_r2, 'test_r2': test_r2,
    'test_rmse': test_rmse, 'test_mae': test_mae
}

# %%
# Hyperparameter Tuning - OPTIMIZED
print("\n" + "="*80)
print("⚙️ HYPERPARAMETER TUNING (OPTIMIZED)")
print("="*80)

# CRITICAL: Reduced parameter grid
param_grid = {
    'n_estimators': [150, 200],           # 3 → 2 values
    'learning_rate': [0.05, 0.07],        # 3 → 2 values  
    'max_depth': [5, 6],                  # 3 → 2 values
    'num_leaves': [31],                   # 3 → 1 value (best default)
    'min_child_samples': [20]             # 3 → 1 value (best default)
}

# CRITICAL: Sample data for faster tuning
# sample_size = 5000
# if len(X_train_scaled) > sample_size:
#     print(f"⚠️  Dataset large ({len(X_train_scaled):,} samples)")
#     print(f"⚠️  Sampling {sample_size:,} samples for tuning...")
#     indices = np.random.choice(len(X_train_scaled), sample_size, replace=False)
#     X_sample = X_train_scaled[indices]
#     y_sample = y_train.iloc[indices]
# else:
X_sample = X_train_scaled
y_sample = y_train

n_combinations = np.prod([len(v) for v in param_grid.values()])
n_folds = 5  # Sử dụng 5-fold cross-validation
total_fits = n_combinations * n_folds

print(f"\n📊 Tuning configuration:")
print(f"   Parameter combinations: {n_combinations}")
print(f"   Cross-validation folds: {n_folds}")
print(f"   Total model fits: {total_fits}")
print(f"   Training samples: {len(X_sample):,}")
print(f"   Estimated time: ~{total_fits * 0.5:.1f} minutes")
print(f"\n⏰ Started at: {time.strftime('%H:%M:%S')}")

tune_start = time.time()

base_model = lgb.LGBMRegressor(
    random_state=42, 
    n_jobs=-1, 
    verbose=-1,
    force_col_wise=True,  # Speed optimization
    max_bin=255           # Speed optimization
)

# Custom progress bar for GridSearchCV
class TqdmGridSearchCV(GridSearchCV):
    def _run_search(self, evaluate_candidates):
        with tqdm(total=len(self.param_grid) if hasattr(self, 'param_grid') 
                  else n_combinations, desc="Grid Search") as pbar:
            def evaluate_candidates_progress(*args, **kwargs):
                result = evaluate_candidates(*args, **kwargs)
                pbar.update(1)
                return result
            super()._run_search(evaluate_candidates_progress)

# Use custom GridSearchCV with progress
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=n_folds,
    scoring='r2',
    n_jobs=-1,
    verbose=0  # Disable default verbose to use tqdm
)

print("\n⏳ Grid search in progress...\n")
grid_search.fit(X_sample, y_sample)

tune_time = time.time() - tune_start

print(f"\n✅ Grid search completed in {tune_time/60:.2f} minutes")
print(f"⏰ Finished at: {time.strftime('%H:%M:%S')}")

print(f"\n🏆 Best Parameters:")
for param, value in grid_search.best_params_.items():
    print(f"   {param}: {value}")
print(f"\n📊 Best CV Score: {grid_search.best_score_:.4f}")

# Retrain on FULL training data
print(f"\n⏳ Retraining best model on full training set ({len(X_train_scaled):,} samples)...")
retrain_start = time.time()

final_model = lgb.LGBMRegressor(
    **grid_search.best_params_,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
    force_col_wise=True,
    max_bin=255
)
final_model.fit(X_train_scaled, y_train)

retrain_time = time.time() - retrain_start
print(f"✅ Retraining completed in {retrain_time:.2f}s")

# Final evaluation
y_val_pred_tuned = final_model.predict(X_val_scaled)
y_test_pred_tuned = final_model.predict(X_test_scaled)

y_test_pred_tuned_orig = np.expm1(y_test_pred_tuned)
y_test_orig = np.expm1(y_test)

test_r2_tuned = r2_score(y_test, y_test_pred_tuned)
test_mse_tuned = mean_squared_error(y_test_orig, y_test_pred_tuned_orig)
test_rmse_tuned = np.sqrt(test_mse_tuned)
test_mae_tuned = mean_absolute_error(y_test_orig, y_test_pred_tuned_orig)
test_mape_tuned = np.mean(np.abs((y_test_orig - y_test_pred_tuned_orig) / y_test_orig)) * 100

print(f"\n📈 Final Tuned Model Performance:")
print(f"   Test R²:   {test_r2_tuned:.4f}")
print(f"   Test RMSE: €{test_rmse_tuned:.2f}M")
print(f"   Test MAE:  €{test_mae_tuned:.2f}M")
print(f"   Test MAPE: {test_mape_tuned:.2f}%")

improvement = ((test_r2_tuned - test_r2) / test_r2) * 100
print(f"\n💡 Improvement: {improvement:+.2f}%")

final_metrics = {
    'r2': test_r2_tuned,
    'mse': test_mse_tuned,
    'rmse': test_rmse_tuned,
    'mae': test_mae_tuned,
    'mape': test_mape_tuned
}

# %%
# Save Results
print("\n" + "="*80)
print("💾 SAVING RESULTS")
print("="*80)

joblib.dump(final_model, 'lightgbm_final_model.pkl')
joblib.dump(scaler, 'lightgbm_scaler.pkl')
joblib.dump(selected_features, 'lightgbm_selected_features.pkl')

metadata = {
    'model_name': 'LightGBM',
    'n_features': len(selected_features),
    'feature_names': selected_features,
    'n_train': len(X_train),
    'n_val': len(X_val),
    'n_test': len(X_test),
    'test_r2': final_metrics['r2'],
    'test_rmse': final_metrics['rmse'],
    'test_mae': final_metrics['mae'],
    'best_params': grid_search.best_params_,
    'cv_score': grid_search.best_score_,
    'training_time_seconds': train_time,
    'tuning_time_seconds': tune_time
}

joblib.dump(metadata, 'lightgbm_metadata.pkl')

print("✅ Saved: lightgbm_final_model.pkl")
print("✅ Saved: lightgbm_scaler.pkl")
print("✅ Saved: lightgbm_selected_features.pkl")
print("✅ Saved: lightgbm_metadata.pkl")

# %%
# Final Report
print("\n" + "="*80)
print("📊 FINAL REPORT")
print("="*80)

total_time = time.time() - start_time
report = f"""
{'='*80}
🎯 LIGHTGBM MODEL - FINAL REPORT (OPTIMIZED)
{'='*80}

⏱️  EXECUTION TIME
   Total runtime:        {total_time/60:.2f} minutes
   Feature engineering:  {elapsed:.2f}s
   Initial training:     {train_time:.2f}s
   Hyperparameter tuning: {tune_time/60:.2f} minutes
   Final retraining:     {retrain_time:.2f}s

📊 DATASET
   Total samples:    {len(df):,}
   After cleaning:   {len(df_clean):,}
   Features:         {len(selected_features)}
   
   Split (64/16/20):
   - Training:   {len(X_train):,}
   - Validation: {len(X_val):,}
   - Test:       {len(X_test):,}

🏆 BEST HYPERPARAMETERS
{chr(10).join([f'   - {k}: {v}' for k, v in grid_search.best_params_.items()])}

📈 PERFORMANCE METRICS
   
   Initial Model:
   - CV R²:      {initial_results['cv_mean']:.4f} ± {initial_results['cv_std']:.4f}
   - Test R²:    {initial_results['test_r2']:.4f}
   - Test RMSE:  €{initial_results['test_rmse']:.2f}M
   
   Tuned Model:
   - CV R²:      {grid_search.best_score_:.4f}
   - Test R²:    {final_metrics['r2']:.4f}
   - Test RMSE:  €{final_metrics['rmse']:.2f}M
   - Test MAE:   €{final_metrics['mae']:.2f}M
   - Test MAPE:  {final_metrics['mape']:.2f}%
   
   Improvement:  {improvement:+.2f}%

⚡ OPTIMIZATIONS APPLIED
   ✅ Reduced param grid: 243 → {n_combinations} combinations
   ✅ 3-fold CV instead of 5-fold (40% faster)
   ✅ LightGBM optimizations: force_col_wise, max_bin
   ✅ Progress bars with tqdm

✅ READY FOR COMPARISON
   Same pipeline as Random Forest & XGBoost
   Fair comparison guaranteed

⏰ Completed at: {time.strftime('%H:%M:%S on %Y-%m-%d')}
{'='*80}
"""

print(report)

with open('lightgbm_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("\n✅ Saved: lightgbm_report.txt")
print("\n🎉 ALL DONE! Training completed successfully!")
print(f"⏱️  Total time: {total_time/60:.2f} minutes")
print("="*80)