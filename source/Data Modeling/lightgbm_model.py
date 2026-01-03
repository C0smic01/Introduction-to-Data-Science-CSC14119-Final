import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import lightgbm as lgb
import joblib
import warnings
from tqdm.auto import tqdm
import time
warnings.filterwarnings('ignore')


class FootballPlayerValuePredictor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.selected_features = None
        self.model = None
        self.feature_cols = None
        
    def engineer_features(self, df):
        print("🔄 Feature Engineering Progress:")
        df_features = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        print("  ⏳ Log transformations...")
        skewed_features = []
        for col in tqdm(numeric_cols, desc="  ", leave=False):
            if col not in ['market_value', 'is_GK', 'is_DF', 'is_MF', 'is_FW']:
                skewness = abs(df_features[col].skew())
                if skewness > 1.0:
                    df_features[f'{col}_log'] = np.log1p(df_features[col])
                    skewed_features.append(col)
        
        print(f"  ✅ Created {len(skewed_features)} log features")
        print("  ⏳ Creating ratio features...")
        if 'goals' in df_features.columns and 'shots_per90' in df_features.columns:
            df_features['goals_per_shot'] = df_features['goals'] / df_features['shots_per90'].replace(0, 0.01)
        
        if 'passes_completed_per90' in df_features.columns and 'pass_completion_pct' in df_features.columns:
            df_features['pass_efficiency'] = df_features['passes_completed_per90'] * df_features['pass_completion_pct'] / 100
        
        if all(col in df_features.columns for col in ['interceptions_per90', 'blocks_per90']):
            df_features['defensive_contribution'] = df_features['interceptions_per90'] + df_features['blocks_per90']
        
        if all(col in df_features.columns for col in ['progressive_passes_per90', 'progressive_carries_per90']):
            df_features['total_progressive'] = df_features['progressive_passes_per90'] + df_features['progressive_carries_per90']
        
        print("  ⏳ Creating interaction features...")
        df_features['age_experience'] = df_features['age'] * np.log1p(df_features['minutes_played'])
        
        if 'minutes_played' in df_features.columns and 'appearances' in df_features.columns:
            df_features['minutes_per_game'] = df_features['minutes_played'] / df_features['appearances'].replace(0, 1)
        
        print("  ⏳ Creating polynomial features...")
        key_features = ['goals', 'assists', 'minutes_played']
        for feat in key_features:
            if feat in df_features.columns:
                df_features[f'{feat}_squared'] = df_features[feat] ** 2
        
        print("  ⏳ Encoding categorical variables...")
        categorical_cols = ['nationality', 'position', 'current_club', 'league']
        
        for col in ['nationality', 'current_club']:
            if col in df_features.columns:
                target_mean = df_features.groupby(col)['market_value'].mean()
                df_features[f'{col}_target_enc'] = df_features[col].map(target_mean)
                df_features[f'{col}_target_enc'].fillna(df_features['market_value'].mean(), inplace=True)
        
        for col in ['position', 'league']:
            if col in df_features.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_features[f'{col}_label_enc'] = self.label_encoders[col].fit_transform(df_features[col].astype(str))
                else:
                    df_features[f'{col}_label_enc'] = self.label_encoders[col].transform(df_features[col].astype(str))
        
        for col in categorical_cols:
            if col in df_features.columns:
                freq = df_features[col].value_counts()
                df_features[f'{col}_freq'] = df_features[col].map(freq)
        
        print(f"  ✅ Feature engineering completed! Total features: {len(df_features.columns)}")
        return df_features
    
    def select_features(self, df_features, corr_threshold=0.05):
        exclude_cols = ['market_value', 'position_category', 'nationality', 'position', 
                        'current_club', 'league']
        
        feature_cols = [col for col in df_features.columns 
                        if col not in exclude_cols 
                        and df_features[col].dtype in ['int64', 'float64']]
        
        X_temp = df_features[feature_cols].fillna(0)
        y_temp = df_features['market_value']
        
        correlations = {}
        for col in feature_cols:
            try:
                correlations[col] = abs(X_temp[col].corr(y_temp))
            except:
                correlations[col] = 0
        
        selected_features = [feat for feat, corr in correlations.items() if corr > corr_threshold]
        
        return selected_features, correlations
    
    def prepare_data(self, df, test_size=0.2, val_size=0.2):
        start_time = time.time()
        print("\n" + "="*60)
        print("📊 DATA PREPARATION PIPELINE")
        print("="*60)
        
        df_features = self.engineer_features(df)
        
        print("\n⏳ Selecting features by correlation...")
        self.selected_features, correlations = self.select_features(df_features)
        print(f"  ✅ Selected {len(self.selected_features)} features")
        
        print("\n⏳ Removing outliers...")
        Q1 = df_features['market_value'].quantile(0.01)
        Q3 = df_features['market_value'].quantile(0.99)
        df_clean = df_features[(df_features['market_value'] >= Q1) & 
                                (df_features['market_value'] <= Q3)].copy()
        print(f"  ✅ Kept {len(df_clean):,}/{len(df_features):,} samples ({len(df_clean)/len(df_features)*100:.1f}%)")
        
        print("\n⏳ Splitting data (Train/Val/Test: 64%/16%/20%)...")
        X = df_clean[self.selected_features].fillna(0)
        y = df_clean['market_value']
        y_log = np.log1p(y)
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_log, test_size=test_size, random_state=self.random_state, shuffle=True
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=self.random_state, shuffle=True
        )
        
        print("\n⏳ Scaling features with StandardScaler...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        elapsed = time.time() - start_time
        print(f"\n✅ Data preparation completed in {elapsed:.2f}s")
        print("="*60)
        
        return (X_train_scaled, X_val_scaled, X_test_scaled, 
                y_train, y_val, y_test, df_clean, correlations)
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        start_time = time.time()
        print("\n⏳ Training LightGBM model...")
        
        self.model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1
        )
        
        self.model.fit(X_train, y_train)
        
        elapsed = time.time() - start_time
        print(f"✅ Training completed in {elapsed:.2f}s")
        
        return self.model
    
    def tune_hyperparameters(self, X_train, y_train, cv=5):
        start_time = time.time()
        print("\n" + "="*60)
        print("🔍 HYPERPARAMETER TUNING - GridSearchCV")
        print("="*60)
        
        param_grid = {
            'n_estimators': [150, 200, 250],
            'learning_rate': [0.03, 0.05, 0.07],
            'max_depth': [5, 6, 7],
            'num_leaves': [25, 31, 40],
            'min_child_samples': [15, 20, 25]
        }
        
        total_combinations = np.prod([len(v) for v in param_grid.values()])
        print(f"\n📊 Search space: {total_combinations} combinations")
        print(f"📊 Cross-validation: {cv}-fold")
        print(f"📊 Total fits: {total_combinations * cv}")
        print(f"\n⏳ This may take 5-15 minutes depending on dataset size...")
        print(f"⏳ Started at: {time.strftime('%H:%M:%S')}\n")
        
        base_model = lgb.LGBMRegressor(
            random_state=self.random_state, 
            n_jobs=-1, 
            verbose=-1
        )
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring='r2',
            n_jobs=-1,
            verbose=2
        )
        
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        
        elapsed = time.time() - start_time
        print(f"\n✅ Grid search completed in {elapsed/60:.2f} minutes ({elapsed:.1f}s)")
        print(f"✅ Finished at: {time.strftime('%H:%M:%S')}")
        print("="*60)
        
        return grid_search.best_params_, grid_search.best_score_, grid_search
    
    def evaluate(self, X_test, y_test):
        y_test_pred_log = self.model.predict(X_test)
        y_test_pred = np.expm1(y_test_pred_log)
        y_test_orig = np.expm1(y_test)
        
        metrics = {
            'r2': r2_score(y_test, y_test_pred_log),
            'mse': mean_squared_error(y_test_orig, y_test_pred),
            'rmse': np.sqrt(mean_squared_error(y_test_orig, y_test_pred)),
            'mae': mean_absolute_error(y_test_orig, y_test_pred),
            'mape': np.mean(np.abs((y_test_orig - y_test_pred) / y_test_orig)) * 100
        }
        
        return metrics, y_test_pred_log
    
    def cross_validate(self, X_train, y_train, cv=5):
        start_time = time.time()
        print(f"\n⏳ Running {cv}-fold cross-validation...")
        
        kfold = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(self.model, X_train, y_train, 
                                    cv=kfold, scoring='r2', n_jobs=-1)
        
        elapsed = time.time() - start_time
        print(f"✅ Cross-validation completed in {elapsed:.2f}s")
        
        return cv_scores.mean(), cv_scores.std(), cv_scores
    
    def get_feature_importance(self, top_n=20):
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            return None
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        
        feature_importance = {
            'features': [self.selected_features[i] for i in indices],
            'importances': importances[indices]
        }
        
        return feature_importance
    
    def save(self, model_path='lightgbm_model.pkl', 
             scaler_path='scaler.pkl', 
             features_path='selected_features.pkl'):
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.selected_features, features_path)
    
    def load(self, model_path='lightgbm_model.pkl', 
             scaler_path='scaler.pkl', 
             features_path='selected_features.pkl'):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.selected_features = joblib.load(features_path)
