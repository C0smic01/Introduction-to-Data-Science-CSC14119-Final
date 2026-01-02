import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


class FootballDataPreprocessor:
    def __init__(self, corr_threshold=0.05):
        self.corr_threshold = corr_threshold
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.target_encodings = {}
        self.frequency_encodings = {}
        self.selected_features = None
        self.is_fitted = False
        
    def create_features(self, df):
        df_features = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            if col not in ['market_value', 'is_GK', 'is_DF', 'is_MF', 'is_FW']:
                skewness = abs(df_features[col].skew())
                if skewness > 1.0:
                    df_features[f'{col}_log'] = np.log1p(df_features[col])
        
        if 'goals' in df_features.columns and 'shots_per90' in df_features.columns:
            df_features['goals_per_shot'] = df_features['goals'] / df_features['shots_per90'].replace(0, 0.01)
        
        if 'passes_completed_per90' in df_features.columns and 'pass_completion_pct' in df_features.columns:
            df_features['pass_efficiency'] = df_features['passes_completed_per90'] * df_features['pass_completion_pct'] / 100
        
        if all(col in df_features.columns for col in ['interceptions_per90', 'blocks_per90']):
            df_features['defensive_contribution'] = df_features['interceptions_per90'] + df_features['blocks_per90']
        
        if all(col in df_features.columns for col in ['progressive_passes_per90', 'progressive_carries_per90']):
            df_features['total_progressive'] = df_features['progressive_passes_per90'] + df_features['progressive_carries_per90']
        
        df_features['age_experience'] = df_features['age'] * np.log1p(df_features['minutes_played'])
        
        if 'minutes_played' in df_features.columns and 'appearances' in df_features.columns:
            df_features['minutes_per_game'] = df_features['minutes_played'] / df_features['appearances'].replace(0, 1)
        
        key_features = ['goals', 'assists', 'minutes_played']
        for feat in key_features:
            if feat in df_features.columns:
                df_features[f'{feat}_squared'] = df_features[feat] ** 2
        
        return df_features
    
    def encode_categoricals(self, df_features, is_training=True):
        for col in ['nationality', 'current_club']:
            if col in df_features.columns:
                if is_training:
                    target_mean = df_features.groupby(col)['market_value'].mean()
                    self.target_encodings[col] = target_mean
                else:
                    target_mean = self.target_encodings[col]
                
                df_features[f'{col}_target_enc'] = df_features[col].map(target_mean)
                df_features[f'{col}_target_enc'].fillna(df_features['market_value'].mean() if is_training else 0, inplace=True)
        
        for col in ['position', 'league']:
            if col in df_features.columns:
                if is_training:
                    self.label_encoders[col] = LabelEncoder()
                    df_features[f'{col}_label_enc'] = self.label_encoders[col].fit_transform(df_features[col].astype(str))
                else:
                    df_features[f'{col}_label_enc'] = self.label_encoders[col].transform(df_features[col].astype(str))
        
        categorical_cols = ['nationality', 'position', 'current_club', 'league']
        for col in categorical_cols:
            if col in df_features.columns:
                if is_training:
                    freq = df_features[col].value_counts()
                    self.frequency_encodings[col] = freq
                else:
                    freq = self.frequency_encodings[col]
                
                df_features[f'{col}_freq'] = df_features[col].map(freq)
                df_features[f'{col}_freq'].fillna(0, inplace=True)
        
        return df_features
    
    def select_features(self, X, y, is_training=True):
        if is_training:
            correlations = {}
            for col in X.columns:
                try:
                    correlations[col] = abs(X[col].corr(y))
                except:
                    correlations[col] = 0
            
            self.selected_features = [feat for feat, corr in correlations.items() 
                                     if corr > self.corr_threshold]
        
        return X[self.selected_features]
    
    def fit(self, df):
        df_features = self.create_features(df)
        df_features = self.encode_categoricals(df_features, is_training=True)
        
        exclude_cols = ['market_value', 'position_category', 'nationality', 'position', 
                       'current_club', 'league']
        
        feature_cols = [col for col in df_features.columns 
                       if col not in exclude_cols 
                       and df_features[col].dtype in ['int64', 'float64']]
        
        X = df_features[feature_cols].fillna(0)
        y = df_features['market_value']
        
        X = self.select_features(X, y, is_training=True)
        
        self.scaler.fit(X)
        self.is_fitted = True
        
        return self
    
    def transform(self, df, scale=True):
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        df_features = self.create_features(df)
        df_features = self.encode_categoricals(df_features, is_training=False)
        
        X = df_features[self.selected_features].fillna(0)
        
        if scale:
            X_scaled = self.scaler.transform(X)
            return pd.DataFrame(X_scaled, columns=self.selected_features, index=X.index)
        
        return X
    
    def fit_transform(self, df, scale=True):
        self.fit(df)
        return self.transform(df, scale=scale)
    
    def save(self, filepath):
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'target_encodings': self.target_encodings,
            'frequency_encodings': self.frequency_encodings,
            'selected_features': self.selected_features,
            'corr_threshold': self.corr_threshold,
            'is_fitted': self.is_fitted
        }, filepath)
    
    @classmethod
    def load(cls, filepath):
        data = joblib.load(filepath)
        preprocessor = cls(corr_threshold=data['corr_threshold'])
        preprocessor.scaler = data['scaler']
        preprocessor.label_encoders = data['label_encoders']
        preprocessor.target_encodings = data['target_encodings']
        preprocessor.frequency_encodings = data['frequency_encodings']
        preprocessor.selected_features = data['selected_features']
        preprocessor.is_fitted = data['is_fitted']
        return preprocessor


def remove_outliers(df, columns=None, n_std=3):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    
    for col in columns:
        if col == 'market_value':
            continue
        
        mean = df_clean[col].mean()
        std = df_clean[col].std()
        lower_bound = mean - n_std * std
        upper_bound = mean + n_std * std
        
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean
