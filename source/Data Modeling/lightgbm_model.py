import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class LightGBMTrainer:
    def __init__(self, param_grid=None, cv_folds=5, random_state=42):
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.model = None
        self.best_params = None
        self.cv_scores = None
        self.param_grid = param_grid or {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 50, 70],
            'min_child_samples': [20, 30, 40],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
    
    def train_with_gridsearch(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        base_model = lgb.LGBMRegressor(
            random_state=self.random_state,
            n_jobs=-1,
            force_col_wise=True
        )
        
        kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=self.param_grid,
            cv=kfold,
            scoring='r2',
            n_jobs=-1,
            verbose=1 if verbose else 0
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        self.cv_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=kfold, scoring='r2', n_jobs=-1
        )
        
        if verbose:
            print(f"\nBest Parameters: {self.best_params}")
            print(f"CV R² Score: {self.cv_scores.mean():.4f} ± {self.cv_scores.std():.4f}")
        
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
            if verbose:
                print(f"Validation R² Score: {val_score:.4f}")
        
        return self
    
    def train_with_early_stopping(self, X_train, y_train, X_val, y_val, 
                                   n_estimators=1000, early_stopping_rounds=50, verbose=True):
        if self.best_params is None:
            params = {
                'max_depth': 7,
                'learning_rate': 0.05,
                'num_leaves': 50,
                'min_child_samples': 30,
                'subsample': 0.9,
                'colsample_bytree': 0.9
            }
        else:
            params = self.best_params.copy()
            params.pop('n_estimators', None)
        
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
            force_col_wise=True,
            **params
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=verbose)]
        )
        
        return self
    
    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        return metrics, y_pred
    
    def get_feature_importance(self, feature_names, top_n=20):
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        return pd.DataFrame({
            'feature': [feature_names[i] for i in indices],
            'importance': importances[indices]
        })
    
    def save(self, filepath):
        joblib.dump({
            'model': self.model,
            'best_params': self.best_params,
            'cv_scores': self.cv_scores,
            'param_grid': self.param_grid
        }, filepath)
    
    @classmethod
    def load(cls, filepath):
        data = joblib.load(filepath)
        trainer = cls(param_grid=data['param_grid'])
        trainer.model = data['model']
        trainer.best_params = data['best_params']
        trainer.cv_scores = data['cv_scores']
        return trainer


def plot_predictions(y_true, y_pred, title="Actual vs Predicted", save_path=None):
    plt.figure(figsize=(10, 6))
    
    plt.scatter(y_true, y_pred, alpha=0.5, s=30)
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Value (M€)', fontsize=12)
    plt.ylabel('Predicted Value (M€)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    plt.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = €{rmse:.2f}M',
             transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_residuals(y_true, y_pred, save_path=None):
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0].axvline(0, color='red', linestyle='--', lw=2)
    axes[0].set_xlabel('Residuals (M€)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Residuals Distribution', fontsize=13, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    axes[1].scatter(y_pred, residuals, alpha=0.5, s=30)
    axes[1].axhline(0, color='red', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Value (M€)', fontsize=11)
    axes[1].set_ylabel('Residuals (M€)', fontsize=11)
    axes[1].set_title('Residuals vs Predicted', fontsize=13, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title('Q-Q Plot', fontsize=13, fontweight='bold')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_importance(importance_df, top_n=20, save_path=None):
    plt.figure(figsize=(10, 8))
    
    importance_df_top = importance_df.head(top_n)
    
    plt.barh(range(len(importance_df_top)), importance_df_top['importance'], alpha=0.7, color='steelblue')
    plt.yticks(range(len(importance_df_top)), importance_df_top['feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(alpha=0.3, axis='x')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def print_metrics(metrics, cv_scores=None):
    print("=" * 80)
    print("MODEL EVALUATION METRICS")
    print("=" * 80)
    print(f"\nTest Set Performance:")
    print(f"  R² Score:  {metrics['r2']:.4f}")
    print(f"  MSE:       €{metrics['mse']:.2f}M²")
    print(f"  RMSE:      €{metrics['rmse']:.2f}M")
    print(f"  MAE:       €{metrics['mae']:.2f}M")
    print(f"  MAPE:      {metrics['mape']:.2f}%")
    
    if cv_scores is not None:
        print(f"\nCross-Validation Performance:")
        print(f"  CV R² Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  Min CV R²:   {cv_scores.min():.4f}")
        print(f"  Max CV R²:   {cv_scores.max():.4f}")
    
    print("=" * 80)
