import numpy as np
import os
import sys
import json
from datetime import datetime

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from model.LassoHomotopy import LassoHomotopyModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

def load_and_preprocess_data(data_path):
    """Load and preprocess the dataset."""
    data = np.loadtxt(data_path, delimiter=',', skiprows=1)
    X = data[:, :-1]
    y = data[:, -1]
    
    # Preprocess data
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1  # Avoid division by zero
    X_scaled = (X - X_mean) / X_std
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_scaled = (y - y_mean) / y_std
    
    return X_scaled, y_scaled, X_mean, X_std, y_mean, y_std

def plot_regularization_path(model, title, save_path):
    """Plot the regularization path showing coefficient changes."""
    plt.figure(figsize=(12, 8))
    coef_path = np.array(model.coef_path_)
    lambda_path = np.array(model.lambda_path_)
    
    for i in range(coef_path.shape[1]):
        plt.plot(lambda_path, coef_path[:, i], label=f'Feature {i+1}')
    
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('Coefficient Value')
    plt.title(f'Regularization Path - {title}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{title.lower().replace(" ", "_")}_regularization_path.png'))
    plt.close()

def plot_cross_validation_scores(model, X, y, title, save_path):
    """Plot cross-validation scores for different lambda values."""
    cv_results = model.cross_validate(X, y, n_folds=5, n_lambda=20)
    
    plt.figure(figsize=(10, 6))
    plt.plot(cv_results['lambda_values'], cv_results['cv_scores'], 'b-', label='CV Score')
    plt.axvline(x=cv_results['best_lambda'], color='r', linestyle='--', label=f'Best Lambda: {cv_results["best_lambda"]:.4f}')
    
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('Cross-Validation Score (R²)')
    plt.title(f'Cross-Validation Scores - {title}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{title.lower().replace(" ", "_")}_cv_scores.png'))
    plt.close()

def plot_feature_importance_with_ci(model, X, y, title, save_path):
    """Plot feature importance with confidence intervals."""
    # Get feature importance scores from multiple cross-validation folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    importance_scores = []
    
    for train_idx, _ in kf.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        model.fit(X_train, y_train)
        importance_scores.append(np.abs(model.coef_))
    
    importance_scores = np.array(importance_scores)
    mean_importance = np.mean(importance_scores, axis=0)
    std_importance = np.std(importance_scores, axis=0)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(mean_importance)), mean_importance)
    plt.errorbar(range(len(mean_importance)), mean_importance, std_importance, fmt='none', color='red', capsize=5)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.title(f'Feature Importance with Confidence Intervals - {title}')
    plt.xlabel('Feature Index')
    plt.ylabel('Absolute Coefficient Value')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{title.lower().replace(" ", "_")}_feature_importance_ci.png'))
    plt.close()

def plot_correlation_matrix(X, title, save_path):
    """Plot correlation matrix of features."""
    plt.figure(figsize=(12, 10))
    correlation_matrix = np.corrcoef(X.T)
    plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar()
    
    # Add correlation values
    for i in range(correlation_matrix.shape[0]):
        for j in range(correlation_matrix.shape[1]):
            plt.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                    ha='center', va='center', color='black')
    
    plt.title(f'Feature Correlation Matrix - {title}')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Index')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{title.lower().replace(" ", "_")}_correlation_matrix.png'))
    plt.close()

def plot_learning_curve(model, X, y, title, save_path):
    """Plot learning curve showing model performance vs training size."""
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)  # Suppress R² warnings
    
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes_abs, train_scores, val_scores = [], [], []
    
    for size in train_sizes:
        n_samples = int(len(X) * size)
        if n_samples < 2:  # Skip if we have less than 2 samples
            continue
            
        train_sizes_abs.append(n_samples)
        
        # Perform cross-validation for this training size
        kf = KFold(n_splits=min(5, n_samples-1), shuffle=True, random_state=42)  # Adjust n_splits for small datasets
        fold_train_scores = []
        fold_val_scores = []
        
        for train_idx, val_idx in kf.split(X[:n_samples]):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            if len(X_train) < 2 or len(X_val) < 2:  # Skip folds with too few samples
                continue
                
            model.fit(X_train, y_train)
            
            # Calculate scores
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            fold_train_scores.append(r2_score(y_train, train_pred))
            fold_val_scores.append(r2_score(y_val, val_pred))
        
        if fold_train_scores and fold_val_scores:  # Only add scores if we have valid results
            train_scores.append(np.mean(fold_train_scores))
            val_scores.append(np.mean(fold_val_scores))
    
    if not train_scores or not val_scores:  # If we don't have any valid scores, return
        print(f"Warning: Could not generate learning curve for {title} due to insufficient data")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes_abs, train_scores, 'o-', label='Training Score')
    plt.plot(train_sizes_abs, val_scores, 'o-', label='Validation Score')
    plt.xlabel('Training Examples')
    plt.ylabel('R² Score')
    plt.title(f'Learning Curve - {title}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{title.lower().replace(" ", "_")}_learning_curve.png'))
    plt.close()
    
    # Restore warning settings
    warnings.filterwarnings('default')

def plot_residuals_vs_fitted(y_true, y_pred, title, save_path):
    """Plot residuals against fitted values."""
    plt.figure(figsize=(10, 6))
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residuals vs Fitted Values - {title}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{title.lower().replace(" ", "_")}_residuals_vs_fitted.png'))
    plt.close()

def plot_qq_plot(y_true, y_pred, title, save_path):
    """Plot Q-Q plot of residuals."""
    from scipy import stats
    plt.figure(figsize=(10, 6))
    residuals = y_true - y_pred
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f'Q-Q Plot of Residuals - {title}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{title.lower().replace(" ", "_")}_qq_plot.png'))
    plt.close()

def plot_feature_selection_path(model, title, save_path):
    """Plot how features are selected/deselected along the regularization path."""
    plt.figure(figsize=(12, 6))
    active_features = np.array([np.sum(np.abs(coef) > 1e-10) for coef in model.coef_path_])
    plt.plot(model.lambda_path_, active_features, 'b-', label='Active Features')
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('Number of Active Features')
    plt.title(f'Feature Selection Path - {title}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{title.lower().replace(" ", "_")}_feature_selection_path.png'))
    plt.close()

def plot_bias_variance_tradeoff(model, X, y, title, save_path):
    """Plot bias-variance tradeoff."""
    from sklearn.model_selection import train_test_split
    n_samples = len(X)
    train_sizes = np.linspace(0.1, 0.9, 9)  # Changed to avoid train_size=1.0
    train_sizes_abs = [int(n_samples * size) for size in train_sizes]
    
    train_scores = []
    val_scores = []
    
    for size in train_sizes_abs:
        size_train_scores = []
        size_val_scores = []
        
        for _ in range(5):  # Repeat 5 times for each size
            X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=size/n_samples, random_state=42)
            model.fit(X_train, y_train)
            
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            size_train_scores.append(r2_score(y_train, train_pred))
            size_val_scores.append(r2_score(y_val, val_pred))
        
        train_scores.append(np.mean(size_train_scores))
        val_scores.append(np.mean(size_val_scores))
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes_abs, train_scores, 'o-', label='Training Score')
    plt.plot(train_sizes_abs, val_scores, 'o-', label='Validation Score')
    plt.xlabel('Training Examples')
    plt.ylabel('R² Score')
    plt.title(f'Bias-Variance Tradeoff - {title}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{title.lower().replace(" ", "_")}_bias_variance.png'))
    plt.close()

def train_model(X, y):
    """Train model with optimized parameters."""
    # Try different lambda_max values
    lambda_values = [0.01, 0.05, 0.1, 0.5, 1.0]
    best_model = None
    best_r2 = -float('inf')
    
    for lambda_max in lambda_values:
        model = LassoHomotopyModel(lambda_max=lambda_max, fit_intercept=True)
        model.fit(X, y)
        
        # Evaluate model
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
    
    return best_model

def evaluate_model(model, X, y, X_mean, X_std, y_mean, y_std, title, save_path):
    """Evaluate model performance and create visualizations."""
    try:
        # Get predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        active_features = np.sum(model.active_set_)
        total_features = X.shape[1]
        
        # Print results
        print(f"\n{title}:")
        print(f"MSE: {mse:.4f}")
        print(f"R²: {r2*100:.2f}%")
        print(f"Active Features: {active_features}/{total_features}")
        print(f"Best Lambda: {model.lambda_max:.4f}")
        
        # Create detailed plots
        plot_regularization_path(model, title, save_path)
        plot_cross_validation_scores(model, X, y, title, save_path)
        plot_feature_importance_with_ci(model, X, y, title, save_path)
        plot_correlation_matrix(X, title, save_path)
        plot_learning_curve(model, X, y, title, save_path)
        plot_residuals_vs_fitted(y, y_pred, title, save_path)
        plot_qq_plot(y, y_pred, title, save_path)
        plot_feature_selection_path(model, title, save_path)
        plot_bias_variance_tradeoff(model, X, y, title, save_path)
        
    except Exception as e:
        print(f"Error in evaluation: {str(e)}")

def main():
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join('LassoHomotopy', 'tests', 'results', timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load datasets
    small_data_path = os.path.join('LassoHomotopy', 'tests', 'small_test.csv')
    collinear_data_path = os.path.join('LassoHomotopy', 'tests', 'collinear_data.csv')
    
    # Load and preprocess data
    X_small, y_small, X_mean_small, X_std_small, y_mean_small, y_std_small = load_and_preprocess_data(small_data_path)
    X_collinear, y_collinear, X_mean_collinear, X_std_collinear, y_mean_collinear, y_std_collinear = load_and_preprocess_data(collinear_data_path)
    
    # Train models with optimized parameters
    print("\nTraining models with optimized parameters...")
    model_small = train_model(X_small, y_small)
    model_collinear = train_model(X_collinear, y_collinear)
    
    # Evaluate models
    print("\nModel Evaluation Results:")
    print("=" * 50)
    
    # Small dataset results
    evaluate_model(model_small, X_small, y_small, X_mean_small, X_std_small, y_mean_small, y_std_small, "Small Dataset", results_dir)
    
    # Collinear dataset results
    evaluate_model(model_collinear, X_collinear, y_collinear, X_mean_collinear, X_std_collinear, y_mean_collinear, y_std_collinear, "Collinear Dataset", results_dir)
    
    print(f"\nResults and plots saved to: {results_dir}")

if __name__ == "__main__":
    main() 