import numpy as np
import sys
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.gradient_boosting import GradientBoostingClassifier

def add_feature_engineering(X):
    """Add engineered features to improve model performance"""
    print("Adding engineered features...")
    # Original features
    X_engineered = X.copy()
    
    # Add interaction terms
    X_engineered = np.column_stack([
        X_engineered,
        X[:, 0] * X[:, 1],  # Interaction between first two features
        X[:, 2] ** 2,       # Quadratic term
        np.sin(X[:, 3]),    # Trigonometric transformation
        np.exp(X[:, 4])     # Exponential transformation
    ])
    
    return X_engineered

def cross_validate(X, y, params, n_splits=5):
    """Perform cross-validation to get more reliable accuracy estimate"""
    print(f"\nStarting {n_splits}-fold cross-validation...")
    fold_size = len(X) // n_splits
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for i in tqdm(range(n_splits), desc="Cross-validation progress"):
        # Split into train and validation
        val_start = i * fold_size
        val_end = (i + 1) * fold_size
        X_val = X[val_start:val_end]
        y_val = y[val_start:val_end]
        X_train = np.concatenate([X[:val_start], X[val_end:]])
        y_train = np.concatenate([y[:val_start], y[val_end:]])
        
        # Train and evaluate
        model = GradientBoostingClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        accuracies.append(np.mean(y_pred == y_val))
        precisions.append(precision_score(y_val, y_pred))
        recalls.append(recall_score(y_val, y_pred))
        f1_scores.append(f1_score(y_val, y_pred))
    
    return {
        'accuracy': np.mean(accuracies),
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'f1': np.mean(f1_scores),
        'std_accuracy': np.std(accuracies)
    }

def evaluate_model(X_train, X_test, y_train, y_test, params):
    """Evaluate model with given parameters"""
    print(f"\nTesting with parameters: {params}")
    
    # Cross-validation
    cv_results = cross_validate(X_train, y_train, params)
    print(f"Cross-validation results:")
    print(f"  Accuracy: {cv_results['accuracy']:.2%} (±{cv_results['std_accuracy']:.2%})")
    print(f"  Precision: {cv_results['precision']:.2%}")
    print(f"  Recall: {cv_results['recall']:.2%}")
    print(f"  F1-score: {cv_results['f1']:.2%}")
    
    # Final evaluation
    print("Training final model...")
    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)
    print("Making predictions...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    test_accuracy = np.mean(y_pred == y_test)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    
    print(f"Test set results:")
    print(f"  Accuracy: {test_accuracy:.2%}")
    print(f"  Precision: {test_precision:.2%}")
    print(f"  Recall: {test_recall:.2%}")
    print(f"  F1-score: {test_f1:.2%}")
    
    return {
        'model': model,
        'cv_accuracy': cv_results['accuracy'],
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1
    }

def main():
    print("Generating synthetic dataset...")
    # Generate a more complex synthetic dataset
    np.random.seed(42)
    n_samples = 2000  # Increased sample size
    X = np.random.randn(n_samples, 5)
    
    # Create a non-linear decision boundary
    y = ((X[:, 0] ** 2 + X[:, 1] ** 2 + 
          np.sin(X[:, 2]) + 
          np.exp(X[:, 3]) + 
          X[:, 4] > 2.5)).astype(int)
    
    # Add engineered features
    X = add_feature_engineering(X)
    
    # Split into train and test sets
    print("Splitting dataset into train and test sets...")
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Try different parameter combinations
    param_combinations = [
        # Original settings
        {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 4, 'min_samples_split': 5},
        # More trees, deeper trees
        {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 6, 'min_samples_split': 3},
        # Fewer trees, higher learning rate
        {'n_estimators': 50, 'learning_rate': 0.2, 'max_depth': 3, 'min_samples_split': 10},
        # Balanced approach
        {'n_estimators': 150, 'learning_rate': 0.1, 'max_depth': 5, 'min_samples_split': 2},
        # New combinations
        {'n_estimators': 300, 'learning_rate': 0.03, 'max_depth': 8, 'min_samples_split': 2},
        {'n_estimators': 75, 'learning_rate': 0.15, 'max_depth': 4, 'min_samples_split': 4},
    ]
    
    best_f1 = 0
    best_results = None
    
    print("\nStarting parameter search...")
    for i, params in enumerate(tqdm(param_combinations, desc="Parameter combinations")):
        print(f"\nTesting parameter combination {i+1}/{len(param_combinations)}")
        results = evaluate_model(X_train, X_test, y_train, y_test, params)
        if results['test_f1'] > best_f1:
            best_f1 = results['test_f1']
            best_results = results
    
    print("\nBest performing model:")
    print(f"Parameters: {params}")
    print(f"Cross-validation accuracy: {best_results['cv_accuracy']:.2%}")
    print(f"Test accuracy: {best_results['test_accuracy']:.2%}")
    print(f"Test precision: {best_results['test_precision']:.2%}")
    print(f"Test recall: {best_results['test_recall']:.2%}")
    print(f"Test F1-score: {best_results['test_f1']:.2%}")
    
    # Show detailed predictions from best model
    print("\nExample predictions from best model:")
    y_pred = best_results['model'].predict(X_test)
    y_proba = best_results['model'].predict_proba(X_test)
    for i in range(5):
        print(f"True class: {y_test[i]}, Predicted class: {y_pred[i]}, "
              f"Probability: {y_proba[i, 1]:.4f}")

if __name__ == "__main__":
    main() 