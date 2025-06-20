import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.gradient_boosting import GradientBoostingClassifier

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics from first principles"""
    # Calculate accuracy
    accuracy = np.mean(y_true == y_pred)
    
    # Calculate confusion matrix
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(unique_classes)
    cm = np.zeros((n_classes, n_classes))
    
    for i in range(len(y_true)):
        cm[int(y_true[i]), int(y_pred[i])] += 1
    
    # Calculate precision, recall, and F1 for each class
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1 = np.zeros(n_classes)
    
    for i in range(n_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
    
    # Calculate weighted averages
    weights = np.bincount(y_true.astype(int)) / len(y_true)
    weighted_precision = np.sum(precision * weights)
    weighted_recall = np.sum(recall * weights)
    weighted_f1 = np.sum(f1 * weights)
    
    return {
        'accuracy': accuracy,
        'precision': weighted_precision,
        'recall': weighted_recall,
        'f1': weighted_f1,
        'confusion_matrix': cm
    }

def load_and_preprocess_data():
    """Load and preprocess the apple quality dataset"""
    print("Loading dataset...")
    # Load the dataset
    df = pd.read_csv('apple_quality.csv')
    
    # Remove any rows with missing values
    df = df.dropna()
    
    # Display basic information
    print("\nDataset Information:")
    print(f"Number of samples: {len(df)}")
    print(f"Number of features: {len(df.columns) - 1}")  # -1 for target
    print("\nFeature names:")
    print(df.columns.tolist())
    
    # Convert quality to binary (0 for bad, 1 for good)
    df['Quality'] = df['Quality'].map({'bad': 0, 'good': 1})
    
    # Display class distribution
    print("\nClass distribution:")
    class_counts = df['Quality'].value_counts()
    print(f"Class 0 (bad): {class_counts[0]} ({class_counts[0]/len(df):.2%})")
    print(f"Class 1 (good): {class_counts[1]} ({class_counts[1]/len(df):.2%})")
    
    # Split features and target
    feature_names = [col for col in df.columns if col not in ['A_id', 'Quality']]
    X = df[feature_names].copy()  # Select only feature columns
    y = df['Quality'].values
    
    # Convert to numpy arrays
    X = X.values.astype(float)
    
    # Split into train and test sets (80-20 split)
    n_samples = len(X)
    n_train = int(0.8 * n_samples)
    indices = np.random.permutation(n_samples)
    train_idx, test_idx = indices[:n_train], indices[n_train:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Balance the training set using random oversampling
    print("\nBalancing the training set...")
    class_counts_train = np.bincount(y_train.astype(int))
    n_minority = np.min(class_counts_train)
    n_majority = np.max(class_counts_train)
    majority_class = np.argmax(class_counts_train)
    minority_class = 1 - majority_class
    
    print(f"Original training set class distribution:")
    print(f"Class 0: {class_counts_train[0]} ({class_counts_train[0]/len(y_train):.2%})")
    print(f"Class 1: {class_counts_train[1]} ({class_counts_train[1]/len(y_train):.2%})")
    
    # Get indices of each class
    minority_indices = np.where(y_train == minority_class)[0]
    
    # Randomly oversample the minority class
    oversampled_indices = np.random.choice(minority_indices, size=n_majority - n_minority, replace=True)
    balanced_indices = np.concatenate([np.arange(len(y_train)), oversampled_indices])
    
    X_train = X_train[balanced_indices]
    y_train = y_train[balanced_indices]
    
    print(f"\nBalanced training set class distribution:")
    class_counts_balanced = np.bincount(y_train.astype(int))
    print(f"Class 0: {class_counts_balanced[0]} ({class_counts_balanced[0]/len(y_train):.2%})")
    print(f"Class 1: {class_counts_balanced[1]} ({class_counts_balanced[1]/len(y_train):.2%})")
    
    return X_train, X_test, y_train, y_test, feature_names

def add_feature_engineering(X, feature_names):
    """Add engineered features to improve model performance"""
    print("\nAdding engineered features...")
    X_engineered = X.copy()
    
    # Get feature indices
    feature_indices = {name: idx for idx, name in enumerate(feature_names)}
    
    # Add interaction terms between related features
    # Quality indicators
    sweetness_idx = feature_indices['Sweetness']
    acidity_idx = feature_indices['Acidity']
    crunchiness_idx = feature_indices['Crunchiness']
    juiciness_idx = feature_indices['Juiciness']
    ripeness_idx = feature_indices['Ripeness']
    
    # Physical characteristics
    size_idx = feature_indices['Size']
    weight_idx = feature_indices['Weight']
    
    # Add polynomial and interaction features
    X_engineered = np.column_stack([
        X_engineered,
        # Quality indicator interactions
        X[:, sweetness_idx] * X[:, acidity_idx],      # Sweetness-Acidity balance
        X[:, crunchiness_idx] * X[:, juiciness_idx],  # Texture profile
        X[:, ripeness_idx] * X[:, sweetness_idx],     # Ripeness-Sweetness relationship
        X[:, ripeness_idx] * X[:, acidity_idx],       # Ripeness-Acidity relationship
        
        # Physical characteristic interactions
        X[:, size_idx] * X[:, weight_idx],            # Density approximation
        
        # Polynomial features for key indicators
        X[:, sweetness_idx] ** 2,                     # Sweetness squared
        X[:, acidity_idx] ** 2,                       # Acidity squared
        X[:, ripeness_idx] ** 2,                      # Ripeness squared
        
        # Important ratios
        X[:, sweetness_idx] / (X[:, acidity_idx] + 1e-6),  # Sweet-Acid ratio
        X[:, crunchiness_idx] / (X[:, juiciness_idx] + 1e-6),  # Texture ratio
        X[:, size_idx] / (X[:, weight_idx] + 1e-6),  # Density ratio
        
        # Composite features
        (X[:, sweetness_idx] + X[:, ripeness_idx]) / 2,  # Average sweetness-ripeness
        (X[:, crunchiness_idx] + X[:, juiciness_idx]) / 2  # Average texture
    ])
    
    return X_engineered

def evaluate_model(X_train, X_test, y_train, y_test, params):
    """Evaluate model with given parameters"""
    print(f"\nTesting with parameters: {params}")
    
    # Train the model
    print("Training model...")
    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    print("\nResults:")
    print(f"Test accuracy: {metrics['accuracy']:.2%}")
    print(f"Test precision: {metrics['precision']:.2%}")
    print(f"Test recall: {metrics['recall']:.2%}")
    print(f"Test F1-score: {metrics['f1']:.2%}")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    return model, metrics['accuracy']

def main():
    # Load and preprocess data
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()
    
    # Add feature engineering
    print("\nAdding feature engineering...")
    X_train = add_feature_engineering(X_train, feature_names)
    X_test = add_feature_engineering(X_test, feature_names)
    
    # Define parameter combinations to try
    param_combinations = [
        # More trees, lower learning rate for better generalization
        {'n_estimators': 300, 'learning_rate': 0.01, 'max_depth': 4, 'min_samples_split': 10},
        
        # Balanced approach with deeper trees
        {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 6, 'min_samples_split': 5},
        
        # Aggressive learning with regularization
        {'n_estimators': 150, 'learning_rate': 0.1, 'max_depth': 5, 'min_samples_split': 15},
        
        # Many shallow trees
        {'n_estimators': 500, 'learning_rate': 0.01, 'max_depth': 3, 'min_samples_split': 10},
        
        # Medium depth, focused learning
        {'n_estimators': 250, 'learning_rate': 0.03, 'max_depth': 5, 'min_samples_split': 8}
    ]
    
    # Evaluate different parameter combinations
    best_accuracy = 0
    best_model = None
    best_params = None
    
    print("\nStarting parameter search...")
    for params in param_combinations:
        model, accuracy = evaluate_model(X_train, X_test, y_train, y_test, params)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_params = params
    
    print("\nBest performing model:")
    print(f"Parameters: {best_params}")
    print(f"Accuracy: {best_accuracy:.2%}")

if __name__ == "__main__":
    main() 