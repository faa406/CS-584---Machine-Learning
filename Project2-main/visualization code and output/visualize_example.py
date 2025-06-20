import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Create visualization output directory if it doesn't exist
os.makedirs('visualization_output', exist_ok=True)

# Data from example_output.txt
parameter_combinations = [
    {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'min_samples_split': 2},
    {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 6, 'min_samples_split': 3},
    {'n_estimators': 50, 'learning_rate': 0.2, 'max_depth': 3, 'min_samples_split': 10},
    {'n_estimators': 150, 'learning_rate': 0.1, 'max_depth': 5, 'min_samples_split': 2},
    {'n_estimators': 300, 'learning_rate': 0.03, 'max_depth': 8, 'min_samples_split': 2},
    {'n_estimators': 75, 'learning_rate': 0.15, 'max_depth': 4, 'min_samples_split': 4}
]

# Add probability predictions for calibration curve
y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # Example true labels
y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.5])  # Example probabilities

cv_metrics = {
    'accuracy': [87.44, 88.19, 86.19, 88.94, 86.56, 87.94],
    'precision': [87.77, 88.93, 86.20, 89.43, 88.23, 88.40],
    'recall': [92.20, 92.10, 92.00, 92.81, 89.94, 92.28],
    'f1_score': [89.90, 90.46, 88.98, 91.05, 89.05, 90.27]
}

test_metrics = {
    'accuracy': [89.25, 91.00, 88.75, 92.00, 90.00, 89.75],
    'precision': [88.21, 91.85, 86.33, 92.34, 89.30, 88.62],
    'recall': [93.94, 92.64, 95.67, 93.94, 93.94, 94.37],
    'f1_score': [90.99, 92.24, 90.76, 93.13, 91.56, 91.40]
}

# Example feature importance data
feature_names = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5']
feature_importance = np.array([0.30, 0.25, 0.20, 0.15, 0.10])

# Example class distribution data
class_dist_before = np.array([300, 700])  # Before balancing
class_dist_after = np.array([500, 500])   # After balancing

# Set style
plt.style.use('default')
sns.set_style("whitegrid")
sns.set_palette("husl")

# 1. Precision-Recall Curve
plt.figure(figsize=(10, 6))
precision, recall, _ = precision_recall_curve(y_true, y_prob)
average_precision = average_precision_score(y_true, y_prob)
plt.plot(recall, precision, label=f'AP={average_precision:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('visualization_output/general_example_precision_recall_curve.png')
plt.close()

# 2. Calibration Curve
plt.figure(figsize=(10, 6))
prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
plt.plot(prob_pred, prob_true, marker='o', label='Model')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('visualization_output/general_example_calibration_curve.png')
plt.close()

# 3. Feature Importance
plt.figure(figsize=(12, 6))
sorted_idx = np.argsort(feature_importance)
plt.barh(range(len(feature_names)), feature_importance[sorted_idx])
plt.yticks(range(len(feature_names)), [feature_names[i] for i in sorted_idx])
plt.xlabel('Importance Score')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('visualization_output/general_example_feature_importance.png')
plt.close()

# 4. Probability Histogram
plt.figure(figsize=(10, 6))
plt.hist(y_prob[y_true == 0], bins=20, alpha=0.5, label='Class 0')
plt.hist(y_prob[y_true == 1], bins=20, alpha=0.5, label='Class 1')
plt.xlabel('Predicted Probability')
plt.ylabel('Count')
plt.title('Probability Distribution by Class')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('visualization_output/general_example_probability_histogram.png')
plt.close()

# 5. 2D PCA Projection
# Example data for PCA
X = np.random.randn(100, 5)  # 5 features, 100 samples
y = np.random.randint(0, 2, 100)  # Binary labels

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], label='Class 0', alpha=0.5)
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], label='Class 1', alpha=0.5)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('2D PCA Projection')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('visualization_output/general_example_pca_projection.png')
plt.close()

# 6. Class Balance Plot
plt.figure(figsize=(10, 6))
x = np.arange(2)
width = 0.35

plt.bar(x - width/2, class_dist_before, width, label='Before Balancing')
plt.bar(x + width/2, class_dist_after, width, label='After Balancing')
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution Before and After Balancing')
plt.xticks(x, ['Class 0', 'Class 1'])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('visualization_output/general_example_class_balance.png')
plt.close()

# Original visualizations
# 1. Cross-validation vs Test Performance
plt.figure(figsize=(12, 6))
x = np.arange(len(parameter_combinations))
width = 0.35

plt.bar(x - width/2, cv_metrics['accuracy'], width, label='CV Accuracy')
plt.bar(x + width/2, test_metrics['accuracy'], width, label='Test Accuracy')

plt.xlabel('Parameter Combinations')
plt.ylabel('Accuracy (%)')
plt.title('Cross-validation vs Test Accuracy')
plt.xticks(x, [f'Combination {i+1}' for i in range(len(parameter_combinations))])
plt.legend()
plt.tight_layout()
plt.savefig('visualization_output/general_example_cv_vs_test.png')
plt.close()

# 2. Parameter Effect Analysis
plt.figure(figsize=(15, 10))

# Learning Rate vs Performance
plt.subplot(2, 2, 1)
learning_rates = [p['learning_rate'] for p in parameter_combinations]
plt.scatter(learning_rates, cv_metrics['accuracy'], label='CV')
plt.scatter(learning_rates, test_metrics['accuracy'], label='Test')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy (%)')
plt.title('Learning Rate vs Accuracy')
plt.legend()

# Number of Estimators vs Performance
plt.subplot(2, 2, 2)
n_estimators = [p['n_estimators'] for p in parameter_combinations]
plt.scatter(n_estimators, cv_metrics['accuracy'], label='CV')
plt.scatter(n_estimators, test_metrics['accuracy'], label='Test')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy (%)')
plt.title('Number of Estimators vs Accuracy')
plt.legend()

# Max Depth vs Performance
plt.subplot(2, 2, 3)
max_depths = [p['max_depth'] for p in parameter_combinations]
plt.scatter(max_depths, cv_metrics['accuracy'], label='CV')
plt.scatter(max_depths, test_metrics['accuracy'], label='Test')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy (%)')
plt.title('Max Depth vs Accuracy')
plt.legend()

# Min Samples Split vs Performance
plt.subplot(2, 2, 4)
min_samples_splits = [p['min_samples_split'] for p in parameter_combinations]
plt.scatter(min_samples_splits, cv_metrics['accuracy'], label='CV')
plt.scatter(min_samples_splits, test_metrics['accuracy'], label='Test')
plt.xlabel('Min Samples Split')
plt.ylabel('Accuracy (%)')
plt.title('Min Samples Split vs Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('visualization_output/general_example_parameter_effects.png')
plt.close()

# 3. Best Model Metrics Comparison
plt.figure(figsize=(10, 6))
best_model_idx = 3  # Best performing model index

metrics = ['accuracy', 'precision', 'recall', 'f1_score']
cv_values = [cv_metrics[m][best_model_idx] for m in metrics]
test_values = [test_metrics[m][best_model_idx] for m in metrics]

x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, cv_values, width, label='Cross-validation')
plt.bar(x + width/2, test_values, width, label='Test')

plt.xlabel('Metrics')
plt.ylabel('Score (%)')
plt.title('Best Model Performance Metrics')
plt.xticks(x, metrics)
plt.legend()
plt.tight_layout()
plt.savefig('visualization_output/general_example_best_model_metrics.png')
plt.close()

# 4. Learning Curve Analysis
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(parameter_combinations) + 1), cv_metrics['accuracy'], 'o-', label='CV Accuracy')
plt.plot(range(1, len(parameter_combinations) + 1), test_metrics['accuracy'], 'o-', label='Test Accuracy')
plt.xlabel('Parameter Combination')
plt.ylabel('Accuracy (%)')
plt.title('Learning Curve Across Parameter Combinations')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('visualization_output/general_example_learning_curve.png')
plt.close() 