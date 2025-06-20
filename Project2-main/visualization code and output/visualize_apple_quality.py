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

# Data from apple_quality_output.txt
parameter_combinations = [
    {'n_estimators': 300, 'learning_rate': 0.01, 'max_depth': 4, 'min_samples_split': 10},
    {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 6, 'min_samples_split': 5},
    {'n_estimators': 150, 'learning_rate': 0.1, 'max_depth': 5, 'min_samples_split': 15},
    {'n_estimators': 500, 'learning_rate': 0.01, 'max_depth': 3, 'min_samples_split': 10},
    {'n_estimators': 250, 'learning_rate': 0.03, 'max_depth': 5, 'min_samples_split': 8}
]

# Add probability predictions for calibration curve
y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # Example true labels
y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.5])  # Example probabilities

metrics = {
    'accuracy': [80.62, 88.00, 86.50, 81.62, 86.50],
    'precision': [80.64, 88.01, 86.50, 81.63, 86.50],
    'recall': [80.62, 88.00, 86.50, 81.62, 86.50],
    'f1_score': [80.62, 88.00, 86.50, 81.62, 86.50]
}

confusion_matrices = [
    np.array([[318, 82], [73, 327]]),
    np.array([[349, 51], [45, 355]]),
    np.array([[344, 56], [52, 348]]),
    np.array([[323, 77], [70, 330]]),
    np.array([[346, 54], [54, 346]])
]

# Example feature importance data
feature_names = ['Sweetness', 'Acidity', 'Crunchiness', 'Juiciness', 'Ripeness', 'Size', 'Weight']
feature_importance = np.array([0.25, 0.20, 0.15, 0.15, 0.10, 0.08, 0.07])

# Example class distribution data
class_dist_before = np.array([400, 600])  # Before balancing
class_dist_after = np.array([600, 600])   # After balancing

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
plt.savefig('visualization_output/apple_quality_precision_recall_curve.png')
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
plt.savefig('visualization_output/apple_quality_calibration_curve.png')
plt.close()

# 3. Feature Importance
plt.figure(figsize=(12, 6))
sorted_idx = np.argsort(feature_importance)
plt.barh(range(len(feature_names)), feature_importance[sorted_idx])
plt.yticks(range(len(feature_names)), [feature_names[i] for i in sorted_idx])
plt.xlabel('Importance Score')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('visualization_output/apple_quality_feature_importance.png')
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
plt.savefig('visualization_output/apple_quality_probability_histogram.png')
plt.close()

# 5. 2D PCA Projection
# Example data for PCA
X = np.random.randn(100, 7)  # 7 features, 100 samples
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
plt.savefig('visualization_output/apple_quality_pca_projection.png')
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
plt.savefig('visualization_output/apple_quality_class_balance.png')
plt.close()

# Original visualizations
# 1. Parameter Performance Comparison
plt.figure(figsize=(12, 6))
x = np.arange(len(parameter_combinations))
width = 0.2

for i, (metric, values) in enumerate(metrics.items()):
    plt.bar(x + i*width, values, width, label=metric)

plt.xlabel('Parameter Combinations')
plt.ylabel('Score (%)')
plt.title('Model Performance Across Different Parameter Combinations')
plt.xticks(x + width*1.5, [f'Combination {i+1}' for i in range(len(parameter_combinations))])
plt.legend()
plt.tight_layout()
plt.savefig('visualization_output/apple_quality_parameter_performance.png')
plt.close()

# 2. Best Model Confusion Matrix
best_cm = confusion_matrices[1]  # Using the best performing model
plt.figure(figsize=(8, 6))
sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Bad', 'Good'],
            yticklabels=['Bad', 'Good'])
plt.title('Confusion Matrix - Best Performing Model')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('visualization_output/apple_quality_confusion_matrix.png')
plt.close()

# 3. Parameter Effect Analysis
plt.figure(figsize=(15, 10))

# Learning Rate vs Accuracy
plt.subplot(2, 2, 1)
learning_rates = [p['learning_rate'] for p in parameter_combinations]
plt.scatter(learning_rates, metrics['accuracy'])
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy (%)')
plt.title('Learning Rate vs Accuracy')

# Number of Estimators vs Accuracy
plt.subplot(2, 2, 2)
n_estimators = [p['n_estimators'] for p in parameter_combinations]
plt.scatter(n_estimators, metrics['accuracy'])
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy (%)')
plt.title('Number of Estimators vs Accuracy')

# Max Depth vs Accuracy
plt.subplot(2, 2, 3)
max_depths = [p['max_depth'] for p in parameter_combinations]
plt.scatter(max_depths, metrics['accuracy'])
plt.xlabel('Max Depth')
plt.ylabel('Accuracy (%)')
plt.title('Max Depth vs Accuracy')

# Min Samples Split vs Accuracy
plt.subplot(2, 2, 4)
min_samples_splits = [p['min_samples_split'] for p in parameter_combinations]
plt.scatter(min_samples_splits, metrics['accuracy'])
plt.xlabel('Min Samples Split')
plt.ylabel('Accuracy (%)')
plt.title('Min Samples Split vs Accuracy')

plt.tight_layout()
plt.savefig('visualization_output/apple_quality_parameter_effects.png')
plt.close()

# 4. Metrics Comparison for Best Model
plt.figure(figsize=(8, 6))
metrics_best = {k: v[1] for k, v in metrics.items()}  # Best model metrics
plt.bar(metrics_best.keys(), metrics_best.values())
plt.title('Performance Metrics - Best Model')
plt.ylabel('Score (%)')
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig('visualization_output/apple_quality_best_model_metrics.png')
plt.close() 