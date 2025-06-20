import numpy as np
from .decision_tree import DecisionTree

class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.initial_prediction = None
        
    def _sigmoid(self, x):
        """Sigmoid function for probability estimation"""
        return 1 / (1 + np.exp(-x))
        
    def _log_loss_gradient(self, y, p):
        """Compute the gradient of the log loss"""
        return y - p
        
    def fit(self, X, y):
        """Fit the gradient boosting model"""
        # Initialize with the log-odds of the positive class
        pos_prob = np.mean(y)
        self.initial_prediction = np.log(pos_prob / (1 - pos_prob))
        current_prediction = np.full(len(y), self.initial_prediction)
        
        for _ in range(self.n_estimators):
            # Compute pseudo-residuals
            p = self._sigmoid(current_prediction)
            residuals = self._log_loss_gradient(y, p)
            
            # Fit a tree to the residuals
            tree = DecisionTree(max_depth=self.max_depth, 
                              min_samples_split=self.min_samples_split)
            tree.fit(X, residuals)
            
            # Update predictions
            tree_prediction = tree.predict_proba(X)  # Get predictions
            current_prediction += self.learning_rate * tree_prediction
            
            self.trees.append(tree)
            
    def predict_proba(self, X):
        """Predict class probabilities"""
        # Start with initial prediction
        predictions = np.full(len(X), self.initial_prediction)
        
        # Add predictions from all trees
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict_proba(X)
            
        # Convert to probabilities using sigmoid
        probabilities = self._sigmoid(predictions)
        return np.column_stack([1 - probabilities, probabilities])
        
    def predict(self, X):
        """Predict class labels"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1) 