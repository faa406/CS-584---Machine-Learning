import numpy as np

class DecisionTree:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        
    def _gini_impurity(self, y):
        """Calculate Gini impurity for a set of labels"""
        if len(y) == 0:
            return 0
        # For regression trees (handling residuals), we use variance
        if np.issubdtype(y.dtype, np.floating):
            return np.var(y)
        # For classification trees, use Gini impurity
        p = np.bincount(y.astype(int)) / len(y)
        return 1 - np.sum(p ** 2)
    
    def _best_split(self, X, y):
        """Find the best split for a node"""
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = ~left_indices
                
                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue
                    
                gini = (len(y[left_indices]) * self._gini_impurity(y[left_indices]) +
                       len(y[right_indices]) * self._gini_impurity(y[right_indices])) / len(y)
                
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
                    
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree"""
        if (depth >= self.max_depth or 
            len(y) < self.min_samples_split or 
            len(np.unique(y)) == 1):
            return np.mean(y)  # Return mean for regression trees
            
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return np.mean(y)  # Return mean for regression trees
            
        left_indices = X[:, feature] <= threshold
        right_indices = ~left_indices
        
        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return {
            'feature': feature,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree
        }
    
    def fit(self, X, y):
        """Fit the decision tree to the training data"""
        self.tree = self._build_tree(X, y)
        
    def _predict_sample(self, x, tree):
        """Predict a single sample"""
        if isinstance(tree, dict):
            if x[tree['feature']] <= tree['threshold']:
                return self._predict_sample(x, tree['left'])
            else:
                return self._predict_sample(x, tree['right'])
        else:
            return tree
            
    def predict_proba(self, X):
        """Predict class probabilities"""
        return np.array([self._predict_sample(x, self.tree) for x in X])
    
    def predict(self, X):
        """Predict class labels"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1) 