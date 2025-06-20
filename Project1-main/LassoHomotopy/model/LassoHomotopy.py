import numpy as np
from typing import Tuple, List, Optional, Dict
import warnings
import matplotlib.pyplot as plt

class LassoHomotopyModel:
    """Implementation of LASSO regularized regression using the Homotopy Method."""
    
    def __init__(self, lambda_max=1.0, fit_intercept=True):
        """Initialize LassoHomotopy model."""
        self.lambda_max = lambda_max
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0 if fit_intercept else None
        self.active_set_ = None
        self.signs_ = None
        self.coef_path_ = None
        self.lambda_path_ = None
        self.t_path_ = None
        self.transition_points_ = None  # Store transition points
        self.n_features_ = None
        self.mean_ = None
        self.std_ = None
        self.X_train_ = None  # Store training data
        self.y_train_ = None  # Store training targets
    
    def _standardize_features(self, X):
        """
        Standardize features for better numerical stability.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
            
        Returns:
        --------
        X_scaled : array-like of shape (n_samples, n_features)
            Standardized training data.
        """
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1  # Avoid division by zero
        return (X - self.mean_) / self.std_
    
    def _add_numerical_stability(self, matrix, epsilon=1e-10):
        """
        Add numerical stability to matrix operations.
        
        Parameters:
        -----------
        matrix : array-like
            Input matrix.
        epsilon : float, optional (default=1e-10)
            Small constant to add for stability.
            
        Returns:
        --------
        matrix : array-like
            Stabilized matrix.
        """
        return matrix + epsilon * np.eye(matrix.shape[0])
    
    def _compute_lambda_max(self, X, y):
        """
        Compute the maximum lambda value.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns:
        --------
        lambda_max : float
            Maximum lambda value.
        """
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
        return np.max(np.abs(X.T @ y))
    
    def _compute_transition_points(self, X, y, active_set, signs, lambda_current):
        """Compute transition points where the active set changes."""
        n_samples, n_features = X.shape
        n_active = np.sum(active_set)
        
        if n_active == 0:
            return np.array([])
        
        # Add small regularization to handle singular matrices
        epsilon = 1e-10
        X_active = X[:, active_set]
        signs_active = signs[active_set]
        
        # Compute dual variables with regularization
        try:
            X_active_inv = np.linalg.inv(X_active.T @ X_active + epsilon * np.eye(n_active))
            dual = y - X_active @ (X_active_inv @ (X_active.T @ y - lambda_current * signs_active))
        except np.linalg.LinAlgError:
            # If still singular, use pseudo-inverse
            X_active_inv = np.linalg.pinv(X_active.T @ X_active + epsilon * np.eye(n_active))
            dual = y - X_active @ (X_active_inv @ (X_active.T @ y - lambda_current * signs_active))
        
        # Compute correlations for inactive features
        correlations = X.T @ dual
        inactive_features = np.where(~active_set)[0]
        
        if len(inactive_features) == 0:
            return np.array([])
        
        # Compute transition points
        transition_points = []
        for j in inactive_features:
            if abs(correlations[j]) > lambda_current:
                # Compute the step size that would make feature j active
                step = (lambda_current - abs(correlations[j])) / (1 - np.abs(correlations[j]))
                if step > 0:
                    transition_points.append(step)
        
        return np.array(transition_points)
    
    def _update_solution(self, X, y, active_set, signs, lambda_current, t):
        """Update solution along the homotopy path."""
        n_samples, n_features = X.shape
        X_active = X[:, active_set]
        signs_active = signs[active_set]
        
        # Add regularization for numerical stability
        epsilon = 1e-10
        n_active = np.sum(active_set)
        
        if n_active == 0:
            return
        
        # Compute solution with regularization
        try:
            X_active_inv = np.linalg.inv(X_active.T @ X_active + epsilon * np.eye(n_active))
            beta_active = X_active_inv @ (X_active.T @ y - lambda_current * (1 - t) * signs_active)
        except np.linalg.LinAlgError:
            # If still singular, use pseudo-inverse
            X_active_inv = np.linalg.pinv(X_active.T @ X_active + epsilon * np.eye(n_active))
            beta_active = X_active_inv @ (X_active.T @ y - lambda_current * (1 - t) * signs_active)
        
        # Update coefficients
        self.coef_[active_set] = beta_active
        self.coef_[~active_set] = 0
    
    def _initialize_single_observation(self, X, y):
        """Initialize model for single observation case."""
        n_features = X.shape[1]
        self.coef_ = np.zeros(n_features)
        correlations = np.abs(X[0] * y[0])
        most_correlated = np.argmax(correlations)
        self.coef_[most_correlated] = y[0] / X[0, most_correlated] if X[0, most_correlated] != 0 else 0
        self.active_set_ = np.zeros(n_features, dtype=bool)
        self.active_set_[most_correlated] = True
        self.signs_ = np.zeros(n_features)
        self.signs_[most_correlated] = np.sign(X[0, most_correlated] * y[0])
        
        # Initialize paths
        self.coef_path_ = [np.zeros(n_features), self.coef_.copy()]
        self.lambda_path_ = [self.lambda_max, self.lambda_max * 0.5]
        self.t_path_ = [0.0, 1.0]
    
    def _cross_validate_lambda(self, X, y, n_folds=5):
        """Perform cross-validation to select the best lambda value."""
        n_samples = X.shape[0]
        fold_size = max(1, n_samples // n_folds)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        cv_scores = []
        for i in range(min(n_folds, n_samples)):
            # Split data into train and validation sets
            val_indices = indices[i * fold_size:min((i + 1) * fold_size, n_samples)]
            train_indices = np.concatenate([indices[:i * fold_size], indices[min((i + 1) * fold_size, n_samples):]])
            
            if len(train_indices) == 0 or len(val_indices) == 0:
                continue
            
            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]
            
            # Fit model on training data
            self._fit_path(X_train, y_train)
            
            # Evaluate on validation data
            scores = []
            for coef in self.coef_path_:
                y_pred = X_val @ coef
                mse = np.mean((y_val - y_pred) ** 2)
                r2 = 1 - mse / np.var(y_val) if np.var(y_val) > 0 else 0.0
                scores.append(r2)  # Use R² score instead of negative MSE
            
            if scores:
                cv_scores.append(scores)
        
        if not cv_scores:  # If no CV scores, return the last path index
            return len(self.coef_path_) - 1
        
        # Average scores across folds
        max_len = min(len(scores) for scores in cv_scores)
        cv_scores = [scores[:max_len] for scores in cv_scores]
        mean_scores = np.mean(cv_scores, axis=0)
        
        # Add small penalty for model complexity
        n_features = np.array([np.sum(np.abs(coef) > 1e-10) for coef in self.coef_path_[:max_len]])
        complexity_penalty = 0.01 * n_features / X.shape[1]
        adjusted_scores = mean_scores - complexity_penalty
        
        return np.argmax(adjusted_scores)
    
    def _fit_path(self, X, y):
        """Fit the solution path without selecting the best lambda."""
        n_samples, n_features = X.shape
        
        # Initialize coefficients and active set
        self.coef_ = np.zeros(n_features)
        self.active_set_ = np.zeros(n_features, dtype=bool)
        self.signs_ = np.zeros(n_features)
        
        # Initialize paths
        self.coef_path_ = [self.coef_.copy()]
        self.lambda_path_ = [self.lambda_max]
        self.t_path_ = [0.0]
        
        # Compute initial correlations
        correlations = np.abs(X.T @ y)
        if np.max(correlations) == 0:
            return
        
        # Initialize active set with most correlated feature
        most_correlated = np.argmax(correlations)
        self.active_set_[most_correlated] = True
        self.signs_[most_correlated] = np.sign(X[:, most_correlated].T @ y)
        
        # Main homotopy optimization loop
        t = 0.0
        max_iter = min(n_samples, n_features) * 10
        iter_count = 0
        min_lambda = self.lambda_max * 0.00001
        
        while t < 1.0 and iter_count < max_iter and self.lambda_path_[-1] > min_lambda:
            # Update solution for current active set
            X_active = X[:, self.active_set_]
            signs_active = self.signs_[self.active_set_]
            
            # Solve least squares for active set
            try:
                coef_active = np.linalg.solve(X_active.T @ X_active, X_active.T @ y)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if matrix is singular
                coef_active = np.linalg.pinv(X_active.T @ X_active) @ X_active.T @ y
            
            # Update coefficients
            self.coef_[self.active_set_] = coef_active
            
            # Compute correlations for inactive features
            inactive_mask = ~self.active_set_
            if np.any(inactive_mask):
                correlations = np.abs(X[:, inactive_mask].T @ (y - X @ self.coef_))
                
                # Find feature to add to active set
                if np.max(correlations) > self.lambda_path_[-1]:
                    new_feature = np.where(inactive_mask)[0][np.argmax(correlations)]
                    self.active_set_[new_feature] = True
                    self.signs_[new_feature] = np.sign(X[:, new_feature].T @ (y - X @ self.coef_))
            
            # Update paths
            self.coef_path_.append(self.coef_.copy())
            self.lambda_path_.append(self.lambda_path_[-1] * 0.95)  # Gradual decrease
            self.t_path_.append(t)
            
            t = 1.0 - (self.lambda_path_[-1] / self.lambda_max)
            iter_count += 1
    
    def fit(self, X, y):
        """Fit the Lasso model using homotopy optimization."""
        if len(X.shape) != 2:
            raise ValueError("X must be a 2D array")
        if len(y.shape) != 1:
            raise ValueError("y must be a 1D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if X.shape[0] == 0:
            raise ValueError("Cannot fit model with empty data")
        
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        n_samples, n_features = X.shape
        
        # Center and scale the data
        if self.fit_intercept:
            y_mean = np.mean(y)
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0)
            X_std[X_std == 0] = 1
            y = y - y_mean
            X = (X - X_mean) / X_std
        
        # Handle single observation case
        if n_samples == 1:
            self._initialize_single_observation(X, y)
            if self.fit_intercept:
                self.coef_ = self.coef_ / X_std
                self.intercept_ = y_mean - X_mean @ self.coef_
            return self
        
        # Fit solution path
        self._fit_path(X, y)
        
        # Select best lambda using cross-validation
        best_lambda_idx = self._cross_validate_lambda(X, y, n_folds=min(5, n_samples))
        self.coef_ = self.coef_path_[best_lambda_idx]
        
        # Scale back coefficients and compute intercept
        if self.fit_intercept:
            self.coef_ = self.coef_ / X_std
            self.intercept_ = y_mean - X_mean @ self.coef_
        else:
            self.intercept_ = 0.0
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        if self.coef_ is None:
            raise ValueError("Model must be fitted before making predictions.")
        
        predictions = X @ self.coef_
        if self.fit_intercept:
            predictions += self.intercept_
        
        return predictions
    
    def plot_regularization_path(self, X, y):
        """
        Plot the regularization path.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot path for each feature
        for i in range(self.n_features_):
            coef_path = [coef[i] for coef in self.coef_path_]
            ax.plot(self.lambda_path_, coef_path, label=f'Feature {i}')
        
        ax.set_xscale('log')
        ax.set_xlabel('Lambda')
        ax.set_ylabel('Coefficients')
        ax.set_title('LASSO Regularization Path')
        ax.grid(True)
        ax.legend()
        
        return fig
    
    def get_feature_importance(self):
        """Compute feature importance based on coefficient magnitudes and stability."""
        if not hasattr(self, 'coef_path_'):
            raise ValueError("Model must be fitted before computing feature importance")
        
        # Convert coefficient path to array and take absolute values
        coef_path_array = np.abs(np.array(self.coef_path_))
        
        # Compute importance as weighted average of absolute coefficients
        # Give more weight to solutions with smaller lambda
        lambda_weights = np.exp(-np.arange(len(self.lambda_path_)))
        lambda_weights = lambda_weights / np.sum(lambda_weights)
        
        # Compute weighted average of absolute coefficients
        importance = np.zeros(coef_path_array.shape[1])
        for i, coef in enumerate(coef_path_array):
            importance += lambda_weights[i] * np.abs(coef)
        
        # Add correlation-based importance
        correlations = np.abs(self.X_train_.T @ self.y_train_)
        correlation_importance = correlations / np.max(correlations)
        
        # Combine coefficient-based and correlation-based importance
        importance = 0.7 * importance + 0.3 * correlation_importance
        
        # Normalize importance scores
        importance = importance / np.max(importance) if np.max(importance) > 0 else correlation_importance
        
        return importance
    
    def plot_feature_importance(self):
        """
        Plot feature importance based on coefficient magnitudes.
        
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        importance = self.get_feature_importance()
        ax.bar(range(self.n_features_), importance)
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Importance (|Coefficient|)')
        ax.set_title('Feature Importance')
        ax.grid(True)
        
        return fig

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'lambda_max': self.lambda_max,
            'fit_intercept': self.fit_intercept
        }

    def set_params(self, **parameters):
        """Set the parameters of this estimator."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def cross_validate(self, X, y, n_folds=5, n_lambda=20):
        """Perform cross-validation to find the best lambda value."""
        n_samples = X.shape[0]
        fold_size = max(1, n_samples // n_folds)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        # Generate lambda sequence
        lambda_values = np.logspace(np.log10(self.lambda_max * 0.01), np.log10(self.lambda_max), n_lambda)
        cv_scores = []
        
        for lambda_val in lambda_values:
            fold_scores = []
            for i in range(min(n_folds, n_samples)):
                # Split data into train and validation sets
                val_indices = indices[i * fold_size:min((i + 1) * fold_size, n_samples)]
                train_indices = np.concatenate([indices[:i * fold_size], indices[min((i + 1) * fold_size, n_samples):]])
                
                if len(train_indices) == 0 or len(val_indices) == 0:
                    continue
                
                X_train, y_train = X[train_indices], y[train_indices]
                X_val, y_val = X[val_indices], y[val_indices]
                
                # Create and fit a new model instance
                model = LassoHomotopyModel(lambda_max=lambda_val, fit_intercept=self.fit_intercept)
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_val)
                
                # Compute validation score (R² score)
                mse = np.mean((y_val - y_pred) ** 2)
                var = np.var(y_val)
                r2 = 1 - mse / var if var > 0 else 0.0
                fold_scores.append(max(0.0, r2))  # Ensure non-negative R²
            
            if fold_scores:
                cv_scores.append(np.mean(fold_scores))
        
        if not cv_scores:
            return {'best_lambda': self.lambda_max, 'cv_scores': [0.0], 'lambda_values': [self.lambda_max]}
        
        # Find best lambda
        best_idx = np.argmax(cv_scores)
        best_lambda = lambda_values[best_idx]
        
        return {
            'best_lambda': best_lambda,
            'cv_scores': cv_scores,
            'lambda_values': lambda_values
        }
    
    def update_with_new_observation(self, x_new, y_new):
        """Update the model with a new observation using gradient-based lambda update."""
        if not hasattr(self, 'X_train_'):
            raise ValueError("Model must be fitted before updating with new observations")
        
        # Ensure x_new is 2D
        x_new = np.asarray(x_new).reshape(1, -1)
        y_new = np.asarray(y_new).reshape(-1)
        
        # Store current state
        old_coef = self.coef_.copy()
        
        # Combine old and new data
        X = np.vstack([self.X_train_, x_new])
        y = np.concatenate([self.y_train_, y_new])
        
        # Center the data if fitting intercept
        if self.fit_intercept:
            y_mean = np.mean(y)
            X_mean = np.mean(X, axis=0)
            y = y - y_mean
            X = X - X_mean
        
        # Update lambda based on prediction error
        y_pred = self.predict(x_new)
        error = np.abs(y_new - y_pred)[0]
        self.lambda_max = max(0.001, self.lambda_max * np.exp(-0.1 * error))
        
        # Refit model with updated data and lambda
        self.fit(X, y)
        
        # Update stored data
        self.X_train_ = X
        self.y_train_ = y
        
        # Check if coefficients have changed significantly
        return not np.array_equal(old_coef, self.coef_)
    
    def plot_solution_path(self):
        """Plot the solution path showing how coefficients change with lambda."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            for i in range(self.coef_path_[0].shape[0]):
                coef_values = [coef[i] for coef in self.coef_path_]
                plt.plot(self.t_path_, coef_values, label=f'Feature {i+1}')
            
            plt.xlabel('t (Homotopy Parameter)')
            plt.ylabel('Coefficient Value')
            plt.title('Lasso Solution Path')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            
            # Save plot to file instead of displaying
            plt.savefig('solution_path.png', bbox_inches='tight')
            plt.close()
            
            return 'solution_path.png'  # Return the filename
        except Exception as e:
            print(f"Warning: Could not create visualization due to: {str(e)}")
            return None

class LassoHomotopyResults:
    def __init__(self, model):
        self.model = model
        self.coef_ = model.coef_
        self.active_set_ = model.active_set_
        self.signs_ = model.signs_
        self.current_lambda_ = model.current_lambda_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the fitted model."""
        return X @ self.coef_

    def plot_regularization_path(self, X, y, n_lambda=100):
        """Plot the regularization path showing how coefficients change with lambda.
        
        Args:
            X: Feature matrix
            y: Target vector
            n_lambda: Number of lambda values to evaluate
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        import matplotlib.pyplot as plt
        
        # Generate lambda values
        lambda_max = self.model._compute_lambda_max(X, y)
        lambda_values = np.logspace(np.log10(lambda_max), np.log10(lambda_max/1000), n_lambda)
        
        # Store coefficients for each lambda
        coef_path = np.zeros((X.shape[1], n_lambda))
        
        # Fit model for each lambda
        for i, lambda_val in enumerate(lambda_values):
            self.model.lambda_max = lambda_val
            self.model.fit(X, y)
            coef_path[:, i] = self.model.coef_
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        for i in range(X.shape[1]):
            plt.plot(lambda_values, coef_path[i, :], label=f'Feature {i+1}')
        
        plt.xscale('log')
        plt.xlabel('Lambda')
        plt.ylabel('Coefficient Value')
        plt.title('LASSO Regularization Path')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        return plt.gcf()

    def plot_feature_importance(self):
        """Plot the absolute values of coefficients to show feature importance.
        
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        importance = np.abs(self.coef_)
        plt.bar(range(len(importance)), importance)
        plt.xlabel('Feature Index')
        plt.ylabel('Absolute Coefficient Value')
        plt.title('Feature Importance')
        plt.tight_layout()
        
        return plt.gcf()

    def get_feature_importance(self):
        """Get feature importance scores based on absolute coefficient values.
        
        Returns:
            dict: Dictionary mapping feature indices to importance scores
        """
        importance = np.abs(self.coef_)
        return {i: score for i, score in enumerate(importance)}
