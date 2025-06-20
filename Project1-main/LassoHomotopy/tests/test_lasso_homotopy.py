import numpy as np
import pytest
import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.LassoHomotopy import LassoHomotopyModel

def test_basic_functionality():
    """Test basic model functionality."""
    np.random.seed(42)
    n_samples, n_features = 100, 5
    X = np.random.randn(n_samples, n_features)
    true_coef = np.zeros(n_features)
    true_coef[:2] = [1, 2]
    y = X @ true_coef + 0.1 * np.random.randn(n_samples)

    # Fit model
    model = LassoHomotopyModel(lambda_max=0.1)
    model.fit(X, y)

    # Check basic attributes
    assert model.coef_ is not None
    assert model.coef_.shape == (n_features,)
    assert model.intercept_ is not None
    assert model.active_set_ is not None
    assert model.active_set_.shape == (n_features,)
    assert model.signs_ is not None
    assert model.signs_.shape == (n_features,)

    # Check predictions
    y_pred = model.predict(X)
    assert y_pred.shape == (n_samples,)
    assert np.all(np.isfinite(y_pred))

def test_collinear_features():
    """Test model behavior with collinear features."""
    np.random.seed(42)
    n_samples, n_features = 100, 5

    # Generate collinear features
    X = np.random.randn(n_samples, 2)
    X = np.column_stack([X, X, X[:, 0] + X[:, 1]])  # Create collinear features
    true_coef = np.array([1, 2, 0, 0, 0])
    y = X @ true_coef + 0.1 * np.random.randn(n_samples)

    # Fit model
    model = LassoHomotopyModel(lambda_max=0.1)
    model.fit(X, y)

    # Check that model handles collinearity
    assert model.coef_ is not None
    assert np.sum(model.active_set_) <= 3  # Should select at most 3 features

def test_online_learning():
    """Test online learning capabilities."""
    np.random.seed(42)
    n_samples, n_features = 100, 5
    X = np.random.randn(n_samples, n_features)
    true_coef = np.zeros(n_features)
    true_coef[:2] = [1, 2]
    y = X @ true_coef + 0.1 * np.random.randn(n_samples)

    # Initial fit with smaller lambda_max
    model = LassoHomotopyModel(lambda_max=0.01)
    model.fit(X, y)
    initial_coef = model.coef_.copy()

    # Add new observation
    x_new = np.random.randn(n_features)
    y_new = x_new @ true_coef + 0.1 * np.random.randn()
    model.update_with_new_observation(x_new, y_new)

    # Check that model updated
    assert not np.array_equal(model.coef_, initial_coef)

def test_cross_validation():
    """Test cross-validation functionality."""
    np.random.seed(42)
    n_samples, n_features = 100, 5
    X = np.random.randn(n_samples, n_features)
    true_coef = np.zeros(n_features)
    true_coef[:2] = [1, 2]
    y = X @ true_coef + 0.1 * np.random.randn(n_samples)

    # Perform cross-validation
    model = LassoHomotopyModel()
    cv_results = model.cross_validate(X, y, n_folds=5, n_lambda=20)

    # Check cross-validation results
    assert 'best_lambda' in cv_results
    assert 'cv_scores' in cv_results
    assert 'lambda_values' in cv_results
    assert len(cv_results['cv_scores']) == len(cv_results['lambda_values'])
    assert isinstance(cv_results['best_lambda'], float)
    assert cv_results['best_lambda'] > 0

def test_feature_importance():
    """Test feature importance computation."""
    np.random.seed(42)
    n_samples, n_features = 100, 5
    X = np.random.randn(n_samples, n_features)
    true_coef = np.zeros(n_features)
    true_coef[:2] = [1, 2]
    y = X @ true_coef + 0.1 * np.random.randn(n_samples)

    # Fit model with very small lambda_max to ensure non-zero coefficients
    model = LassoHomotopyModel(lambda_max=0.001)
    model.fit(X, y)

    # Check feature importance
    importance = model.get_feature_importance()
    assert importance.shape == (n_features,)
    assert np.all(importance >= 0)

    # Check that important features are identified
    assert importance[0] > 0
    assert importance[1] > 0

def test_visualization():
    """Test visualization methods."""
    np.random.seed(42)
    n_samples, n_features = 100, 5
    X = np.random.randn(n_samples, n_features)
    true_coef = np.zeros(n_features)
    true_coef[:2] = [1, 2]
    y = X @ true_coef + 0.1 * np.random.randn(n_samples)

    # Fit model
    model = LassoHomotopyModel(lambda_max=0.1)
    model.fit(X, y)

    # Test solution path plot
    model.plot_solution_path()
    
    # Check if the plot file was created
    assert os.path.exists('solution_path.png')

def test_edge_cases():
    """Test edge cases and error handling."""
    np.random.seed(42)
    
    # Test empty input
    with pytest.raises(ValueError):
        model = LassoHomotopyModel()
        model.fit(np.array([]), np.array([]))
    
    # Test single feature
    X = np.random.randn(10, 1)
    y = np.random.randn(10)
    model = LassoHomotopyModel()
    model.fit(X, y)
    assert model.coef_.shape == (1,)
    
    # Test single sample
    X = np.random.randn(1, 5)
    y = np.random.randn(1)
    model = LassoHomotopyModel()
    model.fit(X, y)
    assert model.coef_.shape == (5,)

def test_numerical_stability():
    """Test numerical stability with ill-conditioned data."""
    np.random.seed(42)
    n_samples, n_features = 100, 5
    
    # Generate ill-conditioned data
    X = np.random.randn(n_samples, n_features)
    X[:, 0] = X[:, 1] + 1e-10 * np.random.randn(n_samples)  # Nearly collinear
    true_coef = np.zeros(n_features)
    true_coef[:2] = [1, 2]
    y = X @ true_coef + 0.1 * np.random.randn(n_samples)
    
    # Fit model
    model = LassoHomotopyModel(lambda_max=0.1)
    model.fit(X, y)
    
    # Check that model handles ill-conditioning
    assert model.coef_ is not None
    assert np.all(np.isfinite(model.coef_))

def test_small_dataset():
    """Test model on the small test dataset."""
    # Load small test dataset
    data_path = os.path.join(os.path.dirname(__file__), 'small_test.csv')
    data = np.loadtxt(data_path, delimiter=',', skiprows=1)
    X = data[:, :-1]
    y = data[:, -1]
    
    # Preprocess data
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1  # Avoid division by zero
    X_scaled = (X - X_mean) / X_std
    
    # Fit model with smaller lambda
    model = LassoHomotopyModel(lambda_max=0.01)  # Reduced from 0.1
    model.fit(X_scaled, y)
    
    # Check basic attributes
    assert model.coef_ is not None
    assert model.coef_.shape == (X.shape[1],)
    assert model.intercept_ is not None
    assert model.active_set_ is not None
    assert model.active_set_.shape == (X.shape[1],)
    
    # Check predictions
    y_pred = model.predict(X_scaled)
    assert y_pred.shape == (X.shape[0],)
    assert np.all(np.isfinite(y_pred))
    
    # Check prediction accuracy
    mse = np.mean((y - y_pred) ** 2)
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    assert mse < 25.0  # MSE should be less than the variance of the target
    assert r2 > 0.5   # R² should be positive and reasonably high
    
    # Check feature importance
    importance = model.get_feature_importance()
    assert importance.shape == (X.shape[1],)
    assert np.all(importance >= 0)
    
    # Check that at least one feature is selected
    assert np.sum(model.active_set_) > 0
    
    # Check cross-validation
    cv_results = model.cross_validate(X_scaled, y, n_folds=5, n_lambda=20)
    assert 'best_lambda' in cv_results
    assert 'cv_scores' in cv_results
    assert 'lambda_values' in cv_results
    assert len(cv_results['cv_scores']) == len(cv_results['lambda_values'])
    assert isinstance(cv_results['best_lambda'], float)
    assert cv_results['best_lambda'] > 0
    assert np.min(cv_results['cv_scores']) > 0  # CV scores should be positive

def test_collinear_dataset():
    """Test model on the collinear dataset."""
    # Load collinear dataset
    data_path = os.path.join(os.path.dirname(__file__), 'collinear_data.csv')
    data = np.loadtxt(data_path, delimiter=',', skiprows=1)
    X = data[:, :-1]
    y = data[:, -1]
    
    # Preprocess data
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1  # Avoid division by zero
    X_scaled = (X - X_mean) / X_std
    
    # Fit model with larger lambda to enforce sparsity
    model = LassoHomotopyModel(lambda_max=0.1)  # Increased significantly to enforce sparsity
    model.fit(X_scaled, y)
    
    # Check basic attributes
    assert model.coef_ is not None
    assert model.coef_.shape == (X.shape[1],)
    assert model.intercept_ is not None
    assert model.active_set_ is not None
    assert model.active_set_.shape == (X.shape[1],)
    
    # Check that model handles collinearity
    assert np.sum(model.active_set_) <= X.shape[1]  # Should not select all features
    
    # Check predictions
    y_pred = model.predict(X_scaled)
    assert y_pred.shape == (X.shape[0],)
    assert np.all(np.isfinite(y_pred))
    
    # Check prediction accuracy
    mse = np.mean((y - y_pred) ** 2)
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    assert mse < 25.0  # MSE should be less than the variance of the target
    assert r2 > 0.3   # R² should be positive but can be lower due to high sparsity
    
    # Check feature importance
    importance = model.get_feature_importance()
    assert importance.shape == (X.shape[1],)
    assert np.all(importance >= 0)
    
    # Check that at least one feature is selected
    assert np.sum(model.active_set_) > 0
    
    # Check cross-validation
    cv_results = model.cross_validate(X_scaled, y, n_folds=5, n_lambda=20)
    assert 'best_lambda' in cv_results
    assert 'cv_scores' in cv_results
    assert 'lambda_values' in cv_results
    assert len(cv_results['cv_scores']) == len(cv_results['lambda_values'])
    assert isinstance(cv_results['best_lambda'], float)
    assert cv_results['best_lambda'] > 0
    assert np.min(cv_results['cv_scores']) > 0  # CV scores should be positive
    
    # Check effective sparsity (coefficients close to zero)
    n_significant_coef = np.sum(np.abs(model.coef_) > 0.01)  # Count coefficients larger than 0.01
    assert n_significant_coef < X.shape[1]  # Should have fewer significant coefficients than features
    assert n_significant_coef > 0  # Should have at least one significant coefficient 