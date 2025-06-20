import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.gradient_boosting import GradientBoostingClassifier

def test_gradient_boosting():
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Split into train and test
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    # Initialize and fit the model
    model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Calculate accuracy in percentage
    accuracy = np.mean(y_pred == y_test) * 100
    print(f"Test accuracy: {accuracy:.2f}%")
    
    # Verify probability outputs
    assert np.all(y_proba >= 0) and np.all(y_proba <= 1), "Probabilities should be between 0 and 1"
    assert np.allclose(np.sum(y_proba, axis=1), 1), "Probabilities should sum to 1"
    
    # Test with different parameters
    model = GradientBoostingClassifier(n_estimators=10, learning_rate=0.5, max_depth=2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test) * 100
    print(f"Test accuracy with different parameters: {accuracy:.2f}%")

if __name__ == "__main__":
    test_gradient_boosting() 