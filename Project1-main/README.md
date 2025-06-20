# Lasso Homotopy Implementation

This project implements the LASSO (Least Absolute Shrinkage and Selection Operator) regularized regression model using the Homotopy Method. The implementation is done from first principles, focusing on providing a clear understanding of how LASSO works and its applications.

## Team Information

This project was developed by the following team members:

1. Meet Patel

   - Student ID: A20558374
   - Role: Core Implementation and Testing

2. Fatima Vahora

   - Student ID: A20555359
   - Role: Data Analysis and Visualization

3. Baozhu Xie
   - Student ID: A20549948
   - Role: Model Development and Documentation

## Overview

The LASSO Homotopy method is a powerful technique for feature selection and regularization in linear regression. It solves the following optimization problem:

min ||y - Xβ||²₂ + λ||β||₁

where:

- y is the target variable
- X is the feature matrix
- β is the coefficient vector
- λ is the regularization parameter
- ||·||₁ represents the L1 norm
- ||·||₂ represents the L2 norm

### Key Features

- Implementation from first principles without relying on scikit-learn's built-in models
- Efficient solution path computation using the Homotopy method
- Support for both small and large-scale datasets
- Comprehensive test suite including collinear data scenarios
- Visualization tools for solution paths and feature importance
- Netflix dataset analysis example

## Implementation Details and Usage Guidelines

### 1. What does the model do and when should it be used?

The implemented Lasso Homotopy model serves several key purposes:

1. **Feature Selection**:

   - Automatically identifies and selects the most important features
   - Reduces model complexity by setting less important coefficients to zero
   - Particularly effective when dealing with high-dimensional data

2. **Regularization**:

   - Prevents overfitting by adding L1 penalty to the loss function
   - Controls model complexity through the regularization parameter λ
   - Helps in finding sparse solutions

3. **Solution Path Analysis**:
   - Computes the entire regularization path efficiently
   - Shows how coefficients change as regularization strength varies
   - Helps in understanding feature importance at different regularization levels

The model should be used when:

- Feature selection is needed in regression problems
- Dealing with collinear features
- Working with high-dimensional data
- Need for interpretable, sparse solutions
- Understanding feature importance across different regularization strengths

### 2. How was the model tested?

The model was thoroughly tested through multiple approaches:

1. **Unit Tests**:

   - Basic functionality tests
   - Edge cases and error conditions
   - Parameter validation
   - Core algorithm correctness

2. **Performance Testing**:

   - Small dataset (3 features): 99.75% R² score
   - Collinear dataset (10 features): 84.09% R² score
   - Netflix dataset: 98.82% R² score

3. **Validation Scenarios**:

   - Feature selection accuracy
   - Solution path computation
   - Convergence properties
   - Numerical stability
   - Memory efficiency

4. **Real-world Application**:
   - Netflix dataset analysis
   - Content classification
   - Feature importance estimation

### 3. What parameters are exposed for tuning?

The implementation exposes several key parameters for performance tuning:

1. **Core Parameters**:

   - `lambda_max`: Maximum regularization parameter (default: 1.0)
   - `fit_intercept`: Whether to include intercept term (default: True)
   - `max_iter`: Maximum iterations for convergence (default: 1000)
   - `tol`: Convergence tolerance (default: 1e-4)

2. **Data Preprocessing Parameters**:

   - Feature scaling options
   - Missing value handling
   - Categorical encoding methods

3. **Model Selection Parameters**:
   - Cross-validation settings
   - Regularization path resolution
   - Early stopping criteria

### 4. Specific Input Challenges and Solutions

Current Limitations:

1. **Memory Usage**:

   - Challenge: High memory consumption with large datasets
   - Solution: Could implement sparse matrix operations and chunked processing

2. **Computation Time**:

   - Challenge: Slow processing for high-dimensional data
   - Solution: Could add parallel processing and early stopping

3. **Numerical Stability**:

   - Challenge: Issues with highly ill-conditioned data
   - Solution: Could implement better numerical conditioning

4. **Sparse Matrix Handling**:
   - Challenge: Inefficient processing of sparse matrices
   - Solution: Could optimize sparse matrix operations

These challenges are not fundamental limitations but rather implementation-specific issues that could be addressed with:

- Better memory management
- Parallel processing implementation
- Improved numerical algorithms
- Sparse matrix optimizations
- Early stopping mechanisms

## Installation

1. Create and activate a virtual environment:

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── LassoHomotopy/
│   ├── model/           # Core implementation
│   └── tests/           # Test files and examples
├── requirements.txt     # Project dependencies
└── README.md           # This file
```

## Usage Examples

### Basic Usage

```python
from LassoHomotopy.model.lasso_homotopy import LassoHomotopyModel

# Initialize the model
model = LassoHomotopyModel(lambda_max=1.0, fit_intercept=True)

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
```

### Running Tests

```bash
cd LassoHomotopy/tests
pytest
```

### Netflix Dataset Analysis

```bash
cd LassoHomotopy/tests
python run_netflix_model.py
```

## Model Parameters

The implementation exposes the following parameters for tuning:

1. `lambda_max`: Maximum value of the regularization parameter (default: 1.0)
2. `fit_intercept`: Whether to fit an intercept term (default: True)
3. `max_iter`: Maximum number of iterations for convergence (default: 1000)
4. `tol`: Convergence tolerance (default: 1e-4)

## Testing and Validation

The model has been thoroughly tested using multiple test files in the `LassoHomotopy/tests` directory:

### Running Tests

1. **Unit Tests** (`test_lasso_homotopy.py`):

   ```bash
   cd LassoHomotopy/tests
   pytest test_lasso_homotopy.py -v
   ```

   This runs the core unit tests for the Lasso Homotopy implementation.

2. **Model Evaluation** (`evaluate_model.py`):

   ```bash
   cd LassoHomotopy/tests
   python evaluate_model.py
   ```

   This script evaluates the model's performance on different datasets and generates performance metrics.

3. **Netflix Dataset Analysis** (`run_netflix_model.py`):
   ```bash
   cd LassoHomotopy/tests
   python run_netflix_model.py
   ```
   This script analyzes the Netflix dataset and generates visualizations.

### Test Cases Include:

- Basic regression functionality
- Feature selection with collinear data
- Solution path computation
- Convergence properties
- Edge cases and error handling

### Test Datasets

The tests use three different datasets:

1. `small_test.csv`: Small dataset for basic functionality testing
2. `collinear_data.csv`: Dataset with collinear features to test feature selection
3. `netflix_titles.csv`: Real-world Netflix dataset for practical application

### Expected Output

Running the tests will generate:

- Test results and performance metrics
- Visualization plots in the respective result folders:
  - `Result given dataset/` for small and collinear data results
  - `Result Netflix dataset/` for Netflix analysis results
- Detailed analysis reports

### Test Coverage

The test suite covers:

1. Model initialization and parameter validation
2. Training and prediction functionality
3. Feature selection capabilities
4. Solution path computation
5. Performance on different data types
6. Error handling and edge cases
7. Real-world application scenarios

## Results and Analysis

### Model Performance Results

The Lasso Homotopy model demonstrated excellent performance across different datasets:

1. **Netflix Dataset Results**:

   - Accuracy: 99.94%
   - R² Score: 0.9882
   - MSE: 0.0460
   - MAE: 0.0051

2. **Small Dataset Results**:

   - Mean Squared Error (MSE): 0.0025
   - R²: 99.75%
   - Active Features: 3/3
   - Best Lambda: 0.0500

3. **Collinear Dataset Results**:
   - Mean Squared Error (MSE): 0.1591
   - R²: 84.09%
   - Active Features: 10/10
   - Best Lambda: 0.1000

### Generated Visualizations

The analysis generated several visualization plots stored in the respective result folders:

1. **Netflix Dataset Analysis** (`Result Netflix dataset/`):

   - `rating_distribution.png`: Distribution of content ratings
   - `content_type_distribution.png`: Movies vs TV Shows distribution
   - `release_year_distribution.png`: Content distribution by year
   - `top_countries.png`: Top 10 content-producing countries
   - `top_genres.png`: Most popular content genres
   - `movie_duration_distribution.png`: Movie length distribution
   - `duration_by_rating.png`: Duration patterns by rating
   - `content_additions_over_time.png`: Content growth trends
   - `rating_distribution_over_time.png`: Rating evolution

2. **Given Dataset Analysis** (`Result given dataset/`):
   - `collinear_dataset_bias_variance.png`: Bias-variance tradeoff analysis
   - `collinear_dataset_feature_selection_path.png`: Feature selection process
   - `collinear_dataset_qq_plot.png`: Residual normality check
   - `collinear_dataset_residuals_vs_fitted.png`: Residual analysis
   - `collinear_dataset_learning_curve.png`: Learning curve analysis
   - `collinear_dataset_correlation_matrix.png`: Feature correlations
   - `collinear_dataset_feature_importance_ci.png`: Feature importance with confidence intervals
   - `collinear_dataset_cv_scores.png`: Cross-validation results
   - `collinear_dataset_regularization_path.png`: Regularization path analysis

### Key Findings

1. **Model Effectiveness**:

   - Excellent performance on clean data (99.75% R²)
   - Robust handling of collinear features (84.09% R²)
   - Strong predictive power on real-world data (98.82% R²)

2. **Feature Selection**:

   - Successful identification of important features
   - Effective handling of collinear features
   - Clear feature importance ranking

3. **Regularization Impact**:

   - Optimal lambda values identified for different datasets
   - Clear regularization paths
   - Effective bias-variance tradeoff

4. **Real-world Application**:
   - Strong performance on Netflix dataset
   - Effective content classification
   - Reliable feature importance estimation

### Analysis Reports

Detailed analysis reports are available in the result folders:

- `netflix_analysis_report.txt`: Comprehensive Netflix dataset analysis
- `model_metrics.txt`: Detailed performance metrics
- `feature_importance.csv`: Feature importance rankings

## Limitations and Future Improvements

### Current Limitations:

1. Memory usage with very large datasets
2. Computation time for high-dimensional feature spaces
3. Handling of extremely sparse matrices

### Potential Improvements:

1. Parallel processing for large datasets
2. Sparse matrix optimization
3. Adaptive step size selection
4. Early stopping criteria
5. Cross-validation utilities

## When to Use This Model

This implementation is particularly useful when:

1. Feature selection is needed in regression problems
2. Dealing with collinear features
3. Understanding the LASSO solution path
4. Educational purposes in machine learning
5. Research on regularization methods

## Acknowledgments

- Netflix dataset for providing real-world testing data
- Academic papers on LASSO Homotopy method
- Contributors and maintainers of the project
