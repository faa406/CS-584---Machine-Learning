import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
import sys

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import LassoHomotopy from model directory
from model.LassoHomotopy import LassoHomotopyModel

# Define output directory
OUTPUT_DIR = os.path.join(current_dir, 'Result Netflix dataset')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_duration_features(duration_str):
    """Extract numerical duration and unit."""
    if pd.isna(duration_str):
        return pd.Series({'duration_value': np.nan, 'is_minutes': 0})
    try:
        value = float(''.join(filter(str.isdigit, str(duration_str))))
        is_minutes = 1 if 'min' in str(duration_str).lower() else 0
        if not is_minutes:  # If it's seasons, multiply by average episodes
            value *= 10  # Assuming average 10 episodes per season
        return pd.Series({'duration_value': value, 'is_minutes': is_minutes})
    except:
        return pd.Series({'duration_value': np.nan, 'is_minutes': 0})

def extract_text_features(text):
    """Extract basic text features."""
    if pd.isna(text):
        return pd.Series({'text_length': 0, 'word_count': 0})
    text = str(text)
    return pd.Series({
        'text_length': len(text),
        'word_count': len(text.split())
    })

def load_and_preprocess_data():
    """Load and preprocess the Netflix titles dataset with enhanced features."""
    print("Loading data...")
    netflix_data_path = os.path.join(current_dir, 'netflix_titles.csv')
    df = pd.read_csv(netflix_data_path)
    
    # Handle missing values
    print("Handling missing values...")
    df['country'] = df['country'].fillna('Unknown')
    df['cast'] = df['cast'].fillna('Unknown Cast')
    df['director'] = df['director'].fillna('Unknown Director')
    df['description'] = df['description'].fillna('')
    
    # Extract duration features
    print("Extracting duration features...")
    duration_features = df['duration'].apply(extract_duration_features)
    df = pd.concat([df, duration_features], axis=1)
    
    # Extract text features from description
    print("Extracting text features...")
    text_features = df['description'].apply(extract_text_features)
    df = pd.concat([df, text_features], axis=1)
    
    # Create genre features from listed_in
    print("Creating genre features...")
    df['genres'] = df['listed_in'].fillna('')
    genres = df['genres'].str.get_dummies(sep=',')
    
    # Process date features
    print("Processing date features...")
    df['date_added'] = pd.to_datetime(df['date_added'], format='mixed')
    df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
    df['added_year'] = df['date_added'].dt.year
    df['added_month'] = df['date_added'].dt.month
    df['age'] = df['added_year'] - df['release_year']
    
    # Create cast and director count features
    print("Creating cast and director features...")
    df['cast_count'] = df['cast'].str.count(',') + 1
    df['director_count'] = df['director'].str.count(',') + 1
    
    # Select and prepare features
    numeric_features = [
        'release_year', 'added_year', 'added_month', 'age',
        'duration_value', 'is_minutes', 'cast_count', 'director_count',
        'text_length', 'word_count'
    ]
    
    categorical_features = ['type', 'country', 'rating']
    
    # Prepare feature transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Prepare feature matrix
    print("Preparing final feature matrix...")
    X = preprocessor.fit_transform(df)
    
    # Add genre features
    X = np.hstack([X, genres.values])
    
    # Prepare target (rating encoded as numerical)
    rating_encoder = OneHotEncoder(sparse_output=False)
    y = rating_encoder.fit_transform(df[['rating']])
    y = np.argmax(y, axis=1)  # Convert to single column
    
    # Get feature names for visualization
    numeric_features_final = numeric_features
    categorical_features_final = []
    for feature, encoder in zip(categorical_features, 
                              preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_):
        categorical_features_final.extend([f"{feature}_{val}" for val in encoder[1:]])
    genre_features = genres.columns.tolist()
    
    all_features = numeric_features_final + categorical_features_final + genre_features
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, all_features

def evaluate_predictions(y_true, y_pred):
    """Calculate various metrics for model evaluation."""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - mse / np.var(y_true)
    mae = np.mean(np.abs(y_true - y_pred))
    accuracy = np.mean(np.round(y_pred) == y_true) * 100  # Convert to percentage
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2,
        'MAE': mae,
        'Accuracy': accuracy
    }

def create_rating_distribution_plot(df):
    """Create a plot showing the distribution of ratings."""
    plt.figure(figsize=(12, 6))
    rating_counts = df['rating'].value_counts()
    sns.barplot(x=rating_counts.index, y=rating_counts.values)
    plt.title('Distribution of Netflix Content Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rating_distribution.png'))
    plt.close()

def create_content_type_plot(df):
    """Create a plot showing content type distribution."""
    plt.figure(figsize=(10, 6))
    type_counts = df['type'].value_counts()
    plt.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
    plt.title('Distribution of Content Types')
    plt.savefig(os.path.join(OUTPUT_DIR, 'content_type_distribution.png'))
    plt.close()

def create_release_year_distribution(df):
    """Create a plot showing content distribution by release year."""
    plt.figure(figsize=(15, 6))
    df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
    yearly_counts = df.groupby('release_year').size()
    plt.plot(yearly_counts.index, yearly_counts.values)
    plt.title('Netflix Content Distribution by Release Year')
    plt.xlabel('Release Year')
    plt.ylabel('Number of Titles')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'release_year_distribution.png'))
    plt.close()

def create_top_countries_plot(df):
    """Create a plot showing top 10 countries by content count."""
    plt.figure(figsize=(12, 6))
    country_counts = df['country'].value_counts().head(10)
    sns.barplot(x=country_counts.values, y=country_counts.index)
    plt.title('Top 10 Countries by Content Count')
    plt.xlabel('Number of Titles')
    plt.ylabel('Country')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'top_countries.png'))
    plt.close()

def create_genre_distribution(df):
    """Create a plot showing top genres distribution."""
    plt.figure(figsize=(12, 6))
    genres = df['listed_in'].str.split(',', expand=True).stack()
    genre_counts = genres.value_counts().head(10)
    sns.barplot(x=genre_counts.values, y=genre_counts.index)
    plt.title('Top 10 Genres on Netflix')
    plt.xlabel('Number of Titles')
    plt.ylabel('Genre')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'top_genres.png'))
    plt.close()

def create_duration_analysis(df):
    """Create plots analyzing content duration."""
    # Extract duration features first
    duration_features = df['duration'].apply(extract_duration_features)
    df = pd.concat([df, duration_features], axis=1)
    
    # Plot 1: Duration distribution for movies
    plt.figure(figsize=(12, 6))
    movies = df[df['type'] == 'Movie'].dropna(subset=['duration_value'])
    sns.histplot(data=movies, x='duration_value', bins=50)
    plt.title('Movie Duration Distribution')
    plt.xlabel('Duration (minutes)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'movie_duration_distribution.png'))
    plt.close()
    
    # Plot 2: Duration by rating
    plt.figure(figsize=(12, 6))
    df_clean = df.dropna(subset=['duration_value', 'rating'])
    sns.boxplot(data=df_clean, x='rating', y='duration_value')
    plt.title('Duration Distribution by Rating')
    plt.xlabel('Rating')
    plt.ylabel('Duration (minutes)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'duration_by_rating.png'))
    plt.close()

def create_trend_analysis(df):
    """Create plots analyzing content trends over time."""
    # Convert date_added to datetime
    df['date_added'] = pd.to_datetime(df['date_added'], format='mixed')
    
    # Plot 1: Content additions over time
    plt.figure(figsize=(15, 6))
    monthly_additions = df.groupby(df['date_added'].dt.to_period('M')).size()
    plt.plot(range(len(monthly_additions)), monthly_additions.values)
    plt.title('Netflix Content Additions Over Time')
    plt.xlabel('Time (months)')
    plt.ylabel('Number of Titles Added')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'content_additions_over_time.png'))
    plt.close()
    
    # Plot 2: Rating distribution over time
    plt.figure(figsize=(15, 6))
    yearly_ratings = pd.crosstab(df['release_year'], df['rating'])
    yearly_ratings.plot(kind='bar', stacked=True)
    plt.title('Rating Distribution Over Time')
    plt.xlabel('Release Year')
    plt.ylabel('Number of Titles')
    plt.legend(title='Rating', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rating_distribution_over_time.png'))
    plt.close()

def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load raw data for visualizations
    netflix_data_path = os.path.join(current_dir, 'netflix_titles.csv')
    df_raw = pd.read_csv(netflix_data_path)
    
    # Load and preprocess data for model
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()
    
    # Initialize and fit the model
    print("\nFitting the LassoHomotopy model...")
    model = LassoHomotopyModel(lambda_max=1.0, fit_intercept=True)
    model.fit(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)
    
    # Evaluate model
    print("\nEvaluating model performance...")
    metrics = evaluate_predictions(y_test, y_pred)
    
    print("\nModel Performance:")
    print(f"Accuracy: {metrics['Accuracy']:.2f}%")
    print(f"R² Score: {metrics['R²']:.4f}")
    print(f"MSE: {metrics['MSE']:.4f}")
    print(f"MAE: {metrics['MAE']:.4f}")
    
    # Generate all visualization plots
    print("\nGenerating visualization plots...")
    create_rating_distribution_plot(df_raw)
    create_content_type_plot(df_raw)
    create_release_year_distribution(df_raw)
    create_top_countries_plot(df_raw)
    create_genre_distribution(df_raw)
    create_duration_analysis(df_raw)
    create_trend_analysis(df_raw)
    
    print("\nAnalysis complete! Check the Result Netflix dataset folder for all visualizations.")

if __name__ == "__main__":
    main() 