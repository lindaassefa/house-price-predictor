import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the house price data"""
    print("Loading data...")
    data = pd.read_csv("cleaned_df.csv")
    
    # Remove rows with missing values
    data = data.dropna()
    print(f"Data shape after cleaning: {data.shape}")
    
    # Separate features and target
    X = data.drop('ListedPrice', axis=1)
    y = data['ListedPrice']
    
    # Define categorical and numerical columns
    categorical_cols = ['State', 'City', 'Street']
    numerical_cols = ['Zipcode', 'Bedroom', 'Bathroom', 'Area', 'PPSq', 'LotArea', 
                     'MarketEstimate', 'RentEstimate', 'Latitude', 'Longitude']
    
    return X, y, categorical_cols, numerical_cols

def create_preprocessing_pipeline(categorical_cols, numerical_cols):
    """Create preprocessing pipeline"""
    # Numerical transformer
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical transformer
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return preprocessor

def train_models(X, y, preprocessor):
    """Train multiple models and return the best one"""
    print("Training models...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models to try
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0)
    }
    
    best_model = None
    best_score = float('inf')
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'MSE': mse,
            'MAE': mae,
            'R2': r2,
            'RMSE': np.sqrt(mse)
        }
        
        print(f"{name} - MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
        
        # Update best model
        if mse < best_score:
            best_score = mse
            best_model = pipeline
    
    return best_model, results

def hyperparameter_tuning(X, y, preprocessor):
    """Perform hyperparameter tuning on the best model"""
    print("Performing hyperparameter tuning...")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create pipeline with Random Forest
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    
    # Define parameter grid
    param_grid = {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__max_depth': [10, 20, None],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4]
    }
    
    # Perform grid search
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {-grid_search.best_score_:.2f}")
    
    return grid_search.best_estimator_

def save_model(model, filename='house_price_model.pkl'):
    """Save the trained model"""
    print(f"Saving model to {filename}...")
    joblib.dump(model, filename)
    print("Model saved successfully!")

def plot_results(results):
    """Plot model comparison results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract metrics
    models = list(results.keys())
    mse_values = [results[model]['MSE'] for model in models]
    mae_values = [results[model]['MAE'] for model in models]
    r2_values = [results[model]['R2'] for model in models]
    rmse_values = [results[model]['RMSE'] for model in models]
    
    # Plot MSE
    axes[0, 0].bar(models, mse_values)
    axes[0, 0].set_title('Mean Squared Error')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot MAE
    axes[0, 1].bar(models, mae_values)
    axes[0, 1].set_title('Mean Absolute Error')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot R2
    axes[1, 0].bar(models, r2_values)
    axes[1, 0].set_title('R² Score')
    axes[1, 0].set_ylabel('R²')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot RMSE
    axes[1, 1].bar(models, rmse_values)
    axes[1, 1].set_title('Root Mean Squared Error')
    axes[1, 1].set_ylabel('RMSE')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run the complete training pipeline"""
    print("=== House Price Prediction Model Training ===")
    
    # Load and preprocess data
    X, y, categorical_cols, numerical_cols = load_and_preprocess_data()
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(categorical_cols, numerical_cols)
    
    # Train models
    best_model, results = train_models(X, y, preprocessor)
    
    # Perform hyperparameter tuning
    tuned_model = hyperparameter_tuning(X, y, preprocessor)
    
    # Save the best model
    save_model(tuned_model)
    
    # Plot results
    plot_results(results)
    
    print("\n=== Training Complete ===")
    print("Best model saved as 'house_price_model.pkl'")
    print("Model comparison plot saved as 'model_comparison.png'")

if __name__ == "__main__":
    main() 