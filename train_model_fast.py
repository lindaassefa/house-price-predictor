import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
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

def train_models_fast(X, y, preprocessor):
    """Train models quickly with minimal hyperparameter tuning"""
    print("Training models (fast mode)...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models to try (simplified)
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
        'Linear Regression': LinearRegression()
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

def save_model(model, filename='house_price_model.pkl'):
    """Save the trained model"""
    print(f"Saving model to {filename}...")
    joblib.dump(model, filename)
    print("Model saved successfully!")

def plot_results_simple(results):
    """Plot simple model comparison results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
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
    """Main function to run the fast training pipeline"""
    print("=== House Price Prediction Model Training (Fast Mode) ===")
    
    # Load and preprocess data
    X, y, categorical_cols, numerical_cols = load_and_preprocess_data()
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(categorical_cols, numerical_cols)
    
    # Train models (fast mode)
    best_model, results = train_models_fast(X, y, preprocessor)
    
    # Save the best model
    save_model(best_model)
    
    # Plot results
    plot_results_simple(results)
    
    print("\n=== Training Complete (Fast Mode) ===")
    print("Best model saved as 'house_price_model.pkl'")
    print("Model comparison plot saved as 'model_comparison.png'")
    
    # Print summary
    print("\nModel Performance Summary:")
    for name, metrics in results.items():
        print(f"{name}:")
        print(f"  - R² Score: {metrics['R2']:.4f}")
        print(f"  - RMSE: ${metrics['RMSE']:,.2f}")
        print(f"  - MAE: ${metrics['MAE']:,.2f}")

if __name__ == "__main__":
    main() 