# 🏠 House Price Predictor

A comprehensive machine learning project that predicts house prices based on various features including location, property details, and market information.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/lindaassefa/house-price-predictor)

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Model Performance](#model-performance)
- [Contributing](#contributing)

## ✨ Features

### 🎯 Core Functionality
- **Data Preprocessing**: Automated cleaning and preparation of housing datasets
- **Feature Engineering**: Advanced feature selection and engineering techniques
- **Model Training**: Multiple machine learning algorithms with hyperparameter tuning
- **Model Evaluation**: Comprehensive performance metrics and visualization
- **Web Application**: Interactive Streamlit interface for easy predictions
- **REST API**: Flask-based API for integration with other applications

### 🏗️ Technical Features
- **Multiple Algorithms**: Random Forest, Gradient Boosting, Linear Regression, Ridge, Lasso
- **Hyperparameter Tuning**: Automated optimization using GridSearchCV
- **Data Validation**: Input validation and error handling
- **Batch Processing**: Support for multiple predictions at once
- **Interactive Visualizations**: Plotly-based charts and maps
- **Responsive Design**: Modern, user-friendly interface

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd House_price_Predictor-main
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import streamlit, flask, sklearn; print('All packages installed successfully!')"
   ```

## 📖 Usage

### 1. Model Training

First, train the machine learning model:

```bash
python train_model.py
```

This will:
- Load and preprocess the housing data
- Train multiple models and compare performance
- Perform hyperparameter tuning
- Save the best model as `house_price_model.pkl`
- Generate performance comparison plots

### 2. Web Application

Launch the Streamlit web application:

```bash
streamlit run app.py
```

The application will open in your browser with:
- **Predict Price**: Interactive form for single predictions
- **Data Analysis**: Comprehensive data visualizations
- **Model Performance**: Model metrics and comparison charts
- **About**: Project information and documentation

### 3. REST API

Start the Flask API server:

```bash
python api.py
```

The API will be available at `http://localhost:5001` with endpoints:
- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `GET /data/stats` - Dataset statistics
- `GET /data/cities/<state>` - Cities by state

## 📁 Project Structure

```
House_price_Predictor-main/
├── cleaned_df.csv              # Cleaned dataset
├── train_model.py              # Model training script
├── app.py                      # Streamlit web application
├── api.py                      # Flask REST API
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── housesales.ipynb            # Original Jupyter notebook
├── Frontend.ipynb              # Frontend development notebook
├── house_price_model.pkl       # Trained model (generated)
├── model_comparison.png        # Performance plots (generated)
└── pyvenv.cfg                  # Virtual environment config
```

## 🔌 API Documentation

### Single Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "State": "CA",
    "City": "Los Angeles",
    "Street": "123 Main St",
    "Zipcode": 90210,
    "Bedroom": 3,
    "Bathroom": 2.5,
    "Area": 2000,
    "PPSq": 500,
    "LotArea": 5000,
    "MarketEstimate": 1000000,
    "RentEstimate": 3000,
    "Latitude": 34.0522,
    "Longitude": -118.2437
  }'
```

### Batch Prediction
```bash
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "properties": [
      {
        "State": "CA",
        "City": "Los Angeles",
        "Street": "123 Main St",
        "Zipcode": 90210,
        "Bedroom": 3,
        "Bathroom": 2.5,
        "Area": 2000,
        "PPSq": 500,
        "LotArea": 5000,
        "MarketEstimate": 1000000,
        "RentEstimate": 3000,
        "Latitude": 34.0522,
        "Longitude": -118.2437
      }
    ]
  }'
```

### Get Dataset Statistics
```bash
curl http://localhost:5000/data/stats
```

## 📊 Model Performance

The model has been trained on a comprehensive dataset with the following performance metrics:

- **Mean Squared Error (MSE)**: Measures prediction accuracy
- **Mean Absolute Error (MAE)**: Average absolute prediction error
- **R² Score**: Model's explanatory power
- **Root Mean Squared Error (RMSE)**: Standard deviation of prediction errors

### Model Comparison
The training script compares multiple algorithms:
- Random Forest Regressor
- Gradient Boosting Regressor
- Linear Regression
- Ridge Regression
- Lasso Regression

The best performing model is automatically selected and saved.

## 🛠️ Technologies Used

- **Python**: Core programming language
- **Scikit-learn**: Machine learning library
- **Streamlit**: Web application framework
- **Flask**: REST API framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualizations
- **Matplotlib/Seaborn**: Static visualizations
- **Joblib**: Model serialization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter any issues or have questions:
1. Check the documentation above
2. Review the error messages in the console
3. Ensure all dependencies are installed correctly
4. Verify the dataset file is present and accessible

## 🎯 Future Enhancements

- [ ] Add more advanced models (Neural Networks, XGBoost)
- [ ] Implement real-time data updates
- [ ] Add user authentication and prediction history
- [ ] Create mobile application
- [ ] Add more geographic features and market indicators
- [ ] Implement ensemble methods for improved accuracy
