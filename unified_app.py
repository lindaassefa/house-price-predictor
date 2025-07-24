import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import os

# Page configuration
st.set_page_config(
    page_title="House Price Predictor - Complete Suite",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .api-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the dataset"""
    try:
        data = pd.read_csv("cleaned_df.csv")
        return data.dropna()
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'cleaned_df.csv' is in the current directory.")
        return None

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = joblib.load("house_price_model.pkl")
        return model
    except FileNotFoundError:
        st.error("Model not found. Please run the training script first.")
        return None

def predict_price_direct(model, input_data):
    """Make price prediction directly using the model"""
    try:
        prediction = model.predict(input_data)
        return prediction[0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def predict_price_api(input_data):
    """Make price prediction using the API"""
    try:
        response = requests.post(
            "http://localhost:5001/predict",
            json=input_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection Error: {str(e)}"}

def get_api_stats():
    """Get API statistics"""
    try:
        response = requests.get("http://localhost:5001/data/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🏠 House Price Predictor - Complete Suite</h1>', unsafe_allow_html=True)
    
    # Load data and model
    data = load_data()
    model = load_model()
    
    if data is None or model is None:
        st.stop()
    
    # Sidebar for navigation
    st.sidebar.title("🏠 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section",
        ["🏠 Predict Price", "📊 Data Analysis", "🔌 API Testing", "📈 Model Performance", "⚙️ System Status", "ℹ️ About"]
    )
    
    if page == "🏠 Predict Price":
        show_prediction_page(data, model)
    elif page == "📊 Data Analysis":
        show_analysis_page(data)
    elif page == "🔌 API Testing":
        show_api_testing_page()
    elif page == "📈 Model Performance":
        show_performance_page()
    elif page == "⚙️ System Status":
        show_system_status_page()
    elif page == "ℹ️ About":
        show_about_page()

def show_prediction_page(data, model):
    """Show the main prediction page"""
    st.header("🏠 Predict House Price")
    st.write("Enter the details of the house to get a price prediction.")
    
    # Create tabs for different prediction methods
    tab1, tab2 = st.tabs(["🎯 Direct Prediction", "🔌 API Prediction"])
    
    with tab1:
        st.subheader("Direct Model Prediction")
        show_prediction_form(data, model, use_api=False)
    
    with tab2:
        st.subheader("API-Based Prediction")
        show_prediction_form(data, model, use_api=True)

def show_prediction_form(data, model, use_api=False):
    """Show prediction form"""
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Location Information")
        state = st.selectbox("State", sorted(data['State'].unique()), key=f"state_{use_api}")
        
        # Filter cities based on selected state
        state_cities = data[data['State'] == state]['City'].unique()
        city = st.selectbox("City", sorted(state_cities), key=f"city_{use_api}")
        
        zipcode = st.number_input("Zipcode", 
                                 min_value=int(data['Zipcode'].min()), 
                                 max_value=int(data['Zipcode'].max()),
                                 value=int(data['Zipcode'].median()),
                                 key=f"zipcode_{use_api}")
        
        latitude = st.number_input("Latitude", 
                                  min_value=float(data['Latitude'].min()), 
                                  max_value=float(data['Latitude'].max()),
                                  value=float(data['Latitude'].median()),
                                  format="%.6f",
                                  key=f"lat_{use_api}")
        
        longitude = st.number_input("Longitude", 
                                   min_value=float(data['Longitude'].min()), 
                                   max_value=float(data['Longitude'].max()),
                                   value=float(data['Longitude'].median()),
                                   format="%.6f",
                                   key=f"lon_{use_api}")
    
    with col2:
        st.subheader("Property Details")
        bedrooms = st.slider("Number of Bedrooms", 
                            min_value=int(data['Bedroom'].min()), 
                            max_value=int(data['Bedroom'].max()),
                            value=int(data['Bedroom'].median()),
                            key=f"bed_{use_api}")
        
        bathrooms = st.slider("Number of Bathrooms", 
                             min_value=float(data['Bathroom'].min()), 
                             max_value=float(data['Bathroom'].max()),
                             value=float(data['Bathroom'].median()),
                             step=0.5,
                             key=f"bath_{use_api}")
        
        area = st.number_input("Total Area (sq ft)", 
                              min_value=float(data['Area'].min()), 
                              max_value=float(data['Area'].max()),
                              value=float(data['Area'].median()),
                              key=f"area_{use_api}")
        
        lot_area = st.number_input("Lot Area (sq ft)", 
                                  min_value=float(data['LotArea'].min()), 
                                  max_value=float(data['LotArea'].max()),
                                  value=float(data['LotArea'].median()),
                                  key=f"lot_{use_api}")
        
        ppsq = st.number_input("Price per Square Foot", 
                              min_value=float(data['PPSq'].min()), 
                              max_value=float(data['PPSq'].max()),
                              value=float(data['PPSq'].median()),
                              key=f"ppsq_{use_api}")
    
    # Additional features
    st.subheader("Additional Information")
    col3, col4 = st.columns(2)
    
    with col3:
        market_estimate = st.number_input("Market Estimate", 
                                         min_value=float(data['MarketEstimate'].min()), 
                                         max_value=float(data['MarketEstimate'].max()),
                                         value=float(data['MarketEstimate'].median()),
                                         key=f"market_{use_api}")
        
        rent_estimate = st.number_input("Rent Estimate", 
                                       min_value=float(data['RentEstimate'].min()), 
                                       max_value=float(data['RentEstimate'].max()),
                                       value=float(data['RentEstimate'].median()),
                                       key=f"rent_{use_api}")
    
    with col4:
        street = st.text_input("Street Address", value="Sample Street", key=f"street_{use_api}")
    
    # Prediction button
    if st.button("🚀 Predict Price", type="primary", key=f"predict_{use_api}"):
        with st.spinner("Making prediction..."):
            # Create input dataframe
            input_data = {
                'State': state,
                'City': city,
                'Street': street,
                'Zipcode': zipcode,
                'Bedroom': bedrooms,
                'Bathroom': bathrooms,
                'Area': area,
                'PPSq': ppsq,
                'LotArea': lot_area,
                'MarketEstimate': market_estimate,
                'RentEstimate': rent_estimate,
                'Latitude': latitude,
                'Longitude': longitude
            }
            
            if use_api:
                # Use API
                result = predict_price_api(input_data)
                if 'error' in result:
                    st.error(f"API Error: {result['error']}")
                else:
                    prediction = result['prediction']
                    confidence = result['confidence_interval']
                    
                    # Display prediction
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.markdown(f"### Predicted House Price (API)")
                    st.markdown(f"## ${prediction:,.2f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.info(f"Confidence Range: ${confidence['lower']:,.2f} - ${confidence['upper']:,.2f}")
                    
                    # Show API response details
                    with st.expander("API Response Details"):
                        st.json(result)
            else:
                # Use direct model
                input_df = pd.DataFrame([input_data])
                prediction = predict_price_direct(model, input_df)
                
                if prediction is not None:
                    # Display prediction
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.markdown(f"### Predicted House Price (Direct)")
                    st.markdown(f"## ${prediction:,.2f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Show confidence interval (simplified)
                    confidence_range = prediction * 0.1  # 10% range
                    st.info(f"Estimated range: ${prediction - confidence_range:,.2f} - ${prediction + confidence_range:,.2f}")
            
            # Show similar properties
            st.subheader("Similar Properties in the Area")
            similar_properties = data[
                (data['State'] == state) & 
                (data['City'] == city) &
                (data['Bedroom'] == bedrooms)
            ].head(5)
            
            if not similar_properties.empty:
                for idx, prop in similar_properties.iterrows():
                    st.write(f"📍 {prop['Street']} - ${prop['ListedPrice']:,.2f} ({prop['Bedroom']} bed, {prop['Bathroom']} bath, {prop['Area']:,.0f} sq ft)")

def show_analysis_page(data):
    """Show data analysis page"""
    st.header("📊 Data Analysis")
    
    # Summary statistics
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Properties", f"{len(data):,}")
    with col2:
        st.metric("Average Price", f"${data['ListedPrice'].mean():,.2f}")
    with col3:
        st.metric("Median Price", f"${data['ListedPrice'].median():,.2f}")
    with col4:
        st.metric("Price Range", f"${data['ListedPrice'].min():,.0f} - ${data['ListedPrice'].max():,.0f}")
    
    # Price distribution
    st.subheader("Price Distribution")
    fig = px.histogram(data, x='ListedPrice', nbins=50, 
                      title="Distribution of House Prices",
                      labels={'ListedPrice': 'Price ($)', 'count': 'Number of Properties'})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Price vs Area
    st.subheader("Price vs Area Relationship")
    fig = px.scatter(data, x='Area', y='ListedPrice', 
                    title="House Price vs Total Area",
                    labels={'Area': 'Total Area (sq ft)', 'ListedPrice': 'Price ($)'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Price by number of bedrooms
    st.subheader("Average Price by Number of Bedrooms")
    bedroom_prices = data.groupby('Bedroom')['ListedPrice'].mean().reset_index()
    fig = px.bar(bedroom_prices, x='Bedroom', y='ListedPrice',
                title="Average Price by Number of Bedrooms",
                labels={'Bedroom': 'Number of Bedrooms', 'ListedPrice': 'Average Price ($)'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Geographic distribution
    st.subheader("Geographic Distribution")
    fig = px.scatter_mapbox(data, lat='Latitude', lon='Longitude', 
                           color='ListedPrice', size='Area',
                           hover_data=['State', 'City', 'Bedroom', 'Bathroom'],
                           title="House Locations and Prices",
                           mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

def show_api_testing_page():
    """Show API testing page"""
    st.header("🔌 API Testing")
    
    # API Status
    st.subheader("API Status")
    try:
        response = requests.get("http://localhost:5001/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.write("✅ API is running and healthy!")
            st.json(health_data)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-box">', unsafe_allow_html=True)
            st.write(f"❌ API returned status code: {response.status_code}")
            st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        st.markdown('<div class="error-box">', unsafe_allow_html=True)
        st.write(f"❌ Cannot connect to API: {str(e)}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # API Endpoints
    st.subheader("Available Endpoints")
    
    endpoints = [
        {"Method": "GET", "Endpoint": "/health", "Description": "Health check"},
        {"Method": "POST", "Endpoint": "/predict", "Description": "Single prediction"},
        {"Method": "POST", "Endpoint": "/predict/batch", "Description": "Batch predictions"},
        {"Method": "GET", "Endpoint": "/data/stats", "Description": "Dataset statistics"},
        {"Method": "GET", "Endpoint": "/data/cities/<state>", "Description": "Cities by state"}
    ]
    
    st.table(pd.DataFrame(endpoints))
    
    # Test API endpoints
    st.subheader("Test API Endpoints")
    
    # Test data stats
    if st.button("📊 Get Dataset Statistics"):
        stats = get_api_stats()
        if stats:
            st.json(stats)
        else:
            st.error("Failed to get statistics from API")
    
    # Test prediction
    st.subheader("Test Prediction")
    test_data = {
        "State": "CA",
        "City": "Los Angeles", 
        "Street": "123 Test St",
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
    
    if st.button("🧪 Test Prediction API"):
        result = predict_price_api(test_data)
        if 'error' in result:
            st.error(f"API Error: {result['error']}")
        else:
            st.success("✅ Prediction successful!")
            st.json(result)

def show_performance_page():
    """Show model performance page"""
    st.header("📈 Model Performance")
    
    # Check if model comparison image exists
    try:
        st.image("model_comparison.png", caption="Model Performance Comparison")
    except FileNotFoundError:
        st.warning("Model performance visualization not available. Please run the training script first.")
    
    st.subheader("Model Metrics")
    st.write("The model has been trained on multiple algorithms and optimized for best performance.")
    st.write("Key metrics include:")
    st.write("- Mean Squared Error (MSE): Measures the average squared difference between predicted and actual prices")
    st.write("- Mean Absolute Error (MAE): Measures the average absolute difference between predicted and actual prices")
    st.write("- R² Score: Indicates how well the model explains the variance in house prices")
    st.write("- Root Mean Squared Error (RMSE): Square root of MSE, in the same units as the target variable")
    
    # Model comparison table
    st.subheader("Model Comparison Results")
    comparison_data = {
        "Model": ["Random Forest", "Linear Regression"],
        "R² Score": [0.9926, -11614403911390316.0000],
        "RMSE": ["$97,159", "$122,003,628,442,793"],
        "MAE": ["$17,859", "$45,282,744,514,222"]
    }
    st.table(pd.DataFrame(comparison_data))

def show_system_status_page():
    """Show system status page"""
    st.header("⚙️ System Status")
    
    # Check all components
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Files Status")
        
        # Check required files
        files_to_check = [
            ("cleaned_df.csv", "Dataset"),
            ("house_price_model.pkl", "Trained Model"),
            ("model_comparison.png", "Performance Plot"),
            ("app.py", "Streamlit App"),
            ("api.py", "Flask API")
        ]
        
        for filename, description in files_to_check:
            if os.path.exists(filename):
                st.success(f"✅ {description}: {filename}")
            else:
                st.error(f"❌ {description}: {filename} (Missing)")
    
    with col2:
        st.subheader("🔌 Services Status")
        
        # Check Streamlit
        try:
            response = requests.get("http://localhost:8501", timeout=5)
            if response.status_code == 200:
                st.success("✅ Streamlit Web App: Running")
            else:
                st.warning(f"⚠️ Streamlit Web App: Status {response.status_code}")
        except:
            st.error("❌ Streamlit Web App: Not responding")
        
        # Check Flask API
        try:
            response = requests.get("http://localhost:5001/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ Flask API: Running")
            else:
                st.warning(f"⚠️ Flask API: Status {response.status_code}")
        except:
            st.error("❌ Flask API: Not responding")
    
    # System information
    st.subheader("📊 System Information")
    import platform
    import psutil
    
    sys_info = {
        "Platform": platform.system(),
        "Python Version": platform.python_version(),
        "CPU Usage": f"{psutil.cpu_percent()}%",
        "Memory Usage": f"{psutil.virtual_memory().percent}%",
        "Disk Usage": f"{psutil.disk_usage('/').percent}%"
    }
    
    for key, value in sys_info.items():
        st.write(f"**{key}**: {value}")

def show_about_page():
    """Show about page"""
    st.header("ℹ️ About This Project")
    
    st.header("🏠 House Price Predictor - Complete Suite")
    
    st.write("This is a comprehensive machine learning application that combines:")
    st.write("- Web Interface: Interactive Streamlit application")
    st.write("- REST API: Flask-based API for integration")
    st.write("- Machine Learning: Advanced prediction models")
    st.write("- Data Analysis: Comprehensive visualizations")
    
    st.subheader("🎯 Features:")
    st.write("- Direct Model Prediction: Use the trained model directly")
    st.write("- API Integration: Test and use the REST API")
    st.write("- Real-time Analysis: Interactive data visualizations")
    st.write("- System Monitoring: Check service status and performance")
    
    st.subheader("🛠️ Technologies:")
    st.write("- Python: Core programming language")
    st.write("- Streamlit: Web application framework")
    st.write("- Flask: REST API framework")
    st.write("- Scikit-learn: Machine learning library")
    st.write("- Plotly: Interactive visualizations")
    st.write("- Pandas: Data manipulation")
    
    st.subheader("📊 Model Performance:")
    st.write("- Random Forest: R² = 0.9926 (Excellent performance)")
    st.write("- RMSE: $97,159")
    st.write("- MAE: $17,859")
    
    st.subheader("🔌 API Endpoints:")
    st.write("- GET /health - Health check")
    st.write("- POST /predict - Single prediction")
    st.write("- POST /predict/batch - Batch predictions")
    st.write("- GET /data/stats - Dataset statistics")
    
    st.write("This unified application provides everything you need for house price prediction in one place!")

if __name__ == "__main__":
    main() 