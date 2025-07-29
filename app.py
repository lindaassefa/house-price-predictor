import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
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
</style>
""", unsafe_allow_html=True)

# API URL
API_URL = "https://house-price-predictor-api-glew.onrender.com"

def check_api_health():
    """Check if the API is available"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        return response.status_code == 200
    except:
        return False

def predict_price_api(property_data):
    """Make prediction using the API"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=property_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.json().get("error", "Prediction failed")}
    except Exception as e:
        return {"error": f"API Error: {str(e)}"}

def get_data_stats():
    """Get data statistics from API"""
    try:
        response = requests.get(f"{API_URL}/data/stats", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🏠 House Price Predictor</h1>', unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ API is not available. Please check the deployment status.")
        st.info("API URL: " + API_URL)
        return
    
    st.success("✅ API is connected and ready!")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Predict Price", "📊 Data Analysis", "📈 API Status", "ℹ️ About"]
    )
    
    if page == "🏠 Predict Price":
        show_prediction_page()
    elif page == "📊 Data Analysis":
        show_analysis_page()
    elif page == "📈 API Status":
        show_status_page()
    elif page == "ℹ️ About":
        show_about_page()

def show_prediction_page():
    """Show the main prediction page"""
    st.header("Predict House Price")
    st.write("Enter the details of the house to get a price prediction.")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Property Details")
        
        # Property information
        state = st.selectbox("State", ["CA", "NY", "TX", "FL", "IL"])
        city = st.text_input("City", "Los Angeles")
        street = st.text_input("Street Address", "123 Main St")
        zipcode = st.number_input("Zipcode", min_value=10000, max_value=99999, value=90210)
        
        # Property features
        bedrooms = st.slider("Bedrooms", 1, 10, 3)
        bathrooms = st.slider("Bathrooms", 1.0, 10.0, 2.5, 0.5)
        area = st.number_input("Area (sq ft)", min_value=100, max_value=10000, value=2000)
        lot_area = st.number_input("Lot Area", min_value=100, max_value=1000, value=500)
        
    with col2:
        st.subheader("Market Information")
        
        market_estimate = st.number_input("Market Estimate ($)", min_value=100000, max_value=5000000, value=1000000)
        rent_estimate = st.number_input("Rent Estimate ($/month)", min_value=500, max_value=10000, value=3000)
        
        # Optional coordinates
        st.subheader("Location (Optional)")
        use_coordinates = st.checkbox("Include coordinates")
        
        if use_coordinates:
            lat = st.number_input("Latitude", min_value=25.0, max_value=50.0, value=34.0522, format="%.4f")
            lon = st.number_input("Longitude", min_value=-125.0, max_value=-65.0, value=-118.2437, format="%.4f")
        else:
            lat = 0.0
            lon = 0.0
    
    # Calculate price per sq ft automatically
    ppsq = market_estimate / area if area > 0 else 500
    
    # Prediction button
    if st.button("🚀 Predict Price", type="primary"):
        with st.spinner("Making prediction..."):
            # Prepare input data for API
            property_data = {
                'State': state,
                'City': city,
                'Street': street,
                'Zipcode': int(zipcode),
                'Bedroom': int(bedrooms),
                'Bathroom': float(bathrooms),
                'Area': float(area),
                'PPSq': float(ppsq),  # Calculated automatically
                'LotArea': float(lot_area),
                'MarketEstimate': float(market_estimate),
                'RentEstimate': float(rent_estimate),
                'Latitude': float(lat),
                'Longitude': float(lon)
            }
            
            # Make prediction using API
            result = predict_price_api(property_data)
            
            if "error" in result:
                st.error(f"❌ {result['error']}")
            else:
                prediction = result["prediction"]
                confidence = result["confidence_interval"]
                
                # Display prediction
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown(f"### Predicted House Price")
                st.markdown(f"## ${prediction:,.2f}")
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Show confidence interval from API
                st.info(f"Estimated range: ${confidence['lower']:,.2f} - ${confidence['upper']:,.2f}")
                
                # Show calculated price per sq ft
                st.success(f"💰 Calculated Price per sq ft: ${ppsq:.0f}")
                
                # Show input data
                with st.expander("📋 Input Data"):
                    st.json(property_data)

def show_analysis_page():
    """Show data analysis page"""
    st.header("📊 Data Analysis")
    
    stats = get_data_stats()
    
    if stats:
        # Summary statistics
        st.subheader("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Properties", f"{stats['total_properties']:,}")
        with col2:
            st.metric("Average Price", f"${stats['price_stats']['mean']:,.2f}")
        with col3:
            st.metric("Median Price", f"${stats['price_stats']['median']:,.2f}")
        with col4:
            st.metric("Price Range", f"${stats['price_stats']['min']:,.0f} - ${stats['price_stats']['max']:,.0f}")
        
        # Price distribution chart
        st.subheader("💰 Price Distribution")
        price_data = {
            "Metric": ["Mean", "Median", "Min", "Max"],
            "Price": [stats['price_stats']['mean'], stats['price_stats']['median'], stats['price_stats']['min'], stats['price_stats']['max']]
        }
        
        fig = px.bar(
            pd.DataFrame(price_data),
            x="Metric",
            y="Price",
            title="Price Statistics",
            color="Price",
            color_continuous_scale="viridis"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Property statistics
        st.subheader("🏠 Property Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Number of States", len(stats["states"]))
            st.metric("Number of Cities", stats["cities_count"])
        
        with col2:
            bedroom_stats = stats["bedroom_stats"]
            st.metric("Bedroom Range", f"{bedroom_stats['min']} - {bedroom_stats['max']}")
            st.metric("Unique Bedroom Counts", len(bedroom_stats['unique_values']))
        
    else:
        st.error("❌ Could not fetch data statistics")

def show_performance_page():
    """Show model performance page"""
    st.header("📈 Model Performance")
    
    # Check if model comparison image exists
    try:
        st.image("model_comparison.png", caption="Model Performance Comparison")
    except FileNotFoundError:
        st.warning("Model performance visualization not available. Please run the training script first.")
    
    st.subheader("Model Metrics")
    st.write("""
    The model has been trained on multiple algorithms and optimized for best performance.
    Key metrics include:
    - **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual prices
    - **Mean Absolute Error (MAE)**: Measures the average absolute difference between predicted and actual prices
    - **R² Score**: Indicates how well the model explains the variance in house prices
    - **Root Mean Squared Error (RMSE)**: Square root of MSE, in the same units as the target variable
    """)

def show_status_page():
    """Show API status page"""
    st.header("📈 API Status")
    
    # Health check
    health_response = requests.get(f"{API_URL}/health", timeout=10)
    
    if health_response.status_code == 200:
        health_data = health_response.json()
        
        st.success("✅ API is Healthy")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Status", health_data["status"])
            st.metric("Model Loaded", "✅" if health_data["model_loaded"] else "❌")
        
        with col2:
            st.metric("Data Loaded", "✅" if health_data["data_loaded"] else "❌")
            st.metric("Timestamp", health_data["timestamp"])
        
        # API endpoints
        st.subheader("🔗 Available Endpoints")
        endpoints = [
            {"Method": "GET", "Endpoint": "/", "Description": "API Information"},
            {"Method": "GET", "Endpoint": "/health", "Description": "Health Check"},
            {"Method": "POST", "Endpoint": "/predict", "Description": "Single Prediction"},
            {"Method": "POST", "Endpoint": "/predict/batch", "Description": "Batch Prediction"},
            {"Method": "GET", "Endpoint": "/data/stats", "Description": "Data Statistics"},
            {"Method": "GET", "Endpoint": "/data/cities/<state>", "Description": "Cities by State"}
        ]
        
        st.table(pd.DataFrame(endpoints))
        
    else:
        st.error("❌ API is not responding")

def show_about_page():
    """Show about page"""
    st.header("ℹ️ About This Project")
    
    st.write("""
    ## House Price Predictor
    
    This application uses machine learning to predict house prices based on various features including:
    - **Location**: State, City, Zipcode, Latitude, Longitude
    - **Property Details**: Number of bedrooms, bathrooms, total area, lot area
    - **Market Information**: Price per square foot, market estimates, rent estimates
    
    ### How it works:
    1. **Data Preprocessing**: The model cleans and prepares the housing data
    2. **Feature Engineering**: Categorical variables are encoded and numerical variables are scaled
    3. **Model Training**: Multiple algorithms are trained and compared
    4. **Hyperparameter Tuning**: The best model is optimized for performance
    5. **Prediction**: Users can input property details to get price predictions
    
    ### Technologies Used:
    - **Python**: Core programming language
    - **Scikit-learn**: Machine learning library
    - **Streamlit**: Web application framework
    - **Pandas**: Data manipulation
    - **Plotly**: Interactive visualizations
    
    ### Model Performance:
    The model has been trained on a comprehensive dataset of house sales and provides
    accurate price predictions with confidence intervals.
    """)

if __name__ == "__main__":
    main() 