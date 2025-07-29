import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px

# Configure the page
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API URL
API_URL = "https://house-price-predictor-api-glew.onrender.com"

def check_api_health():
    """Check if the API is available"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        return response.status_code == 200
    except:
        return False

def get_data_stats():
    """Get data statistics from API"""
    try:
        response = requests.get(f"{API_URL}/data/stats", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def predict_price(property_data):
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

def main():
    st.title("🏠 House Price Predictor")
    st.markdown("---")
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ API is not available. Please check the deployment status.")
        st.info("API URL: " + API_URL)
        return
    
    st.success("✅ API is connected and ready!")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Predict Price", "📊 Data Analysis", "📈 API Status"]
    )
    
    if page == "🏠 Predict Price":
        show_prediction_page()
    elif page == "📊 Data Analysis":
        show_analysis_page()
    elif page == "📈 API Status":
        show_status_page()

def show_prediction_page():
    st.header("🏠 Predict House Price")
    st.write("Enter the details of the house to get a price prediction.")
    
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
            property_data = {
                "State": state,
                "City": city,
                "Street": street,
                "Zipcode": int(zipcode),
                "Bedroom": int(bedrooms),
                "Bathroom": float(bathrooms),
                "Area": float(area),
                "PPSq": float(ppsq),  # Calculated automatically
                "LotArea": float(lot_area),
                "MarketEstimate": float(market_estimate),
                "RentEstimate": float(rent_estimate),
                "Latitude": float(lat),
                "Longitude": float(lon)
            }
            
            result = predict_price(property_data)
            
            if "error" in result:
                st.error(f"❌ {result['error']}")
            else:
                prediction = result["prediction"]
                confidence = result["confidence_interval"]
                
                # Display results
                st.success("✅ Prediction completed!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Predicted Price", f"${prediction:,.0f}")
                
                with col2:
                    st.metric("Confidence Lower", f"${confidence['lower']:,.0f}")
                
                with col3:
                    st.metric("Confidence Upper", f"${confidence['upper']:,.0f}")
                
                # Show calculated price per sq ft
                st.info(f"💰 Calculated Price per sq ft: ${ppsq:.0f}")
                
                # Show input data
                with st.expander("📋 Input Data"):
                    st.json(property_data)

def show_analysis_page():
    st.header("📊 Data Analysis")
    
    stats = get_data_stats()
    
    if stats:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Price Statistics")
            price_stats = stats["price_stats"]
            
            st.metric("Average Price", f"${price_stats['mean']:,.0f}")
            st.metric("Median Price", f"${price_stats['median']:,.0f}")
            st.metric("Min Price", f"${price_stats['min']:,.0f}")
            st.metric("Max Price", f"${price_stats['max']:,.0f}")
        
        with col2:
            st.subheader("🏠 Property Statistics")
            st.metric("Total Properties", stats["total_properties"])
            st.metric("Number of States", len(stats["states"]))
            st.metric("Number of Cities", stats["cities_count"])
            
            st.subheader("📊 Bedroom Range")
            bedroom_stats = stats["bedroom_stats"]
            st.write(f"Min: {bedroom_stats['min']} | Max: {bedroom_stats['max']}")
            st.write(f"Available: {bedroom_stats['unique_values']}")
        
        # Price distribution chart
        st.subheader("💰 Price Distribution")
        price_data = {
            "Metric": ["Mean", "Median", "Min", "Max"],
            "Price": [price_stats['mean'], price_stats['median'], price_stats['min'], price_stats['max']]
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
        
    else:
        st.error("❌ Could not fetch data statistics")

def show_status_page():
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

if __name__ == "__main__":
    main() 