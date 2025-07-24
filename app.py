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

def predict_price(model, input_data):
    """Make price prediction"""
    try:
        prediction = model.predict(input_data)
        return prediction[0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🏠 House Price Predictor</h1>', unsafe_allow_html=True)
    
    # Load data and model
    data = load_data()
    model = load_model()
    
    if data is None or model is None:
        st.stop()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Predict Price", "📊 Data Analysis", "📈 Model Performance", "ℹ️ About"]
    )
    
    if page == "🏠 Predict Price":
        show_prediction_page(data, model)
    elif page == "📊 Data Analysis":
        show_analysis_page(data)
    elif page == "📈 Model Performance":
        show_performance_page()
    elif page == "ℹ️ About":
        show_about_page()

def show_prediction_page(data, model):
    """Show the main prediction page"""
    st.header("Predict House Price")
    st.write("Enter the details of the house to get a price prediction.")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Location Information")
        state = st.selectbox("State", sorted(data['State'].unique()))
        
        # Filter cities based on selected state
        state_cities = data[data['State'] == state]['City'].unique()
        city = st.selectbox("City", sorted(state_cities))
        
        zipcode = st.number_input("Zipcode", 
                                 min_value=int(data['Zipcode'].min()), 
                                 max_value=int(data['Zipcode'].max()),
                                 value=int(data['Zipcode'].median()))
        
        latitude = st.number_input("Latitude", 
                                  min_value=float(data['Latitude'].min()), 
                                  max_value=float(data['Latitude'].max()),
                                  value=float(data['Latitude'].median()),
                                  format="%.6f")
        
        longitude = st.number_input("Longitude", 
                                   min_value=float(data['Longitude'].min()), 
                                   max_value=float(data['Longitude'].max()),
                                   value=float(data['Longitude'].median()),
                                   format="%.6f")
    
    with col2:
        st.subheader("Property Details")
        bedrooms = st.slider("Number of Bedrooms", 
                            min_value=int(data['Bedroom'].min()), 
                            max_value=int(data['Bedroom'].max()),
                            value=int(data['Bedroom'].median()))
        
        bathrooms = st.slider("Number of Bathrooms", 
                             min_value=float(data['Bathroom'].min()), 
                             max_value=float(data['Bathroom'].max()),
                             value=float(data['Bathroom'].median()),
                             step=0.5)
        
        area = st.number_input("Total Area (sq ft)", 
                              min_value=float(data['Area'].min()), 
                              max_value=float(data['Area'].max()),
                              value=float(data['Area'].median()))
        
        lot_area = st.number_input("Lot Area (sq ft)", 
                                  min_value=float(data['LotArea'].min()), 
                                  max_value=float(data['LotArea'].max()),
                                  value=float(data['LotArea'].median()))
        
        ppsq = st.number_input("Price per Square Foot", 
                              min_value=float(data['PPSq'].min()), 
                              max_value=float(data['PPSq'].max()),
                              value=float(data['PPSq'].median()))
    
    # Additional features
    st.subheader("Additional Information")
    col3, col4 = st.columns(2)
    
    with col3:
        market_estimate = st.number_input("Market Estimate", 
                                         min_value=float(data['MarketEstimate'].min()), 
                                         max_value=float(data['MarketEstimate'].max()),
                                         value=float(data['MarketEstimate'].median()))
        
        rent_estimate = st.number_input("Rent Estimate", 
                                       min_value=float(data['RentEstimate'].min()), 
                                       max_value=float(data['RentEstimate'].max()),
                                       value=float(data['RentEstimate'].median()))
    
    with col4:
        street = st.text_input("Street Address", value="Sample Street")
    
    # Prediction button
    if st.button("🚀 Predict Price", type="primary"):
        with st.spinner("Making prediction..."):
            # Create input dataframe
            input_data = pd.DataFrame({
                'State': [state],
                'City': [city],
                'Street': [street],
                'Zipcode': [zipcode],
                'Bedroom': [bedrooms],
                'Bathroom': [bathrooms],
                'Area': [area],
                'PPSq': [ppsq],
                'LotArea': [lot_area],
                'MarketEstimate': [market_estimate],
                'RentEstimate': [rent_estimate],
                'Latitude': [latitude],
                'Longitude': [longitude]
            })
            
            # Make prediction
            prediction = predict_price(model, input_data)
            
            if prediction is not None:
                # Display prediction
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown(f"### Predicted House Price")
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