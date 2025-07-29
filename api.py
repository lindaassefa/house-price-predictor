from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables
model = None
data = None

def load_model_and_data():
    """Load the trained model and data"""
    global model, data
    
    try:
        # Load model
        model = joblib.load("house_price_model.pkl")
        logger.info("Model loaded successfully")
        
        # Load data for validation
        data = pd.read_csv("cleaned_df.csv").dropna()
        logger.info("Data loaded successfully")
        
        return True
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return False
    except Exception as e:
        logger.error(f"Error loading model/data: {e}")
        return False

def validate_input(input_data):
    """Validate input data"""
    required_fields = [
        'State', 'City', 'Street', 'Zipcode', 'Bedroom', 'Bathroom', 
        'Area', 'PPSq', 'LotArea', 'MarketEstimate', 'RentEstimate'
    ]
    
    # Check for required fields
    missing_fields = [field for field in required_fields if field not in input_data]
    if missing_fields:
        return False, f"Missing required fields: {missing_fields}"
    
    # Handle optional latitude and longitude
    if 'Latitude' not in input_data or 'Longitude' not in input_data:
        # Set default values for missing coordinates
        input_data['Latitude'] = 0.0
        input_data['Longitude'] = 0.0
    
    # Validate data types and ranges
    try:
        # Validate numerical fields
        numerical_fields = {
            'Zipcode': (int, data['Zipcode'].min(), data['Zipcode'].max()),
            'Bedroom': (int, data['Bedroom'].min(), data['Bedroom'].max()),
            'Bathroom': (float, data['Bathroom'].min(), data['Bathroom'].max()),
            'Area': (float, data['Area'].min(), data['Area'].max()),
            'PPSq': (float, data['PPSq'].min(), data['PPSq'].max()),
            'LotArea': (float, data['LotArea'].min(), data['LotArea'].max()),
            'MarketEstimate': (float, data['MarketEstimate'].min(), data['MarketEstimate'].max()),
            'RentEstimate': (float, data['RentEstimate'].min(), data['RentEstimate'].max())
        }
        
        # Validate optional coordinates if provided
        if input_data['Latitude'] != 0.0 or input_data['Longitude'] != 0.0:
            coord_fields = {
                'Latitude': (float, data['Latitude'].min(), data['Latitude'].max()),
                'Longitude': (float, data['Longitude'].min(), data['Longitude'].max())
            }
            numerical_fields.update(coord_fields)
        
        for field, (data_type, min_val, max_val) in numerical_fields.items():
            value = input_data[field]
            
            # Try to convert to the expected type
            try:
                if data_type == int:
                    value = int(value)
                elif data_type == float:
                    value = float(value)
                input_data[field] = value
            except (ValueError, TypeError):
                return False, f"Field {field} must be {data_type.__name__}"
            
            if value < min_val or value > max_val:
                return False, f"Field {field} must be between {min_val} and {max_val}"
        
        # Validate categorical fields
        if input_data['State'] not in data['State'].unique():
            return False, f"Invalid State: {input_data['State']}"
        
        state_cities = data[data['State'] == input_data['State']]['City'].unique()
        if input_data['City'] not in state_cities:
            return False, f"Invalid City {input_data['City']} for State {input_data['State']}"
        
        return True, "Validation successful"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information"""
    return jsonify({
        'message': '🏠 House Price Predictor API',
        'version': '1.0.0',
        'status': 'active',
        'endpoints': {
            'health': '/health',
            'predict': '/predict',
            'batch_predict': '/predict/batch',
            'data_stats': '/data/stats',
            'cities_by_state': '/data/cities/<state>'
        },
        'documentation': 'https://github.com/lindaassefa/house-price-predictor',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
        'data_loaded': data is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict house price endpoint"""
    try:
        # Get input data
        input_data = request.get_json()
        
        if not input_data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Validate input
        is_valid, validation_message = validate_input(input_data)
        if not is_valid:
            return jsonify({'error': validation_message}), 400
        
        # Create DataFrame for prediction
        prediction_input = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(prediction_input)[0]
        
        # Calculate confidence interval (simplified)
        confidence_range = prediction * 0.1  # 10% range
        
        # Prepare response
        response = {
            'prediction': float(prediction),
            'confidence_interval': {
                'lower': float(prediction - confidence_range),
                'upper': float(prediction + confidence_range)
            },
            'input_data': input_data,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction made: ${prediction:,.2f}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint"""
    try:
        # Get input data
        input_data = request.get_json()
        
        if not input_data or 'properties' not in input_data:
            return jsonify({'error': 'No properties data provided'}), 400
        
        properties = input_data['properties']
        if not isinstance(properties, list):
            return jsonify({'error': 'Properties must be a list'}), 400
        
        predictions = []
        errors = []
        
        for i, prop in enumerate(properties):
            try:
                # Validate input
                is_valid, validation_message = validate_input(prop)
                if not is_valid:
                    errors.append({
                        'index': i,
                        'error': validation_message,
                        'data': prop
                    })
                    continue
                
                # Create DataFrame for prediction
                prediction_input = pd.DataFrame([prop])
                
                # Make prediction
                prediction = model.predict(prediction_input)[0]
                
                # Calculate confidence interval
                confidence_range = prediction * 0.1
                
                predictions.append({
                    'index': i,
                    'prediction': float(prediction),
                    'confidence_interval': {
                        'lower': float(prediction - confidence_range),
                        'upper': float(prediction + confidence_range)
                    },
                    'input_data': prop
                })
                
            except Exception as e:
                errors.append({
                    'index': i,
                    'error': str(e),
                    'data': prop
                })
        
        response = {
            'predictions': predictions,
            'errors': errors,
            'total_processed': len(properties),
            'successful_predictions': len(predictions),
            'failed_predictions': len(errors),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Batch prediction completed: {len(predictions)} successful, {len(errors)} failed")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

@app.route('/data/stats', methods=['GET'])
def get_data_stats():
    """Get dataset statistics"""
    try:
        if data is None:
            return jsonify({'error': 'Data not loaded'}), 500
        
        stats = {
            'total_properties': len(data),
            'price_stats': {
                'mean': float(data['ListedPrice'].mean()),
                'median': float(data['ListedPrice'].median()),
                'min': float(data['ListedPrice'].min()),
                'max': float(data['ListedPrice'].max()),
                'std': float(data['ListedPrice'].std())
            },
            'bedroom_stats': {
                'min': int(data['Bedroom'].min()),
                'max': int(data['Bedroom'].max()),
                'unique_values': sorted(data['Bedroom'].unique().tolist())
            },
            'bathroom_stats': {
                'min': float(data['Bathroom'].min()),
                'max': float(data['Bathroom'].max()),
                'unique_values': sorted(data['Bathroom'].unique().tolist())
            },
            'area_stats': {
                'mean': float(data['Area'].mean()),
                'median': float(data['Area'].median()),
                'min': float(data['Area'].min()),
                'max': float(data['Area'].max())
            },
            'states': sorted(data['State'].unique().tolist()),
            'cities_count': len(data['City'].unique())
        }
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting data stats: {str(e)}")
        return jsonify({'error': f'Failed to get data stats: {str(e)}'}), 500

@app.route('/data/cities/<state>', methods=['GET'])
def get_cities_by_state(state):
    """Get cities for a specific state"""
    try:
        if data is None:
            return jsonify({'error': 'Data not loaded'}), 500
        
        if state not in data['State'].unique():
            return jsonify({'error': f'State {state} not found'}), 404
        
        cities = sorted(data[data['State'] == state]['City'].unique().tolist())
        
        return jsonify({
            'state': state,
            'cities': cities,
            'count': len(cities)
        })
        
    except Exception as e:
        logger.error(f"Error getting cities for state {state}: {str(e)}")
        return jsonify({'error': f'Failed to get cities: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load model and data on startup
    if load_model_and_data():
        logger.info("Starting Flask API server...")
        # Use environment variable for port or default to 5001
        port = int(os.environ.get('PORT', 5001))
        app.run(debug=False, host='0.0.0.0', port=port)
    else:
        logger.error("Failed to load model or data. Exiting.")
        exit(1) 