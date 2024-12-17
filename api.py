import os
import logging
import traceback
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import pandas as pd

from pipeline import AquaculturePredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Flask App Configuration
app = Flask(__name__)
CORS(app)

# Define upload and model directories
UPLOAD_FOLDER = 'uploads/'
MODELS_FOLDER = 'models/'
VISUALIZATIONS_FOLDER = 'visualizations/'

# Create necessary directories
for folder in [UPLOAD_FOLDER, MODELS_FOLDER, VISUALIZATIONS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Configure Flask app with directories
app.config.update({
    'UPLOAD_FOLDER': UPLOAD_FOLDER,
    'MODELS_FOLDER': MODELS_FOLDER,
    'VISUALIZATIONS_FOLDER': VISUALIZATIONS_FOLDER
})

# Global predictor initialization
predictor = None

def validate_input_features(input_data):
    """
    Validate input features before prediction
    """
    if not input_data:
        raise ValueError("Input data cannot be empty")
    
    # Add more specific validations as needed

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "success",
        "message": "Welcome to the Aquaculture Prediction API! The server is up and running.",
        "endpoints": {
            "/upload_dataset": "POST - Upload a dataset for training",
            "/train": "POST - Train a model for a specific target",
            "/predict": "POST - Make predictions using trained models",
            "/feature_importance": "GET - Get feature importance plot"
            
        }
    }), 200




@app.route('/train', methods=['POST'])
def train_model():
    """
    Train model for a specific target variable
    """
    global predictor
    try:
        # Validate JSON input
        if not request.is_json:
            return jsonify({
                'status': 'error',
                'message': 'Request must be JSON'
            }), 415
        
        target = request.json.get('target', 'YieldKgPerM3')
        valid_targets = ['YieldKgPerM3', 'EconomicPotential', 'SustainabilityScore']
        
        if target not in valid_targets:
            return jsonify({
                'status': 'error',
                'message': f'Invalid target. Choose from {valid_targets}'
            }), 400
        
        if predictor is None:
            return jsonify({
                'status': 'error',
                'message': 'No dataset loaded. Upload a dataset first.'
            }), 400
        
        metrics = predictor.train_model(target)
        return jsonify({
            'status': 'success',
            'target': target,
            'metrics': metrics
        })
    except Exception as e:
        logger.error(f"Training error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Make predictions with comprehensive error handling
    """
    global predictor
    try:
        # Validate JSON input
        if not request.is_json:
            return jsonify({
                'status': 'error',
                'message': 'Request must be JSON'
            }), 415
        
        # Get input data
        input_data = request.get_json()
        
        # Validate predictor is initialized
        if predictor is None:
            return jsonify({
                'status': 'error',
                'message': 'No dataset loaded. Upload a dataset first.'
            }), 400
        
        # Validate target
        target = input_data.pop('target', 'YieldKgPerM3')
        valid_targets = ['YieldKgPerM3', 'EconomicPotential', 'SustainabilityScore']
        
        if target not in valid_targets:
            return jsonify({
                'status': 'error',
                'message': f'Invalid target. Choose from {valid_targets}'
            }), 400
        
        try:
            # Validate input features
            validate_input_features(input_data)
            
            # Remove any encoded columns
            input_data = {k: v for k, v in input_data.items() 
                          if not k.endswith('_encoded')}
            
            # Make prediction
            prediction_result = predictor.predict(input_data, target)
            
            return jsonify({
                'status': 'success',
                **prediction_result
            })
        
        except ValueError as ve:
            logger.error(f"Prediction validation error: {ve}")
            return jsonify({
                'status': 'error',
                'message': 'Input data validation failed',
                'details': str(ve)
            }), 400
        
        except Exception as pred_error:
            logger.error(f"Prediction error: {pred_error}")
            return jsonify({
                'status': 'error',
                'message': 'Prediction failed',
                'details': str(pred_error)
            }), 500
    
    except Exception as e:
        logger.error(f"Unexpected prediction error: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Unexpected error during prediction',
            'details': str(e)
        }), 500

@app.route('/feature_importance', methods=['GET'])
def get_feature_importance():
    """
    Retrieve feature importance visualization
    """
    global predictor
    try:
        target = request.args.get('target', 'YieldKgPerM3')
        valid_targets = ['YieldKgPerM3', 'EconomicPotential', 'SustainabilityScore']
        
        if target not in valid_targets:
            return jsonify({
                'status': 'error',
                'message': f'Invalid target. Choose from {valid_targets}'
            }), 400
        
        if predictor is None:
            return jsonify({
                'status': 'error',
                'message': 'No dataset loaded. Upload a dataset first.'
            }), 400
        
        plot_path = predictor.generate_feature_importance(target)
        return send_file(plot_path, mimetype='image/png')
    
    except Exception as e:
        logger.error(f"Feature importance error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    """
    Upload a new dataset for training
    """
    global predictor
    try:
        if 'file' not in request.files:
            return jsonify({
                'status': 'error', 
                'message': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'status': 'error', 
                'message': 'No selected file'
            }), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Reinitialize predictor with new dataset
        predictor = AquaculturePredictor(filepath)
        
        return jsonify({
            'status': 'success', 
            'message': 'Dataset uploaded and loaded successfully',
            'filename': filename
        })
    
    except Exception as e:
        logger.error(f"Dataset upload error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/market_trend_analysis', methods=['GET'])
def market_trend_analysis():
    """
    Perform market trend analysis
    """
    global predictor
    try:
        if predictor is None:
            return jsonify({
                'status': 'error',
                'message': 'No dataset loaded. Upload a dataset first.'
            }), 400
        
        # Load the dataset
        df = predictor.df
        
        # Group by system type and calculate average metrics
        market_trends = df.groupby('SystemType').agg({
            'YieldKgPerM3': ['mean', 'std'],
            'EconomicPotential': ['mean', 'std'],
            'SustainabilityScore': ['mean', 'std']
        }).reset_index()
        
        market_trends.columns = [
            'SystemType', 
            'AvgYield', 'YieldStdDev', 
            'AvgEconomicPotential', 'EconomicPotentialStdDev', 
            'AvgSustainabilityScore', 'SustainabilityScoreStdDev'
        ]
        
        return jsonify({
            'status': 'success',
            'market_trends': market_trends.to_dict(orient='records')
        })
    
    except Exception as e:
        logger.error(f"Market trend analysis error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
