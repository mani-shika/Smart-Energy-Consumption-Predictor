"""
Deployable Flask application for Energy Consumption Predictor
Ready for Heroku, Railway, Render, or any cloud platform
"""
from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import io
import base64
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class EnergyPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'temperature', 'humidity', 'occupancy', 'hour', 'day_of_week', 
            'month', 'is_weekend', 'hvac_usage', 'lighting_usage', 
            'electronics_usage', 'building_type_encoded', 'building_size',
            'temp_occupancy', 'temp_humidity', 'is_business_hours', 
            'is_peak_hours', 'total_usage', 'hvac_ratio'
        ]
        self.train_model()
    
    def generate_training_data(self, n_samples=2000):
        """Generate synthetic training data"""
        np.random.seed(42)
        data = []
        
        for i in range(n_samples):
            # Basic features
            hour = np.random.randint(0, 24)
            day_of_week = np.random.randint(0, 7)
            month = np.random.randint(1, 13)
            
            # Temperature with seasonal patterns
            base_temp = 20 + 8 * np.sin(2 * np.pi * (month - 1) / 12)
            daily_temp_var = 6 * np.sin(2 * np.pi * (hour - 6) / 24)
            temperature = base_temp + daily_temp_var + np.random.normal(0, 3)
            
            humidity = max(20, min(80, np.random.normal(50, 15)))
            
            # Occupancy patterns
            is_weekend = day_of_week >= 5
            if is_weekend:
                if 8 <= hour <= 22:
                    occupancy = np.random.randint(1, 6)
                else:
                    occupancy = np.random.randint(0, 3)
            else:
                if 9 <= hour <= 17:
                    occupancy = np.random.randint(5, 15)
                elif 6 <= hour <= 8 or 18 <= hour <= 20:
                    occupancy = np.random.randint(2, 8)
                else:
                    occupancy = np.random.randint(0, 4)
            
            # Building type
            building_types = [0, 1, 2, 3]  # encoded: residential, office, retail, industrial
            building_type = np.random.choice(building_types)
            building_size = np.random.randint(1000, 12000)
            
            # Usage patterns
            comfort_temp = 22
            temp_diff = abs(temperature - comfort_temp)
            hvac_usage = (0.1 * temp_diff * (1 + occupancy * 0.05) * 
                         (1.0 if 6 <= hour <= 22 else 0.6) * np.random.uniform(0.8, 1.2))
            
            if 7 <= hour <= 19:
                lighting_base = 0.02
            else:
                lighting_base = 0.08
            lighting_usage = lighting_base * occupancy * np.random.uniform(0.5, 1.5)
            
            electronics_usage = occupancy * np.random.uniform(0.1, 0.4)
            
            # Calculate total consumption
            base_load = 0.3 + building_type * 0.1
            total_consumption = (hvac_usage + lighting_usage + electronics_usage + 
                               base_load + np.random.normal(0, 0.1))
            
            data.append({
                'temperature': temperature,
                'humidity': humidity,
                'occupancy': occupancy,
                'hour': hour,
                'day_of_week': day_of_week,
                'month': month,
                'is_weekend': int(is_weekend),
                'hvac_usage': hvac_usage,
                'lighting_usage': lighting_usage,
                'electronics_usage': electronics_usage,
                'building_type_encoded': building_type,
                'building_size': building_size,
                'energy_consumption': max(0.1, total_consumption)
            })
        
        return pd.DataFrame(data)
    
    def create_features(self, df):
        """Create engineered features"""
        # Interaction features
        df['temp_occupancy'] = df['temperature'] * df['occupancy']
        df['temp_humidity'] = df['temperature'] * df['humidity'] / 100
        
        # Time-based features
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        df['is_peak_hours'] = ((df['hour'] >= 18) & (df['hour'] <= 21)).astype(int)
        
        # Usage ratios
        df['total_usage'] = df['hvac_usage'] + df['lighting_usage'] + df['electronics_usage']
        df['hvac_ratio'] = df['hvac_usage'] / (df['total_usage'] + 0.001)
        
        return df
    
    def train_model(self):
        """Train the gradient boosting model"""
        logger.info("Generating training data...")
        df = self.generate_training_data()
        
        logger.info("Creating features...")
        df = self.create_features(df)
        
        # Prepare features and target
        X = df[self.feature_names]
        y = df['energy_consumption']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        logger.info("Training Gradient Boosting model...")
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Calculate training score
        train_score = self.model.score(X_scaled, y)
        logger.info(f"Model trained successfully. R² Score: {train_score:.4f}")
        
        return train_score
    
    def predict(self, input_data):
        """Make prediction"""
        if not self.is_trained:
            return None
        
        try:
            # Create DataFrame from input
            df = pd.DataFrame([input_data])
            
            # Encode building type
            building_type_map = {'residential': 0, 'office': 1, 'retail': 2, 'industrial': 3}
            if 'building_type' in df.columns:
                df['building_type_encoded'] = df['building_type'].map(building_type_map).fillna(1)
                df = df.drop('building_type', axis=1)
            
            # Create engineered features
            df = self.create_features(df)
            
            # Ensure all features are present
            for feature in self.feature_names:
                if feature not in df.columns:
                    df[feature] = 0
            
            # Reorder columns to match training
            X = df[self.feature_names]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            return round(max(0.1, prediction), 3)
        
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None
    
    def get_feature_importance(self):
        """Get feature importance"""
        if not self.is_trained:
            return []
        
        importance_data = list(zip(self.feature_names, self.model.feature_importances_))
        importance_data.sort(key=lambda x: x[1], reverse=True)
        return importance_data[:10]  # Top 10

# Initialize predictor
predictor = EnergyPredictor()

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Energy Consumption Predictor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        .container { 
            max-width: 900px; 
            margin: 0 auto; 
            padding: 20px;
        }
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        .main-card {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }
        .form-group {
            display: flex;
            flex-direction: column;
        }
        label {
            font-weight: 600;
            margin-bottom: 8px;
            color: #555;
        }
        input, select {
            padding: 12px;
            border: 2px solid #e1e5e9;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s ease;
        }
        input:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }
        .predict-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease;
            width: 100%;
        }
        .predict-btn:hover {
            transform: translateY(-2px);
        }
        .predict-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        .result {
            margin-top: 25px;
            padding: 20px;
            border-radius: 10px;
            display: none;
        }
        .result.success {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
        }
        .result.error {
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            color: white;
        }
        .result h3 {
            margin-bottom: 15px;
            font-size: 1.5em;
        }
        .prediction-value {
            font-size: 2.5em;
            font-weight: bold;
            margin: 15px 0;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }
        .info-section {
            background: white;
            border-radius: 15px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .feature-importance {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }
        .feature-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 12px;
            background: #f8f9fa;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }
        @media (max-width: 768px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
            .header h1 {
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏢 Smart Energy Predictor</h1>
            <p>AI-Powered Energy Consumption Forecasting</p>
        </div>

        <div class="main-card">
            <form id="predictionForm">
                <div class="form-grid">
                    <div class="form-group">
                        <label for="temperature">Temperature (°C)</label>
                        <input type="number" id="temperature" name="temperature" value="22" step="0.1" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="humidity">Humidity (%)</label>
                        <input type="number" id="humidity" name="humidity" value="50" min="0" max="100" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="occupancy">Number of People</label>
                        <input type="number" id="occupancy" name="occupancy" value="5" min="0" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="hour">Hour (0-23)</label>
                        <input type="number" id="hour" name="hour" value="14" min="0" max="23" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="day_of_week">Day of Week</label>
                        <select id="day_of_week" name="day_of_week" required>
                            <option value="0">Monday</option>
                            <option value="1">Tuesday</option>
                            <option value="2" selected>Wednesday</option>
                            <option value="3">Thursday</option>
                            <option value="4">Friday</option>
                            <option value="5">Saturday</option>
                            <option value="6">Sunday</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="month">Month</label>
                        <select id="month" name="month" required>
                            <option value="1">January</option>
                            <option value="2">February</option>
                            <option value="3">March</option>
                            <option value="4">April</option>
                            <option value="5">May</option>
                            <option value="6" selected>June</option>
                            <option value="7">July</option>
                            <option value="8">August</option>
                            <option value="9">September</option>
                            <option value="10">October</option>
                            <option value="11">November</option>
                            <option value="12">December</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="hvac_usage">HVAC Usage (kW)</label>
                        <input type="number" id="hvac_usage" name="hvac_usage" value="1.5" step="0.1" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="lighting_usage">Lighting Usage (kW)</label>
                        <input type="number" id="lighting_usage" name="lighting_usage" value="0.3" step="0.1" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="electronics_usage">Electronics Usage (kW)</label>
                        <input type="number" id="electronics_usage" name="electronics_usage" value="0.8" step="0.1" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="building_type">Building Type</label>
                        <select id="building_type" name="building_type" required>
                            <option value="office" selected>Office</option>
                            <option value="residential">Residential</option>
                            <option value="retail">Retail</option>
                            <option value="industrial">Industrial</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="building_size">Building Size (sq ft)</label>
                        <input type="number" id="building_size" name="building_size" value="5000" required>
                    </div>
                </div>
                
                <button type="submit" class="predict-btn">
                    🔮 Predict Energy Consumption
                </button>
            </form>
            
            <div id="result" class="result"></div>
        </div>
        
        <div class="info-section">
            <h3>🎯 Model Performance</h3>
            <p>This AI model uses Gradient Boosting algorithm trained on synthetic energy consumption data with realistic patterns.</p>
            <div id="featureImportance"></div>
        </div>
    </div>
    
    <script>
        // Load feature importance on page load
        fetch('/api/feature-importance')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const container = document.getElementById('featureImportance');
                    container.innerHTML = '<h4>Top Important Features:</h4><div class="feature-importance"></div>';
                    const grid = container.querySelector('.feature-importance');
                    
                    data.features.forEach(([feature, importance]) => {
                        const item = document.createElement('div');
                        item.className = 'feature-item';
                        item.innerHTML = `<span>${feature.replace('_', ' ')}</span><span>${importance.toFixed(3)}</span>`;
                        grid.appendChild(item);
                    });
                }
            });

        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const submitBtn = document.querySelector('.predict-btn');
            const resultDiv = document.getElementById('result');
            
            // Disable button and show loading
            submitBtn.disabled = true;
            submitBtn.textContent = '🔄 Predicting...';
            
            const formData = new FormData(this);
            const data = Object.fromEntries(formData.entries());
            
            // Convert numeric fields
            const numericFields = ['temperature', 'humidity', 'occupancy', 'hour', 'day_of_week', 
                                 'month', 'hvac_usage', 'lighting_usage', 'electronics_usage', 'building_size'];
            numericFields.forEach(field => {
                data[field] = parseFloat(data[field]);
            });
            
            // Add derived fields
            data['is_weekend'] = (data['day_of_week'] >= 5) ? 1 : 0;
            
            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (result.success) {
                    resultDiv.className = 'result success';
                    resultDiv.innerHTML = `
                        <h3>✅ Prediction Result</h3>
                        <div class="prediction-value">${result.prediction} kWh</div>
                        <p>Based on your building conditions and usage patterns, this is the predicted hourly energy consumption.</p>
                    `;
                } else {
                    resultDiv.className = 'result error';
                    resultDiv.innerHTML = `<h3>❌ Error</h3><p>${result.error}</p>`;
                }
                
                resultDiv.style.display = 'block';
                resultDiv.scrollIntoView({ behavior: 'smooth' });
                
            } catch (error) {
                resultDiv.className = 'result error';
                resultDiv.innerHTML = `<h3>❌ Error</h3><p>Network error: ${error.message}</p>`;
                resultDiv.style.display = 'block';
            } finally {
                // Re-enable button
                submitBtn.disabled = false;
                submitBtn.textContent = '🔮 Predict Energy Consumption';
            }
        });
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    """Main application page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    try:
        input_data = request.get_json()
        
        if not input_data:
            return jsonify({'success': False, 'error': 'No input data provided'})
        
        prediction = predictor.predict(input_data)
        
        if prediction is None:
            return jsonify({'success': False, 'error': 'Prediction failed'})
        
        return jsonify({
            'success': True,
            'prediction': prediction
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/feature-importance', methods=['GET'])
def feature_importance():
    """Get feature importance data"""
    try:
        features = predictor.get_feature_importance()
        return jsonify({
            'success': True,
            'features': features
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for deployment platforms"""
    return jsonify({
        'status': 'healthy',
        'model_trained': predictor.is_trained,
        'service': 'Energy Consumption Predictor'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
