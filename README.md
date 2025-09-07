# Smart-Energy-Consumption-Predictor
An AI-powered web application that predicts energy consumption for buildings using machine learning. Built with Flask, scikit-learn, and a modern responsive UI.
🌟 Features

**Real-time Predictions**: Get instant energy consumption forecasts

**Smart ML Model**: Uses Gradient Boosting algorithm with engineered features

**Responsive Design**: Works perfectly on desktop and mobile devices

**Multiple Building Types**: Supports residential, office, retail, and industrial buildings

**Feature Importance**: Visualize which factors most impact energy consumption

**Easy Deployment**: Ready for major cloud platforms

# 🚀 Quick Start
Local Development

### 1 .Clone the repository
git clone https://github.com/yourusername/energy-consumption-predictor.git

cd energy-consumption-predictor

### 2.Install dependencies

pip install -r requirements.txt

### 3.Run the application

python app.py

# 🤖 Model Details

**Algorithm** : Gradient Boosting Regressor

**Features**: 18 engineered features including temperature, occupancy, time patterns

**Training Data** : 2000+ synthetic samples with realistic energy patterns

**Performance**: R² Score > 0.85 on training data

# 🎯 API Endpoints

GET / - Main application interface

POST /api/predict - Get energy consumption prediction

GET /api/feature-importance - Get model feature importance

GET /health - Health check endpoin
