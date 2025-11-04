# MindMetrics — Digital Twin of Stress Monitoring System

## Overview

MindMetrics is an intelligent stress prediction and wellness monitoring system that leverages a Digital Twin approach to assess user stress levels using physiological and behavioral data. The system analyzes key health indicators including heart rate, sleep duration, step count, sleep quality, and activity levels to provide real-time stress assessments and personalized wellness recommendations.

The Digital Twin methodology creates a virtual representation of an individual's health state, enabling continuous monitoring and predictive analysis. By combining machine learning algorithms with heuristic health assessment techniques, MindMetrics delivers accurate stress predictions on a scale of 0-10, categorizes stress levels, and generates actionable health suggestions tailored to each user's unique profile.

This system is designed to support proactive health management by identifying stress patterns early and recommending evidence-based interventions to improve overall well-being.

## Features

- **Real-time Stress Prediction**: Analyzes physiological and behavioral inputs to predict stress levels on a standardized 0-10 scale with up to 93.43% accuracy
- **Multiple ML Models**: Choose from 4 different algorithms (Decision Tree, Logistic Regression, Random Forest, XGBoost) based on your needs
- **Digital Twin Modeling**: Creates a personalized health profile that adapts to individual user characteristics
- **Personalized Wellness Suggestions**: Generates context-aware recommendations based on predicted stress levels and input parameters
- **Hybrid Prediction System**: Combines best-in-class ML models (93.43% accuracy) with heuristic health assessment for robust predictions
- **Input Validation**: Comprehensive validation of all user inputs with medical reference ranges
- **Confidence Scoring**: Dynamic prediction confidence (60-98%) based on multiple physiological factors
- **Interactive User Interface**: Clean, responsive frontend with dark mode support
- **Model Performance Tracking**: Real-time display of which model is being used and its accuracy
- **RESTful API**: Well-documented FastAPI backend with Swagger UI for testing and integration
- **Cross-Origin Support**: CORS-enabled for seamless frontend-backend communication

## Technology Stack

### Frontend
- **HTML5**: Semantic markup for structured content presentation
- **CSS3**: Responsive styling with modern design principles
- **JavaScript (ES6)**: Client-side logic for form handling and API communication

### Backend
- **FastAPI**: High-performance Python web framework for building APIs
- **Python 3.9+**: Core programming language for backend logic
- **Uvicorn**: ASGI server for running FastAPI applications
- **Pydantic**: Data validation and settings management using Python type annotations

### Machine Learning
- **scikit-learn**: Random Forest Classifier for stress prediction
- **pandas**: Data manipulation and preprocessing
- **NumPy**: Numerical computing for feature engineering
- **joblib**: Model serialization and deserialization

## Input Parameters and Output Fields

### Input Parameters

| Parameter | Type | Description | Valid Range / Options |
|-----------|------|-------------|----------------------|
| `age` | Integer | User's age in years | 1-120 years |
| `gender` | String | User's gender | Male, Female, Other, Prefer not to say |
| `heart_rate` | Float | Heart rate in beats per minute | 40-200 BPM |
| `sleep_duration` | Float | Sleep duration in hours | 0-24 hours |
| `step_count` | Integer | Daily step count | 0-50000 steps |
| `sleep_quality` | String | Subjective sleep quality assessment | Poor, Average, Good |
| `activity_level` | String | Daily physical activity level | Sedentary, Active, Highly Active |

### Output Fields

| Field | Type | Description |
|-------|------|-------------|
| `predicted_stress` | Float | Predicted stress level on 0-10 scale (0=Relaxed, 10=Extremely Stressed) |
| `stress_category` | String | Categorical stress classification (Low Stress, Moderate Stress, High Stress) |
| `suggestion` | String | Personalized wellness recommendation based on stress level and input parameters |
| `confidence` | Float | Prediction confidence score (60-98%) based on input consistency and physiological indicators |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check endpoint to verify API status |
| `GET` | `/model-info` | Retrieve detailed information about the ML model, accuracy, and prediction methodology |
| `GET` | `/health` | Comprehensive health check with model status and available endpoints |
| `POST` | `/predict` | Primary prediction endpoint accepting user health data and returning stress assessment |
| `POST` | `/debug_predict` | Debug endpoint providing raw model outputs, scaled values, and feature details for testing |

## Installation and Setup

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git (for cloning the repository)
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Quick Setup (Windows - Recommended)

#### Using Automated Scripts (Easiest)

1. **Open PowerShell in project directory**
   ```powershell
   cd MindMetrics-1
   ```

2. **Run setup script**
   ```powershell
   .\setup.ps1
   ```

3. **Start the backend server**
   ```powershell
   .\run.ps1
   ```

4. **Open the frontend**
   - Navigate to `frontend` folder
   - Double-click `index.html` or open in browser

**Note:** If you get execution policy errors, run PowerShell as Administrator and execute:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step-by-Step Setup Guide (Manual)

#### 1. Clone/Download the Repository

If you have the project folder, navigate to it. Otherwise:
```bash
git clone https://github.com/yourusername/MindMetrics.git
cd MindMetrics-1
```

#### 2. Create Virtual Environment

**Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies

```powershell
cd backend
pip install -r requirements.txt
```

The following packages will be installed:
- fastapi==0.104.1
- uvicorn==0.24.0
- pydantic==2.5.0
- joblib==1.3.2
- pandas==2.1.3
- scikit-learn==1.3.2
- numpy==1.26.2

#### 4. Run the Backend Server

**Method 1: Direct Python**
```powershell
python main.py
```

**Method 2: Using Uvicorn**
```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

API documentation (Swagger UI) will be available at: `http://localhost:8000/docs`

#### 5. Open the Frontend

**Windows:**
```powershell
cd ..\frontend
start index.html
```

**macOS:**
```bash
cd ../frontend
open index.html
```

**Linux:**
```bash
cd ../frontend
xdg-open index.html
```

Alternatively, you can:
- Double-click `index.html` in File Explorer
- Use VS Code's Live Server extension
- Use any local web server

#### 6. Use the Application

1. Fill in all required health metrics in the web interface
2. Select your preferred ML model (Random Forest is default)
3. Click the "Analyze Stress Level" button
4. View your stress prediction, category, and personalized wellness suggestions
5. Note the confidence score indicating prediction reliability

## Folder Structure

```
MindMetrics-1/
│
├── backend/
│   ├── main.py                  # FastAPI application with all endpoints and logic
│   ├── analyze_data.py          # Data analysis and scaling parameter calculation
│   ├── create_scaler.py         # Scaler creation utilities
│   ├── requirements.txt         # Python dependencies
│   └── __pycache__/             # Python cache files (ignored by git)
│
├── frontend/
│   ├── index.html               # Main HTML structure and form
│   ├── style.css                # Styling and responsive design
│   └── script.js                # Client-side JavaScript for API interaction
│
├── models/
│   ├── random_forest_model.joblib        # Trained Random Forest model
│   ├── decision_tree_model.joblib        # Trained Decision Tree model
│   ├── logistic_regression_model.joblib  # Trained Logistic Regression model
│   └── xgboost_pipeline_model.joblib     # Trained XGBoost model
│
├── cleaned_stress_data.csv                           # Preprocessed training dataset
├── DPDEL-FORM (Responses) - Form responses 1.csv    # Raw survey response data
├── dpel_final.ipynb                                 # Jupyter notebook for model training and analysis
├── model_evaluation_scores.csv                      # Model performance metrics
├── .gitignore                   # Git ignore rules
└── README.md                    # Project documentation (this file)
```

## Model Information

### Available ML Models & Performance Metrics

Based on actual training data evaluation (see `model_evaluation_scores.csv`):

#### 1. 🌳 Decision Tree Classifier (RECOMMENDED - HIGHEST ACCURACY)
- **Accuracy:** 93.43%
- **F1-Score:** 93.67%
- **Best For:** Clinical applications, medical diagnostics, understanding decision factors
- **Advantages:** Most accurate, highly interpretable, shows exact decision paths
- **Use When:** Accuracy is paramount, need to explain predictions to patients/doctors
- **Speed:** Fast predictions

#### 2. 📊 Logistic Regression (RECOMMENDED - PRODUCTION)
- **Accuracy:** 93.43%
- **F1-Score:** 92.71%
- **Best For:** Production systems, mobile apps, real-time applications
- **Advantages:** Fastest predictions, simple deployment, excellent accuracy
- **Use When:** Speed is critical, large-scale deployments needed
- **Speed:** Fastest (linear complexity)

#### 3. 🌲 Random Forest Classifier (ROBUST)
- **Accuracy:** 82.85%
- **F1-Score:** 79.47%
- **Best For:** General purpose stress prediction, handling noisy or incomplete data
- **Advantages:** Most robust against overfitting, handles missing values well
- **Use When:** Balanced performance needed, data quality varies
- **Speed:** Moderate (ensemble of trees)

#### 4. ⚡ XGBoost Classifier (ADVANCED)
- **Accuracy:** ~87-90% (estimated)
- **F1-Score:** ~85-88% (estimated)
- **Best For:** Research purposes, complex pattern recognition, maximum sophistication
- **Advantages:** Industry-standard for competitions, handles complex interactions
- **Use When:** Maximum model complexity required, research/analysis focus
- **Speed:** Slower (gradient boosting)

### Model Selection Guide

**For Medical/Clinical Use:** Decision Tree (93.43% accuracy + interpretable decision paths)

**For Mobile/Production Apps:** Logistic Regression (93.43% accuracy + fastest predictions)

**For General Purpose:** Random Forest (82.85% accuracy + handles noisy data well)

**For Research/Analysis:** XGBoost (~90% accuracy + captures complex patterns)

### Algorithm Details

**Decision Tree Classifier:**
- Algorithm: CART (Classification and Regression Trees)
- 18 engineered features including sleep efficiency and activity ratios

**Training Data:**
- Dataset: Health and lifestyle survey responses (DPEL dataset)
- Features: Age, heart rate, sleep patterns, activity levels, physiological measurements
- Target: Self-reported stress levels (0-10 scale)

### Feature Engineering

The models incorporate derived features:
- **Sleep efficiency:** Ratio of sleep duration to sleep quality
- **Activity-to-stress ratio:** Physical activity normalized by stress indicators
- **Categorical encoding:** Gender, sleep quality, activity level
- **Standard scaling:** Applied using training data statistics for consistency

### Prediction Method

**Hybrid Approach:** 
- 60% Heuristic health assessment (based on medical guidelines)
- 40% ML model prediction (using selected algorithm)
- Confidence scoring based on physiological indicator consistency (60-98%)

This hybrid approach ensures realistic, varied predictions that account for both data-driven patterns and established medical knowledge.

## Future Enhancements

1. **Advanced Analytics Dashboard**: Implement comprehensive visualization of historical stress trends and patterns over time

2. **Multi-Modal Data Integration**: Expand input parameters to include dietary habits, caffeine intake, social interaction metrics, and environmental factors

3. **Real-Time Wearable Integration**: Connect with fitness trackers and smartwatches for automatic data collection (Fitbit, Apple Watch, Garmin)

4. **Longitudinal Tracking**: Develop user profile system with secure authentication to track stress levels over weeks and months

5. **Predictive Alerts**: Implement proactive notifications when stress indicators suggest elevated risk based on historical patterns

6. **Enhanced ML Models**: Explore deep learning architectures (LSTM, Transformer) for temporal pattern recognition and improved accuracy

7. **Personalized Intervention Plans**: Generate detailed, multi-day wellness programs tailored to individual stress profiles and preferences

8. **Clinical Integration**: HIPAA-compliant version for healthcare provider use with secure patient data management

9. **Mobile Application**: Develop native iOS and Android applications for improved accessibility and push notifications

10. **Explainable AI**: Integrate SHAP or LIME for interpretable predictions showing which factors contribute most to stress levels

## Author

**Harshad Agrawal**  
Symbiosis Institute of Technology  
B.Tech in Computer Science and Engineering

**Project Context:**  
This project was developed as part of the DPEL (Digital Product Engineering Lab) Phase-2 Mini Project Evaluation. It demonstrates the application of machine learning, API development, and user interface design in creating a practical health monitoring solution using Digital Twin methodology.

**Key Achievements:**
- ✅ Achieved 93.43% prediction accuracy using Decision Tree and Logistic Regression models
- ✅ Implemented hybrid prediction system combining ML (40%) + heuristic analysis (60%)
- ✅ Developed RESTful API with comprehensive documentation (Swagger UI)
- ✅ Created responsive web interface with dark mode support
- ✅ Integrated 4 different ML models with real-time selection capability
- ✅ Implemented dynamic confidence scoring (60-98%) based on physiological indicators

**Academic Year:** 2024-2025

**GitHub Repository:** [https://github.com/Harshad2321/MindMetrics](https://github.com/Harshad2321/MindMetrics)

## License

This project is an academic submission and is intended for educational purposes. For any commercial use or redistribution, please contact the author.

## Acknowledgments

Special thanks to the DPEL faculty and mentors at Symbiosis Institute of Technology for their guidance and support throughout the development of this project.

---

**Last Updated:** November 2025
