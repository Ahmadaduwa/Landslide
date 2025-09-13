# Landslide Detection System

A comprehensive machine learning system for landslide detection and risk assessment using multiple anomaly detection algorithms. This project implements three different approaches: Isolation Forest, LSTM Autoencoders, and Random Forest classifiers to analyze environmental sensor data and predict landslide risks.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Data](#data)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This landslide detection system processes environmental sensor data including soil moisture, rainfall, temperature, humidity, and geological measurements to identify potential landslide risks. The system uses multiple machine learning approaches to provide robust anomaly detection and risk assessment.

## ✨ Features

- **Multi-Model Approach**: Implements three different ML algorithms for comprehensive analysis
- **Real-time Data Processing**: Handles time-series sensor data with proper preprocessing
- **Anomaly Detection**: Identifies unusual patterns that may indicate landslide risks
- **Risk Assessment**: Provides confidence scores and risk levels
- **Data Visualization**: Interactive plots and heatmaps for analysis
- **Model Persistence**: Saves trained models for deployment

## 📁 Project Structure

```
Landslide/
├── data/                          # Dataset files
│   ├── Landslide_dataSet1.csv     # Primary dataset 1
│   ├── Landslide_dataSet2.csv     # Primary dataset 2
│   ├── dev101_prepared.csv        # Device 101 processed data
│   ├── dev102_prepared.csv        # Device 102 processed data
│   └── *_resample*.csv           # Resampled datasets
├── models/                        # Model implementations
│   ├── IsalationForest/          # Isolation Forest model
│   ├── LSTM_Autoencoders/        # LSTM Autoencoder model
│   └── RendomForest.ipynb        # Random Forest model
├── modelFile/                     # Saved model files
│   ├── iso_model.joblib          # Isolation Forest model
│   ├── lstm_autoencoder_model4.h5 # LSTM model
│   └── *.joblib                  # Scaler and other models
├── logs/                         # Training logs
├── DataCleaning.ipynb            # Data preprocessing notebook
└── README.md                     # This file
```

## 📊 Data

The system processes environmental sensor data with the following features:

- **timestamp**: Time of measurement
- **devID**: Device identifier
- **soil_mean**: Soil moisture measurements
- **rain_mean**: Rainfall data
- **temp_mean**: Temperature readings
- **humi_mean**: Humidity levels
- **geo_mean**: Geological measurements
- **hour**: Hour of day (derived feature)

### Data Preprocessing

The `DataCleaning.ipynb` notebook handles:
- Timestamp formatting and cleaning
- Missing value handling
- Outlier removal
- Feature engineering
- Data validation and filtering

## 🤖 Models

### 1. Isolation Forest
- **Location**: `models/IsalationForest/`
- **Purpose**: Unsupervised anomaly detection
- **Features**: Identifies outliers in multi-dimensional sensor data
- **Output**: Anomaly scores and binary classification

### 2. LSTM Autoencoder
- **Location**: `models/LSTM_Autoencoders/`
- **Purpose**: Time-series anomaly detection
- **Features**: 
  - Sequence length: 30 time steps
  - Encoder-decoder architecture
  - Reconstruction error-based detection
- **Output**: Reconstruction errors and anomaly probabilities

### 3. Random Forest
- **Location**: `models/RendomForest.ipynb`
- **Purpose**: Supervised classification
- **Features**: Risk score calculation and classification
- **Output**: Risk levels and confidence scores

## 🚀 Installation

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- Required packages (install via pip):

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow seaborn folium xgboost joblib
```

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Landslide
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

## 💻 Usage

### Data Preprocessing

1. Open `DataCleaning.ipynb`
2. Run all cells to process raw sensor data
3. Generated cleaned datasets will be saved in the `data/` directory

### Model Training

#### Isolation Forest
```bash
cd models/IsalationForest/
jupyter notebook Isolation_Forest.ipynb
```

#### LSTM Autoencoder
```bash
cd models/LSTM_Autoencoders/
jupyter notebook LSTM_Autoencoders.ipynb
```

#### Random Forest
```bash
cd models/
jupyter notebook RendomForest.ipynb
```

### Model Validation

Each model directory contains validation notebooks:
- `Isolation_Forest_Validate*.ipynb`
- `LSTM_Autoencoders_Validate*.ipynb`

### Using Trained Models

```python
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load Isolation Forest model
iso_model = joblib.load('modelFile/iso_model.joblib')
scaler = joblib.load('modelFile/iso_scaler.joblib')

# Load and preprocess new data
new_data = pd.read_csv('new_sensor_data.csv')
scaled_data = scaler.transform(new_data[['soil_mean', 'rain_mean', 'temp_mean', 'humi_mean', 'geo_mean']])

# Make predictions
predictions = iso_model.predict(scaled_data)
anomaly_scores = iso_model.decision_function(scaled_data)
```

## 📈 Results

The system provides:
- **Anomaly Detection**: Binary classification of normal vs. anomalous readings
- **Risk Scoring**: Continuous risk scores for fine-grained assessment
- **Confidence Levels**: Uncertainty quantification for predictions
- **Visualization**: Interactive plots and heatmaps for analysis

### Performance Metrics

Each model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC (where applicable)

## 🔧 Configuration

### Model Parameters

- **Isolation Forest**: Contamination rate, number of estimators
- **LSTM Autoencoder**: Sequence length, hidden units, learning rate
- **Random Forest**: Number of trees, max depth, feature selection

### Data Parameters

- Resampling frequency: 1T (1 minute), 10T (10 minutes), 30s (30 seconds)
- Feature scaling: StandardScaler normalization
- Validation split: 80/20 train/test

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

For questions or support, please open an issue in the repository.

---

**Note**: This system is designed for research and educational purposes. For production landslide monitoring, ensure proper validation and integration with professional geological assessment systems.
