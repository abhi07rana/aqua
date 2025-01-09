# AquaForecast All™

AquaForecast All™ is an advanced predictive analytics tool designed for the aquaculture industry. It uses machine learning to forecast key metrics like yield, economic potential, and sustainability scores for aquatic systems. The project is built in Python, leveraging Flask for API deployment and powerful libraries for data processing and model training.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [How It Works](#how-it-works)
- [Endpoints](#endpoints)
- [Deployment](#deployment)
- [Usage](#usage)
- [Dataset Information](#dataset-information)
- [GitHub Repository](#github-repository)
- [Contributing](#contributing)
- [License](#license)

---

## Features
- Predictive modeling using Random Forest regression for aquaculture metrics.
- Feature importance visualizations to help identify critical factors.
- API endpoints for training, prediction, dataset upload, and market trend analysis.
- Recommendations for optimizing aquaculture parameters.

---

## Installation

### Prerequisites
- Python 3.7+
- A virtual environment (recommended)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/abhi07rana/aqua.git
   cd aqua
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Required Python Libraries
These libraries are listed in `requirements.txt`:
- Flask
- Flask-CORS
- pandas
- scikit-learn
- numpy
- joblib
- matplotlib
- seaborn
- gunicorn

---

## How It Works

### Data Collection
- **Datasets**: The tool uses datasets from NOAA for environmental data and genetic datasets for aquatic species (e.g., [FishBase](https://www.fishbase.org)).

### Data Preparation
- **Feature Engineering**: Features like water temperature, salinity, and dissolved oxygen are preprocessed and encoded for model training.

### Predictive Modeling
- **Model**: Random Forest Regression.
- **Targets**:
  - Yield (Kg per m³).
  - Economic potential.
  - Sustainability scores.

### Deployment
- The project is deployed using Flask and hosted on Render. Check the live version [here](https://aqua-forest.onrender.com).

---

## Endpoints

### Base URL
`https://aqua-forest.onrender.com`

### API Routes

1. **Home**: `/`
   - Method: `GET`
   - Description: Returns a welcome message and available endpoints.

2. **Upload Dataset**: `/upload_dataset`
   - Method: `POST`
   - Description: Upload a dataset for training.
   - Example Request:
     ```bash
     curl -X POST -F "file=@path/to/dataset.csv" https://aqua-forest.onrender.com/upload_dataset
     ```

3. **Train Model**: `/train`
   - Method: `POST`
   - Description: Train a predictive model for a specified target.
   - Example Request:
     ```bash
     curl -X POST -H "Content-Type: application/json" -d '{"target": "YieldKgPerM3"}' https://aqua-forest.onrender.com/train
     ```

4. **Predict**: `/predict`
   - Method: `POST`
   - Description: Make predictions using the trained model.
   - Example Request:
     ```bash
     curl -X POST -H "Content-Type: application/json" -d '{"WaterTemperature": 25, "Salinity": 30, ...}' https://aqua-forest.onrender.com/predict
     ```

5. **Feature Importance**: `/feature_importance`
   - Method: `GET`
   - Description: Get feature importance visualization for a specific target.

6. **Market Trend Analysis**: `/market_trend_analysis`
   - Method: `GET`
   - Description: Analyze market trends based on the uploaded dataset.

---

## Deployment

### Local Deployment

1. Run the Flask app locally:
   ```bash
   python api.py
   ```

2. Access the app at `http://127.0.0.1:8080`.

### Render Deployment

The application is deployed on Render. Visit the live version [here](https://aqua-forest.onrender.com).

---

## Usage

1. **Upload Dataset**: Start by uploading a dataset using the `/upload_dataset` endpoint.
2. **Train Model**: Train models for different targets like `YieldKgPerM3`.
3. **Predict**: Use the `/predict` endpoint to get predictions for custom inputs.
4. **Analyze**: Generate feature importance plots and market trend analyses.

---

## Dataset Information
- The project uses the dataset: `aquaculture_comprehensive_dataset.csv`.
- Ensure this dataset is placed in the appropriate directory or uploaded via the `/upload_dataset` endpoint for model training.

---

## GitHub Repository
The complete source code is available on GitHub:
[AquaForecast All™ GitHub Repository](https://github.com/abhi07rana/aqua)

---

## Contributing

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.

### Guidelines
- Write clean and commented code.
- Ensure backward compatibility with existing API endpoints.
- Add or update unit tests for all new features.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

