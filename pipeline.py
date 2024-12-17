import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class AquaculturePredictor:
    def __init__(self, dataset_path):
        """
        Initialize the Aquaculture Prediction Model
        
        :param dataset_path: Path to the aquaculture dataset
        """
        # Setup logging
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Validate dataset path
        if not os.path.exists(dataset_path):
            self.logger.error(f"Dataset file not found: {dataset_path}")
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        try:
            # Load dataset
            self.df = pd.read_csv(dataset_path, parse_dates=['Date'])
            self.logger.info(f"Dataset loaded successfully: {dataset_path}")
            
            # Feature preparation
            self.prepare_features()
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise
    
    def prepare_features(self):
        """
        Prepare features for modeling
        """
        try:
            # Categorical encoding
            self.label_encoders = {}
            categorical_cols = ['SystemType', 'Species', 'Location']
            
            for col in categorical_cols:
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
            
            # Feature selection
            self.features = [
                'WaterTemperature', 'Salinity', 'PHLevel', 
                'DissolvedOxygen', 'NitrogenLevel', 'PhosphorusLevel',
                'StockingDensity', 'SystemType_encoded', 
                'Species_encoded', 'Location_encoded'
            ]
            
            # Target variables
            self.targets = [
                'YieldKgPerM3', 
                'EconomicPotential', 
                'SustainabilityScore'
            ]
            
            self.logger.info("Features prepared successfully")
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            raise
    
    def train_model(self, target='YieldKgPerM3', test_size=0.2):
        """
        Train machine learning models
        
        :param target: Target variable to predict
        :param test_size: Proportion of test dataset
        :return: Model performance metrics
        """
        try:
            # Ensure target is valid
            if target not in self.targets:
                raise ValueError(f"Invalid target. Choose from {self.targets}")
            
            # Prepare X and y
            X = self.df[self.features]
            y = self.df[target]
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest
            rf_model = RandomForestRegressor(
                n_estimators=100, 
                random_state=42
            )
            rf_model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = rf_model.predict(X_test_scaled)
            
            # Performance metrics
            metrics = {
                'mean_squared_error': mean_squared_error(y_test, y_pred),
                'mean_absolute_error': mean_absolute_error(y_test, y_pred),
                'r2_score': r2_score(y_test, y_pred)
            }
            
            # Cross-validation
            cv_scores = cross_val_score(
                rf_model, X_train_scaled, y_train, 
                cv=5, scoring='neg_mean_squared_error'
            )
            metrics['cross_val_scores'] = -cv_scores.mean()
            
            # Ensure models directory exists
            os.makedirs('models', exist_ok=True)
            
            # Save model and scaler
            joblib.dump(rf_model, f'models/{target}_model.pkl')
            joblib.dump(scaler, f'models/{target}_scaler.pkl')
            joblib.dump(self.label_encoders, 'models/label_encoders.pkl')
            
            self.logger.info(f"Model for {target} trained and saved successfully")
            return metrics
        
        except Exception as e:
            self.logger.error(f"Error training model for {target}: {e}")
            raise
    
    def generate_feature_importance(self, target='YieldKgPerM3'):
        """
        Generate feature importance visualization
        
        :param target: Target variable
        :return: Feature importance plot path
        """
        try:
            # Ensure models directory exists
            os.makedirs('visualizations', exist_ok=True)
            
            # Load model
            model = joblib.load(f'models/{target}_model.pkl')
            
            # Create feature importance plot
            plt.figure(figsize=(10, 6))
            feature_imp = pd.Series(
                model.feature_importances_, 
                index=self.features
            ).sort_values(ascending=False)
            
            sns.barplot(x=feature_imp.values, y=feature_imp.index)
            plt.title(f'Feature Importance for {target} Prediction')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            plot_path = f'visualizations/{target}_feature_importance.png'
            plt.savefig(plot_path)
            plt.close()
            
            self.logger.info(f"Feature importance plot generated: {plot_path}")
            return plot_path
        
        except Exception as e:
            self.logger.error(f"Error generating feature importance plot: {e}")
            raise
    
    def generate_recommendations(self, input_data, predicted_yield):
            
            """
            Generate actionable recommendations based on input features and predicted yield.
            
            :param input_data: Input features
            :param predicted_yield: Predicted yield value
            :return: List of recommendations
            """
            recommendations = []
            
            # Example checks for water temperature
            if input_data.get('WaterTemperature') < 20:
                recommendations.append("Increase water temperature to improve yield.")
            elif input_data.get('WaterTemperature') > 28:
                recommendations.append("Lower water temperature to prevent stress on species.")
            
            # Check dissolved oxygen levels
            if input_data.get('DissolvedOxygen') < 5:
                recommendations.append("Increase dissolved oxygen levels for better fish health.")
            
            # Check pH level
            if input_data.get('PHLevel') < 6.5 or input_data.get('PHLevel') > 8.5:
                recommendations.append("Adjust pH levels to the optimal range (6.5 - 8.5).")
            
            # Check stocking density
            if input_data.get('StockingDensity') > 100:
                recommendations.append("Reduce stocking density to prevent overcrowding.")

            # Default message if no recommendations
            if not recommendations:
                recommendations.append("System parameters are within optimal range.")

            return recommendations
    
    
    def predict(self, input_data, target='YieldKgPerM3'):
        """
        Make predictions using the trained model
        
        :param input_data: Input features for prediction
        :param target: Model to use for prediction
        :return: Prediction results
        """
        try:
            # Validate input data
            required_features = [
                'WaterTemperature', 'Salinity', 'PHLevel', 
                'DissolvedOxygen', 'NitrogenLevel', 'PhosphorusLevel',
                'StockingDensity', 'SystemType', 'Species', 'Location', 
                'OperatingCosts', 'GrossRevenue', 'MarketDemandGrowth'
            ]
            
            for feature in required_features:
                if feature not in input_data:
                    raise ValueError(f"Missing required feature: {feature}")
            
            # Load model and scaler
            model = joblib.load(f'models/{target}_model.pkl')
            scaler = joblib.load(f'models/{target}_scaler.pkl')
            label_encoders = joblib.load('models/label_encoders.pkl')
            
            # Encode categorical variables
            for col in ['SystemType', 'Species', 'Location']:
                input_data[f'{col}_encoded'] = label_encoders[col].transform([input_data[col]])[0]
            
            # Prepare input data
            input_df = pd.DataFrame([input_data])
            input_features = input_df[self.features]
            
            # Scale and predict
            input_scaled = scaler.transform(input_features)
            prediction = model.predict(input_scaled)
            
            # Generate comprehensive insights
            insights = self.generate_comprehensive_insights(input_data, prediction[0])
            
            return {
                'prediction': float(prediction[0]),
                'insights': insights
            }
        
        except FileNotFoundError:
            self.logger.error(f"Model for {target} not found. Train the model first.")
            raise ValueError(f"Model for {target} not found. Train the model first.")
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            raise
    
    def generate_comprehensive_insights(self, input_data, predicted_yield):
        """
        Generate comprehensive insights with enhanced economic potential analysis
        
        :param input_data: Input features
        :param predicted_yield: Predicted yield value
        :return: Dictionary of detailed insights
        """
        def economic_potential_analysis(predicted_yield, input_data):
            """
            Comprehensive economic potential analysis with expanded metrics
            """
            # Location-specific economic factors
            location_factors = {
                'Norway': {'market_stability': 1.2, 'infrastructure_score': 1.1},
                'Chile': {'market_stability': 1.0, 'infrastructure_score': 0.9},
                'Scotland': {'market_stability': 1.1, 'infrastructure_score': 1.0},
                'Canada (British Columbia)': {'market_stability': 1.0, 'infrastructure_score': 1.1},
                # Add more locations as needed
                'Default': {'market_stability': 1.0, 'infrastructure_score': 1.0}
            }
            
            # Retrieve location-specific factors
            location = input_data.get('Location', 'Default')
            loc_factors = location_factors.get(location, location_factors['Default'])
            
            # Retrieve economic metrics from input data
            operating_costs = input_data.get('OperatingCosts', 0)
            gross_revenue = input_data.get('GrossRevenue', 0)
            market_demand_growth = input_data.get('MarketDemandGrowth', 0)
            
            # Calculate economic potential
            try:
                # Economic potential calculation with location and market factors
                total_economic_potential = (
                    (gross_revenue - operating_costs) * 
                    loc_factors['market_stability'] * 
                    loc_factors['infrastructure_score'] * 
                    (1 + market_demand_growth/100)
                )
                
                # Economic potential categorization
                if total_economic_potential < 0:
                    potential_rating = "Negative Economic Potential"
                    recommendation = "Urgent need for cost optimization and system redesign"
                elif total_economic_potential < 5000:
                    potential_rating = "Low Economic Potential"
                    recommendation = "Significant improvements needed in cost structure and efficiency"
                elif total_economic_potential < 20000:
                    potential_rating = "Moderate Economic Potential"
                    recommendation = "Good foundation, focus on scaling and targeted improvements"
                elif total_economic_potential < 50000:
                    potential_rating = "High Economic Potential"
                    recommendation = "Strong performance, explore market expansion strategies"
                else:
                    potential_rating = "Exceptional Economic Potential"
                    recommendation = "Excellent system performance, consider replication and investment"
                
                return {
                    "total_economic_potential": round(total_economic_potential, 2),
                    "operating_costs": round(operating_costs, 2),
                    "gross_revenue": round(gross_revenue, 2),
                    "location_market_stability": loc_factors['market_stability'],
                    "market_demand_growth": market_demand_growth,
                    "potential_rating": potential_rating,
                    "recommendation": recommendation
                }
            except Exception as e:
                return {
                    "total_economic_potential": None,
                    "error": str(e),
                    "analysis": "Unable to calculate economic potential"
                }
        

        def yield_interpretation(yield_value):
            """
            Provide a more nuanced interpretation of yield
            """
            interpretations = [
                {"range": (float('-inf'), 10), "category": "Very Low Yield", "description": "Significant challenges in production"},
                {"range": (10, 25), "category": "Low Yield", "description": "Below optimal production levels"},
                {"range": (25, 40), "category": "Moderate Yield", "description": "Acceptable production with room for improvement"},
                {"range": (40, 60), "category": "Good Yield", "description": "Strong production performance"},
                {"range": (60, float('inf')), "category": "Exceptional Yield", "description": "Highly efficient production system"}
            ]
            
            for interp in interpretations:
                if interp["range"][0] <= yield_value < interp["range"][1]:
                    return {
                        "value": round(yield_value, 2),
                        "category": interp["category"],
                        "description": interp["description"]
                    }
                
        def production_efficiency_score(input_data, yield_value):
            """
            Calculate a comprehensive production efficiency score
            """
            # Define weight factors for different parameters
            efficiency_factors = {
                'water_temperature': {
                    'optimal_range': (20, 28),
                    'weight': 0.2
                },
                'dissolved_oxygen': {
                    'optimal_range': (5, 10),
                    'weight': 0.2
                },
                'ph_level': {
                    'optimal_range': (6.5, 8.5),
                    'weight': 0.15
                },
                'stocking_density': {
                    'optimal_range': (50, 100),
                    'weight': 0.15
                },
                'nitrogen_level': {
                    'optimal_range': (1, 3),
                    'weight': 0.15
                },
                'phosphorus_level': {
                    'optimal_range': (0.1, 0.5),
                    'weight': 0.15
                }
            }
            
            def calculate_factor_score(value, optimal_range):
                """Calculate score based on proximity to optimal range"""
                min_val, max_val = optimal_range
                if min_val <= value <= max_val:
                    return 1.0
                elif value < min_val:
                    return max(0, 1 - (min_val - value) / min_val)
                else:
                    return max(0, 1 - (value - max_val) / max_val)
            
            total_score = 0
            factor_mapping = {
                'water_temperature': input_data.get('WaterTemperature', 0),
                'dissolved_oxygen': input_data.get('DissolvedOxygen', 0),
                'ph_level': input_data.get('PHLevel', 0),
                'stocking_density': input_data.get('StockingDensity', 0),
                'nitrogen_level': input_data.get('NitrogenLevel', 0),
                'phosphorus_level': input_data.get('PhosphorusLevel', 0)
            }
            
            for factor, value in factor_mapping.items():
                optimal_range = efficiency_factors[factor]['optimal_range']
                weight = efficiency_factors[factor]['weight']
                total_score += calculate_factor_score(value, optimal_range) * weight
            
            # Normalize and adjust based on yield
            yield_adjustment = min(yield_value / 50, 1)  # Cap adjustment at 1
            efficiency_score = total_score * 100 * yield_adjustment
            
            return {
                "score": round(efficiency_score, 2),
                "rating": (
                    "Excellent" if efficiency_score > 85 else
                    "Very Good" if efficiency_score > 70 else
                    "Good" if efficiency_score > 55 else
                    "Needs Improvement" if efficiency_score > 40 else
                    "Poor"
                )
            }
        

        # Modify insights to use enhanced economic potential analysis
        insights = {
            "yield_analysis": yield_interpretation(predicted_yield),
            "economic_potential": economic_potential_analysis(predicted_yield, input_data),
            "production_efficiency": production_efficiency_score(
                input_data, 
                predicted_yield
            ),
            "system_details": {
                "system_type": input_data.get('SystemType', 'Not Specified'),
                "species": input_data.get('Species', 'Not Specified'),
                "location": input_data.get('Location', 'Not Specified')
            },
            "recommendations": self.generate_recommendations(input_data, predicted_yield)
        }
        
        return insights



# Example usage for demonstration
if __name__ == '__main__':
    try:
        # Train models for different targets
        predictor = AquaculturePredictor('aquaculture_comprehensive_dataset.csv')
        
        for target in ['YieldKgPerM3', 'EconomicPotential', 'SustainabilityScore']:
            print(f"\nTraining model for {target}")
            metrics = predictor.train_model(target)
            print("Model Metrics:", metrics)
            
            # Generate feature importance plot
            predictor.generate_feature_importance(target)
    
    except Exception as e:
        print(f"Error in main execution: {e}")