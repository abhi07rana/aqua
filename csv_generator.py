import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

class AquacultureDatasetGenerator:
    def __init__(self, num_records=10000):
        """
        Initialize dataset generator with comprehensive configurations
        """
        self.num_records = num_records
        
        # Expanded system types with more detailed categorization
        self.system_types = [
            'Coastal Extensive Pond', 
            'Coastal Intensive Pond', 
            'Offshore Cage System', 
            'Deep Sea Cage System',
            'Recirculating Aquaculture System (RAS) - Small Scale',
            'Recirculating Aquaculture System (RAS) - Industrial Scale',
            'Integrated Multi-Trophic Aquaculture (IMTA) - Coastal',
            'Integrated Multi-Trophic Aquaculture (IMTA) - Offshore',
            'Inland Raceway System',
            'Reservoir Aquaculture'
        ]
        
        # More diverse and specific species
        self.species = [
            'Atlantic Salmon', 
            'Chinook Salmon', 
            'Rainbow Trout', 
            'European Sea Bass', 
            'Gilthead Sea Bream', 
            'Tilapia (Nile)', 
            'Tilapia (Blue)',
            'Black Tiger Shrimp', 
            'White Leg Shrimp', 
            'Giant River Prawn',
            'European Eel', 
            'Carp (Common)', 
            'Carp (Silver)', 
            'Catfish (Channel)', 
            'Barramundi'
        ]
        
        # More geographically diverse locations
        self.locations = [
            'Norway', 
            'Chile', 
            'Scotland', 
            'Canada (British Columbia)', 
            'Canada (Atlantic)', 
            'USA (Hawaii)', 
            'USA (Gulf Coast)', 
            'Indonesia', 
            'Vietnam', 
            'Thailand', 
            'China (Southern Coast)', 
            'China (Inland Provinces)', 
            'Brazil', 
            'Ecuador', 
            'Greece', 
            'Turkey', 
            'Spain', 
            'Portugal', 
            'Australia (West Coast)', 
            'Australia (East Coast)'
        ]
        
        # Enhanced market and economic parameters
        self.market_trends = {
            'Salmon Species': {
                'base_price': 5.5,
                'price_volatility': 0.15,
                'demand_growth_rate': 0.03,
                'production_costs': 3.2
            },
            'Trout Species': {
                'base_price': 4.8,
                'price_volatility': 0.12,
                'demand_growth_rate': 0.02,
                'production_costs': 2.9
            },
            'Sea Bass/Bream': {
                'base_price': 6.2,
                'price_volatility': 0.18,
                'demand_growth_rate': 0.025,
                'production_costs': 4.1
            },
            'Tilapia': {
                'base_price': 3.5,
                'price_volatility': 0.10,
                'demand_growth_rate': 0.04,
                'production_costs': 2.1
            },
            'Shrimp': {
                'base_price': 7.0,
                'price_volatility': 0.20,
                'demand_growth_rate': 0.035,
                'production_costs': 4.5
            },
            'Eel/Carp/Catfish': {
                'base_price': 4.0,
                'price_volatility': 0.13,
                'demand_growth_rate': 0.02,
                'production_costs': 2.5
            }
        }
        
    def generate_environmental_data(self):
        """
        Generate realistic environmental data with expanded complexity
        """
        # Water Temperature (more nuanced based on location and season)
        water_temp = np.random.normal(22, 5)  # Broader range
        
        # Salinity with location-specific variations
        salinity = np.random.normal(35, 3)  # More realistic variation
        
        # pH Level with ecosystem-specific ranges
        ph_level = np.random.normal(7.5, 0.5)
        
        # Dissolved Oxygen with depth and system considerations
        dissolved_oxygen = np.random.normal(6, 1.5)
        
        # Nutrient levels (new feature)
        nitrogen = np.random.normal(2.5, 0.5)
        phosphorus = np.random.normal(0.3, 0.1)
        
        return {
            'WaterTemperature': round(water_temp, 2),
            'Salinity': round(salinity, 2),
            'PHLevel': round(ph_level, 2),
            'DissolvedOxygen': round(dissolved_oxygen, 2),
            'NitrogenLevel': round(nitrogen, 2),
            'PhosphorusLevel': round(phosphorus, 2)
        }
    
    def calculate_yield_and_economic_factors(self, system_type, species, location, env_data):
        """
        Calculate yield with advanced multi-factor economic modeling
        """
        # System type yield multipliers
        system_multipliers = {
            'Coastal Extensive Pond': 0.8,
            'Coastal Intensive Pond': 1.2,
            'Offshore Cage System': 1.5,
            'Deep Sea Cage System': 1.7,
            'Recirculating Aquaculture System (RAS) - Small Scale': 1.0,
            'Recirculating Aquaculture System (RAS) - Industrial Scale': 1.4,
            'Integrated Multi-Trophic Aquaculture (IMTA) - Coastal': 1.1,
            'Integrated Multi-Trophic Aquaculture (IMTA) - Offshore': 1.3,
            'Inland Raceway System': 1.2,
            'Reservoir Aquaculture': 0.9
        }
        
        # Species-specific yield factors
        species_factors = {
            'Atlantic Salmon': 1.3,
            'Chinook Salmon': 1.2,
            'Rainbow Trout': 1.1,
            'European Sea Bass': 1.0,
            'Gilthead Sea Bream': 0.9,
            'Tilapia (Nile)': 1.2,
            'Tilapia (Blue)': 1.1,
            'Black Tiger Shrimp': 1.4,
            'White Leg Shrimp': 1.3,
            'Giant River Prawn': 1.2,
            'European Eel': 0.8,
            'Carp (Common)': 1.0,
            'Carp (Silver)': 0.9,
            'Catfish (Channel)': 1.1,
            'Barramundi': 1.2
        }
        
        # Location-specific adjustment factors
        location_factors = {
            'Norway': 1.2,  # Advanced aquaculture infrastructure
            'Chile': 1.1,   # Favorable marine conditions
            'Scotland': 1.0,
            'Canada (British Columbia)': 1.1,
            'Indonesia': 0.9,  # Developing infrastructure
            'Vietnam': 0.9,
            'USA (Gulf Coast)': 1.0,
            # Add more location-specific adjustments
        }
        
        # Determine price category
        price_category = (
            'Salmon Species' if 'Salmon' in species else
            'Trout Species' if 'Trout' in species else
            'Sea Bass/Bream' if 'Bass' in species or 'Bream' in species else
            'Tilapia' if 'Tilapia' in species else
            'Shrimp' if 'Shrimp' in species else
            'Eel/Carp/Catfish'
        )
        
        # Market trend data for the species
        market_trend = self.market_trends[price_category]
        
        # Environmental impact coefficients
        env_coefficients = (
            1 - abs(env_data['WaterTemperature'] - 22)/10 *
            1 - abs(env_data['Salinity'] - 35)/5 *
            1 - abs(env_data['PHLevel'] - 7.5)/1 *
            env_data['DissolvedOxygen']/8
        )
        
        # Base stocking density with more variability
        stocking_density = np.random.uniform(50, 200)
        
        # Location factor adjustment
        location_factor = location_factors.get(location, 1.0)
        
        # Comprehensive yield calculation
        base_yield = (
            system_multipliers.get(system_type, 1.0) * 
            species_factors.get(species, 1.0) * 
            location_factor *
            stocking_density * 
            env_coefficients * 
            np.random.uniform(0.8, 1.2)
        )
        
        # Dynamic market price calculation
        # Incorporate price volatility, demand growth, and production costs
        market_price = (
            market_trend['base_price'] * 
            (1 + market_trend['price_volatility'] * np.random.uniform(-1, 1)) *
            (1 + market_trend['demand_growth_rate'])
        )
        
        # Economic Potential Calculation
        operating_costs = market_trend['production_costs'] * base_yield
        gross_revenue = base_yield * market_price
        economic_potential = gross_revenue - operating_costs
        
        return {
            'StockingDensity': round(stocking_density, 2),
            'FeedConversionRatio': round(np.random.normal(1.5, 0.3), 2),
            'YieldKgPerM3': round(base_yield, 2),
            'MarketPrice': round(market_price, 2),
            'OperatingCosts': round(operating_costs, 2),
            'GrossRevenue': round(gross_revenue, 2),
            'EconomicPotential': round(economic_potential, 2),
            'MarketDemandGrowth': round(market_trend['demand_growth_rate'] * 100, 2)
        }
    
    def generate_dataset(self):
        """
        Generate comprehensive aquaculture dataset
        """
        data = []
        start_date = datetime(2018, 1, 1)
        
        for _ in range(self.num_records):
            # Random selections
            system_type = random.choice(self.system_types)
            species = random.choice(self.species)
            location = random.choice(self.locations)
            current_date = start_date + timedelta(days=random.randint(0, 365*5))
            
            # Generate data components
            env_data = self.generate_environmental_data()
            economic_data = self.calculate_yield_and_economic_factors(system_type, species, location, env_data)
            
            # Combine all data
            record = {
                'Date': current_date,
                'SystemType': system_type,
                'Species': species,
                'Location': location,
                **env_data,
                **economic_data,
                'SustainabilityScore': round(np.random.uniform(50, 100), 2)
            }
            
            data.append(record)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        return df
    
    def save_dataset(self, filename='aquaculture_comprehensive_dataset.csv'):
        """
        Generate and save dataset
        """
        dataset = self.generate_dataset()
        dataset.to_csv(filename, index=False)
        print(f"Dataset generated with {len(dataset)} records.")
        print("\nDataset Overview:")
        print(dataset.describe())
        return dataset

# Generate dataset
generator = AquacultureDatasetGenerator(num_records=10000)
dataset = generator.save_dataset()