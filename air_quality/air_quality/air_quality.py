# air_quality_model_training.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

class AirQualityModelTrainer:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_names = []
        self.results = {}
        
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the air quality data"""
        print("Loading and preprocessing data...")
        
        # Read the CSV file
        df = pd.read_csv(file_path, sep=";", decimal=",", header=0)
        
        # Drop unnecessary columns if they exist
        columns_to_drop = ['Unnamed: 15', 'Unnamed: 16']
        existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        if existing_columns_to_drop:
            df.drop(existing_columns_to_drop, axis=1, inplace=True)
        
        # Drop rows with all NaN values
        df.dropna(how='all', inplace=True)
        
        # Handle datetime conversion
        if 'Time' in df.columns and 'Date' in df.columns:
            # Fix time format if needed
            if df['Time'].dtype == 'object':
                df['Time'] = df['Time'].str.replace('.', ':', regex=False)
            
            # Create DateTime by combining Date and Time columns
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
            
            # Delete old columns
            df = df.drop(['Date', 'Time'], axis=1)
            df.set_index('DateTime', inplace=True)
        
        elif 'DateTime' in df.columns:
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            df.set_index('DateTime', inplace=True)
        
        # Handle missing values with imputation
        imputer = SimpleImputer(strategy='median')
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
        
        # Create time-based features
        df_features = df_imputed.copy()
        df_features['hour'] = df_features.index.hour
        df_features['day'] = df_features.index.day
        df_features['month'] = df_features.index.month
        df_features['dayofweek'] = df_features.index.dayofweek
        
        # Define target and features
        target = 'C6H6(GT)'
        target_pollutants = ['CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']
        
        # Check if target exists
        if target not in df_features.columns:
            raise ValueError(f"Target column '{target}' not found in the data")
        
        features = [col for col in df_features.columns if col not in target_pollutants]
        self.feature_names = features
        
        X = df_features[features]
        y = df_features[target]
        
        # Split data chronologically (70% train, 30% test)
        train_size = int(0.7 * len(df_features))
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Data preprocessing complete. Features: {len(features)}")
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """Train multiple regression models"""
        print("Training models...")
        
        models_config = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100, 
                learning_rate=0.1, 
                random_state=42
            )
        }
        
        # Train all models
        for name, model in models_config.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.models[name] = model
        
        print("Model training complete!")
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate trained models and return performance metrics"""
        print("Evaluating models...")
        
        for name, model in self.models.items():
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.results[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2
            }
            
            print(f"{name}: RMSE={rmse:.4f}, R²={r2:.4f}")
        
        return self.results
    
    def create_plots(self):
        """Create visualization plots for model comparison"""
        results_df = pd.DataFrame(self.results).T
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # MSE comparison
        axes[0, 0].bar(results_df.index, results_df['MSE'])
        axes[0, 0].set_title('Mean Squared Error (MSE)')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # RMSE comparison
        axes[0, 1].bar(results_df.index, results_df['RMSE'])
        axes[0, 1].set_title('Root Mean Squared Error (RMSE)')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        axes[1, 0].bar(results_df.index, results_df['MAE'])
        axes[1, 0].set_title('Mean Absolute Error (MAE)')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # R² comparison
        axes[1, 1].bar(results_df.index, results_df['R²'])
        axes[1, 1].set_title('R² Score')
        axes[1, 1].set_ylabel('R²')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        plt.show()
        
        return fig
    
    def save_models(self, save_dir='saved_models'):
        """Save trained models and scaler"""
        print("Saving models...")
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save each model
        for name, model in self.models.items():
            filename = f"{name.replace(' ', '_').lower()}_model.pkl"
            filepath = os.path.join(save_dir, filename)
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {name} to {filepath}")
        
        # Save scaler
        scaler_path = os.path.join(save_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Saved scaler to {scaler_path}")
        
        # Save feature names
        features_path = os.path.join(save_dir, 'feature_names.pkl')
        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_names, f)
        print(f"Saved feature names to {features_path}")
        
        # Save results
        results_path = os.path.join(save_dir, 'model_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"Saved results to {results_path}")
    
    def get_best_model(self):
        """Get the best performing model based on R² score"""
        if not self.results:
            return None
        
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['R²'])
        return best_model_name, self.models[best_model_name]

def main():
    """Main function to run the complete training pipeline"""
    # Initialize trainer
    trainer = AirQualityModelTrainer()
    
    # Load and preprocess data (replace with your file path)
    file_path = r'\Users\91981\OneDrive\Documents\air_quality\Datasets\AirQuality.csv'  # Replace with actual file path
    
    try:
        X_train, X_test, y_train, y_test = trainer.load_and_preprocess_data(file_path)
        
        # Train models
        trainer.train_models(X_train, y_train)
        
        # Evaluate models
        results = trainer.evaluate_models(X_test, y_test)
        
        # Create plots
        trainer.create_plots()
        
        # Save models
        trainer.save_models()
        
        # Print best model
        best_name, best_model = trainer.get_best_model()
        print(f"\nBest performing model: {best_name}")
        print(f"R² Score: {results[best_name]['R²']:.4f}")
        
        # Print feature names for reference
        print(f"\nFeature names for prediction:")
        for i, feature in enumerate(trainer.feature_names):
            print(f"{i+1}. {feature}")
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Please update the file path.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()