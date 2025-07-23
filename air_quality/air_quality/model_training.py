import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def train_models(X_train, y_train, scaler):
    """
    Train multiple regression models
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Gradient Boosting Regressor': GradientBoostingRegressor(
            n_estimators=100, 
            learning_rate=0.1, 
            random_state=42
        ),
        'Random Forest Regressor': RandomForestRegressor(
            n_estimators=100, 
            random_state=42
        )
    }
    
    # Train all models
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

def evaluate_models(models, X_test, y_test, scaler):
    """
    Evaluate trained models and return performance metrics
    """
    results = {}
    
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        }
    
    return results

def make_prediction(model, features, scaler):
    """
    Make a single prediction using a trained model
    """
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    return prediction[0]