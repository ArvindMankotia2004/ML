# streamlit_app.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# Set page configuration
st.set_page_config(
    page_title="Air Quality Prediction App",
    page_icon="🌍",
    layout="wide"
)

@st.cache_resource
def load_models():
    """Load saved models and scaler"""
    models = {}
    scaler = None
    feature_names = []
    results = {}
    
    save_dir = 'saved_models'
    
    try:
        # Load models
        model_files = {
            'Linear Regression': 'linear_regression_model.pkl',
            'Random Forest': 'random_forest_model.pkl',
            'Gradient Boosting': 'gradient_boosting_model.pkl'
        }
        
        for name, filename in model_files.items():
            filepath = os.path.join(save_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    models[name] = pickle.load(f)
        
        # Load scaler
        scaler_path = os.path.join(save_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        
        # Load feature names
        features_path = os.path.join(save_dir, 'feature_names.pkl')
        if os.path.exists(features_path):
            with open(features_path, 'rb') as f:
                feature_names = pickle.load(f)
        
        # Load results
        results_path = os.path.join(save_dir, 'model_results.pkl')
        if os.path.exists(results_path):
            with open(results_path, 'rb') as f:
                results = pickle.load(f)
        
        return models, scaler, feature_names, results
    
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {}, None, [], {}

def make_prediction(model, features, scaler):
    """Make prediction using selected model"""
    try:
        # Scale the features
        features_scaled = scaler.transform([features])
        # Make prediction
        prediction = model.predict(features_scaled)
        return prediction[0]
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

def main():
    """Main Streamlit app"""
    
    # Title and description
    st.title("🌍 Air Quality Prediction App")
    st.markdown("### Predict C6H6 (Benzene) concentration in air")
    st.markdown("This app uses machine learning models to predict benzene levels based on environmental factors.")
    
    # Load models
    models, scaler, feature_names, results = load_models()
    
    if not models or scaler is None or not feature_names:
        st.error("⚠️ Models not found! Please run the training script first.")
        st.stop()
    
    # Sidebar for model selection and performance
    st.sidebar.header("📊 Model Information")
    
    # Model selection
    selected_model = st.sidebar.selectbox(
        "Choose a model:",
        list(models.keys())
    )
    
    # Display model performance
    if results and selected_model in results:
        st.sidebar.subheader(f"📈 {selected_model} Performance")
        metrics = results[selected_model]
        st.sidebar.metric("R² Score", f"{metrics['R²']:.4f}")
        st.sidebar.metric("RMSE", f"{metrics['RMSE']:.4f}")
        st.sidebar.metric("MAE", f"{metrics['MAE']:.4f}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔧 Input Features")
        st.markdown("Enter the values for each environmental factor:")
        
        # Create input fields for each feature
        input_values = {}
        
        # Define typical ranges for each feature (you may need to adjust these)
        feature_ranges = {
            'PT08.S1(CO)': (647, 2040),
            'NMHC(GT)': (7, 1189),
            'PT08.S2(NMHC)': (383, 1479),
            'PT08.S3(NOx)': (322, 1365),
            'PT08.S4(NO2)': (551, 1672),
            'PT08.S5(O3)': (221, 1333),
            'T': (-1.9, 44.6),
            'RH': (9.2, 88.7),
            'AH': (0.1, 2.2),
            'hour': (0, 23),
            'day': (1, 31),
            'month': (1, 12),
            'dayofweek': (0, 6)
        }
        
        # Create two columns for input fields
        input_col1, input_col2 = st.columns(2)
        
        for i, feature in enumerate(feature_names):
            col = input_col1 if i % 2 == 0 else input_col2
            
            with col:
                if feature in feature_ranges:
                    min_val, max_val = feature_ranges[feature]
                    default_val = (min_val + max_val) / 2
                    
                    input_values[feature] = st.number_input(
                        f"{feature}",
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(default_val),
                        step=0.1 if feature in ['T', 'RH', 'AH'] else 1.0,
                        help=f"Range: {min_val} - {max_val}"
                    )
                else:
                    input_values[feature] = st.number_input(
                        f"{feature}",
                        value=0.0,
                        step=0.1
                    )
    
    with col2:
        st.header("🎯 Prediction")
        
        # Prediction button
        if st.button("🔮 Make Prediction", type="primary"):
            # Convert input values to list in correct order
            features = [input_values[feature] for feature in feature_names]
            
            # Make prediction
            prediction = make_prediction(models[selected_model], features, scaler)
            
            if prediction is not None:
                st.success(f"### Predicted C6H6 concentration:")
                st.metric("Benzene (C6H6)", f"{prediction:.2f} μg/m³")
                
                # Add interpretation
                if prediction < 5:
                    st.info("🟢 **Low** benzene levels")
                elif prediction < 15:
                    st.warning("🟡 **Moderate** benzene levels")
                else:
                    st.error("🔴 **High** benzene levels")
        
        # Help section
        st.markdown("---")
        st.subheader("ℹ️ Help")
        st.markdown("""
        **Feature Descriptions:**
        - **PT08.S1-S5**: Sensor responses
        - **T**: Temperature (°C)
        - **RH**: Relative Humidity (%)
        - **AH**: Absolute Humidity
        - **hour/day/month**: Time features
        - **dayofweek**: 0=Monday, 6=Sunday
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("*This app is for educational purposes. Actual air quality measurements should be done with calibrated instruments.*")

if __name__ == "__main__":
    main()