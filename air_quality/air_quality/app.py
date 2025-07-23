import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import load_and_preprocess_data
from model_training import train_models, evaluate_models
from visualization import create_plots

# Set page config
st.set_page_config(
    page_title="Air Quality Prediction App",
    page_icon="🌬️",
    layout="wide"
)

# Title and description
st.title("🌬️ Air Quality Prediction Dashboard")
st.markdown("This app predicts C6H6 (Benzene) concentrations using machine learning models.")

# Sidebar for file upload
st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV file", 
    type=['csv'],
    help="Upload an air quality dataset in CSV format"
)

# Main content
if uploaded_file is not None:
    try:
        # Load and preprocess data
        with st.spinner("Loading and preprocessing data..."):
            df_processed, X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(uploaded_file)
        
        st.success("Data loaded and preprocessed successfully!")
        
        # Display data info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df_processed))
        with col2:
            st.metric("Training Size", len(X_train))
        with col3:
            st.metric("Testing Size", len(X_test))
        
        # Data preview
        st.subheader("📊 Data Preview")
        st.dataframe(df_processed.head())
        
        # Model training section
        st.subheader("🤖 Model Training & Evaluation")
        
        if st.button("Train Models", type="primary"):
            with st.spinner("Training models..."):
                models = train_models(X_train, y_train, scaler)
                results = evaluate_models(models, X_test, y_test, scaler)
            
            # Display results
            st.subheader("📈 Model Performance")
            
            # Create results dataframe
            results_df = pd.DataFrame(results).T
            st.dataframe(results_df.round(4))
            
            # Best model
            best_model = results_df['R²'].idxmax()
            st.success(f"🏆 Best Model: {best_model} (R² = {results_df.loc[best_model, 'R²']:.4f})")
            
            # Visualizations
            st.subheader("📊 Visualizations")
            
            # Model comparison chart
            fig_comparison = create_plots(results_df)
            st.pyplot(fig_comparison)
            
            # Feature importance (for Random Forest)
            if 'Random Forest Regressor' in models:
                st.subheader("🔍 Feature Importance (Random Forest)")
                rf_model = models['Random Forest Regressor']
                feature_names = [col for col in df_processed.columns if col not in ['CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']]
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': rf_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=importance_df.head(10), x='Importance', y='Feature', ax=ax)
                ax.set_title('Top 10 Feature Importance')
                st.pyplot(fig)
    
    except Exception as e:
        st.error(f"Error processing the data: {str(e)}")
        st.info("Please make sure your CSV file has the correct format and columns.")

else:
    st.info("👆 Please upload a CSV file to get started.")
    
    # Sample data format
    st.subheader("📋 Expected Data Format")
    st.markdown("""
    Your CSV file should contain the following columns:
    - Date and Time columns (or DateTime)
    - Air quality measurements including C6H6(GT)
    - Other pollutant measurements like CO(GT), NOx(GT), NO2(GT)
    - Environmental factors
    
    The app will automatically preprocess the data and create time-based features.
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")