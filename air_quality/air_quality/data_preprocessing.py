import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_and_preprocess_data(uploaded_file):
    """
    Load and preprocess the air quality data
    """
    # Read the CSV file
    df = pd.read_csv(uploaded_file, sep=";", decimal=",", header=0)
    
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
    
    X = df_features[features]
    y = df_features[target]
    
    # Split data chronologically (70% train, 30% test)
    train_size = int(0.7 * len(df_features))
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return df_features, X_train_scaled, X_test_scaled, y_train, y_test, scaler

def get_feature_names(df_features):
    """
    Get feature names excluding target pollutants
    """
    target_pollutants = ['CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']
    return [col for col in df_features.columns if col not in target_pollutants]