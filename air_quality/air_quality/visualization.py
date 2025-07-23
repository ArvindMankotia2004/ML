import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def create_plots(results_df):
    """
    Create visualization plots for model comparison
    """
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
    return fig

def plot_predictions_vs_actual(y_true, y_pred, model_name):
    """
    Create scatter plot of predictions vs actual values
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(y_true, y_pred, alpha=0.5)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'{model_name}: Predictions vs Actual Values')
    
    return fig

def plot_residuals(y_true, y_pred, model_name):
    """
    Create residual plot
    """
    residuals = y_true - y_pred
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    ax.set_title(f'{model_name}: Residual Plot')
    
    return fig