
from pycaret.datasets import get_data
from pycaret.regression import setup, create_model, save_model

print("Starting the model training script...")

# Load the dataset
try:
    data = get_data('insurance')
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error loading dataset: {e}")

# Initialize PyCaret regression setup
try:
    print("Setting up the regression model...")
    regression_setup = setup(data, target='charges')
    print("Model setup completed!")
except Exception as e:
    print(f"Error during setup: {e}")

# Create a regression model (example: Linear Regression)
try:
    print("Creating the regression model...")
    model = create_model('lr')  # You can replace 'lr' with other models like 'rf' for RandomForest, etc.
    print("Model created successfully!")
except Exception as e:
    print(f"Error creating model: {e}")

# Save the trained model to the specified path
try:
    print("Saving the trained model...")
    save_model(model, '/root/insurance_predict/deployment_28042020_v2')
    print("Model saved successfully!")
except Exception as e:
    print(f"Error saving model: {e}")

import mlflow
import os

# Setting up the MLflow tracking URI for DagsHub
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# Start the MLflow run
with mlflow.start_run():
    # Assuming there's model training here
    # Log your parameters, metrics, and model
    mlflow.log_param("training_method", "xgboost")
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log the model (assuming the model variable is called 'trained_model')
    mlflow.sklearn.log_model(trained_model, "model")
    
    print("Logged training information to MLflow.")

    # Evidently Integration
    from evidently.dashboard import Dashboard
    from evidently.dashboard.tabs import DataDriftTab, RegressionPerformanceTab
    
    # Assuming `train_data` and `test_data` are defined in your training script
    dashboard = Dashboard(tabs=[DataDriftTab(), RegressionPerformanceTab()])
    dashboard.calculate(train_data, test_data)
    
    # Save the report locally
    dashboard.save("evidently_report.html")
    
    # Optional: Upload this to S3 or a similar service for monitoring
    print("Generated and saved Evidently report.")
