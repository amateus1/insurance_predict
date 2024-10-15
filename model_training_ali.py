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