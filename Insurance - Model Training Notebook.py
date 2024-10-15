#!/usr/bin/env python
# coding: utf-8

# In[1]:

# Uncomment the lines below if you haven't installed pycaret or requests
# !pip install pycaret==3.0
# !pip install requests

# In[2]:

import pycaret
print(pycaret.__version__)

# In[3]:

from pycaret.datasets import get_data
data = get_data('insurance')

# # Experiment 2

# In[4]:

from pycaret.regression import setup, create_model, save_model, load_model, plot_model

# Setup the PyCaret environment
s2 = setup(data, target='charges',
           normalize=True,
           polynomial_features=True,
           #silent=True,
           bin_numeric_features=['age', 'bmi'])

# In[5]:

# Create a Linear Regression model
lr = create_model('lr')

# In[6]:

# Plot the model (optional)
plot_model(lr)

# In[7]:

# Save the trained model
save_model(lr, 'D:\\MLOPS_PROJECTS\\insurance-models\\deployment_28042020_v2.pkl')

# In[8]:

# Load the saved model for deployment or testing
deployment_28042020 = load_model('D:\\MLOPS_PROJECTS\\insurance-models\\deployment_28042020_v2.pkl')

# Print model information
print(deployment_28042020)

# ### Execute the Application

# In[9]:

# Run Streamlit application (use this in a terminal, not inside the notebook)
# !streamlit run app.py

# #### Example: Making a POST request with the deployed model (optional)

# In[10]:

import requests
url = 'https://pycaret-insurance.herokuapp.com/predict_api'
pred = requests.post(url, json={'age': 55, 'sex': 'male', 'bmi': 59, 'children': 1, 'smoker': 'male', 'region': 'northwest'})
print(pred.json())

