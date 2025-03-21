import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from PIL import Image
from io import BytesIO
from pycaret.regression import load_model, predict_model

# Detect OS and set model path accordingly
if os.name == "nt":  # Windows
    model_path = r"D:\MLOPS_PROJECTS\insurance-end-2-end\deployment_28042020_v2"
else:  # Linux/macOS
    model_path = "/root/insurance_predict/deployment_28042020_v2"

# Try to load the model and handle errors
try:
    model = load_model(model_path)
    st.success(f"Model loaded successfully from: {model_path}")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    model = None  # Ensure model is defined, even if it fails

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['prediction_label'][0]
    return predictions

def run():
    # Fetch the logo and hospital images from the URL
    response = requests.get('https://s3.ap-east-1.amazonaws.com/employee-churn.optimops.ai/logo_v1.png')
    image = Image.open(BytesIO(response.content))
    st.image(image, use_column_width=False)

    response_hospital = requests.get('https://s3.ap-east-1.amazonaws.com/employee-churn.optimops.ai/hospital.jpg')
    image_hospital = Image.open(BytesIO(response_hospital.content))

    response_pycaret_logo = requests.get('https://s3.ap-east-1.amazonaws.com/employee-churn.optimops.ai/logo.png')
    image_pycaret_logo = Image.open(BytesIO(response_pycaret_logo.content))

    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?",
        ("Online", "Batch"))

    st.sidebar.info('This app is created to predict patient hospital charges')
    st.sidebar.success('https://optimops.ai')
    st.sidebar.image(image_hospital)
    st.sidebar.image(image_pycaret_logo)

    st.title("Insurance Charges Prediction App")

    if add_selectbox == 'Online':
        # Your form elements
        age = st.number_input('Age', min_value=1, max_value=100, value=25)
        sex = st.selectbox('Sex', ['male', 'female'])
        bmi = st.number_input('BMI', min_value=10, max_value=50, value=10)
        children = st.selectbox('Children', [0,1,2,3,4,5,6,7,8,9,10])
        smoker = 'yes' if st.checkbox('Smoker') else 'no'
        region = st.selectbox('Region', ['southwest', 'northwest', 'northeast', 'southeast'])

        input_dict = {'age': age, 'sex': sex, 'bmi': bmi, 'children': children, 'smoker': smoker, 'region': region}
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = '$ ' + str(round(output, 2))
            st.success(f'The predicted cost is {output}')

        # Display input table
        st.write("Input data for prediction:")
        st.dataframe(input_df)

    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model, data=data)
            st.write(predictions)

if __name__ == '__main__':
    run()
