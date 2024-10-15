from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Load the model using the correct file path
model = load_model('D:\\MLOPS_PROJECTS\\insurance-end-2-end\\deployment_28042020_v2')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions_df['Predicted_cost'] = predictions_df['prediction_label'].round(2)  # Rename and round to 2 decimal places
    return predictions_df

def run():
    # Fetch the logo from the URL using requests
    logo_url = 'https://s3.ap-east-1.amazonaws.com/employee-churn.optimops.ai/logo_v1.png'
    response = requests.get(logo_url)
    image = Image.open(BytesIO(response.content))  # Open image from the fetched content
    
    image_hospital = Image.open('hospital.jpg')

    st.image(image, use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?",
        ("Online", "Batch"))

    # Update the sidebar URL to https://optimops.ai
    st.sidebar.info('This app is created to predict patient hospital charges')
    st.sidebar.success('https://optimops.ai')
    st.sidebar.image(image_hospital)

    st.title("Insurance Charges Prediction App")

    if add_selectbox == 'Online':
        age = st.number_input('Age', min_value=1, max_value=100, value=25)
        sex = st.selectbox('Sex', ['male', 'female'])
        bmi = st.number_input('BMI', min_value=10, max_value=50, value=10)
        children = st.selectbox('Children', [0,1,2,3,4,5,6,7,8,9,10])
        smoker = 'yes' if st.checkbox('Smoker') else 'no'
        region = st.selectbox('Region', ['southwest', 'northwest', 'northeast', 'southeast'])

        input_dict = {'age': age, 'sex': sex, 'bmi': bmi, 'children': children, 'smoker': smoker, 'region': region}
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            predictions_df = predict(model=model, input_df=input_df)
            st.success(f"The predicted charge is $ {predictions_df['Predicted_cost'][0]}")
            
            # Display the input features and prediction
            st.write(predictions_df[['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'Predicted_cost']])

    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model, data=data)
            st.write(predictions)

if __name__ == '__main__':
    run()
