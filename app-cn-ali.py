from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
import requests
from PIL import Image
from io import BytesIO

# Adjusted the model path to the new project location
model = load_model('/root/insurance_predict/deployment_28042020_v2')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['prediction_label'][0]
    return predictions

def run():
    # Fetch the logo image from the URL
    response = requests.get('https://s3.ap-east-1.amazonaws.com/employee-churn.optimops.ai/logo_v1.png')
    image = Image.open(BytesIO(response.content))
    st.image(image, use_column_width=False)

    # Fetch the hospital image (ensure you replace the URL with the correct one)
    response_hospital = requests.get('https://s3.ap-east-1.amazonaws.com/employee-churn.optimops.ai/hospital.jpg')
    image_hospital = Image.open(BytesIO(response_hospital.content))

    add_selectbox = st.sidebar.selectbox(
        "您想如何预测？",  # "How would you like to predict?"
        ("在线", "批量"))  # "Online" and "Batch"

    st.sidebar.info('此应用程序旨在预测患者的医院费用')  # "This app is created to predict patient hospital charges"
    st.sidebar.success('https://optimops.ai')
    st.sidebar.image(image_hospital)

    st.title("保险费用预测应用程序")  # "Insurance Charges Prediction App"

    if add_selectbox == '在线':  # "Online"
        age = st.number_input('年龄', min_value=1, max_value=100, value=25)  # "Age"
        sex = st.selectbox('性别', ['男性', '女性'])  # "Sex" -> "male", "female"
        bmi = st.number_input('BMI', min_value=10, max_value=50, value=10)
        children = st.selectbox('孩子数量', [0,1,2,3,4,5,6,7,8,9,10])  # "Children"
        smoker = '是' if st.checkbox('吸烟者') else '否'  # "Smoker" -> yes/no
        region = st.selectbox('地区', ['西南', '西北', '东北', '东南'])  # "Region" -> regions in Chinese

        input_dict = {'age': age, 'sex': sex, 'bmi': bmi, 'children': children, 'smoker': smoker, 'region': region}
        input_df = pd.DataFrame([input_dict])

        if st.button("预测"):  # "Predict"
            output = predict(model=model, input_df=input_df)
            output = '¥ ' + str(round(output, 2))  # Currency symbol changed to yuan (¥)
            st.success(f'预测的费用是 {output}')  # "The predicted cost is"

        # Display input table
        st.write("用于预测的输入数据:")  # "Input data for prediction:"
        st.dataframe(input_df)

    if add_selectbox == '批量':  # "Batch"
        file_upload = st.file_uploader("上传用于预测的CSV文件", type=["csv"])  # "Upload csv file for predictions"

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model, data=data)
            st.write(predictions)

if __name__ == '__main__':
    run()
