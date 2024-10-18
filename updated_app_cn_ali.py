
from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import requests
from PIL import Image
from io import BytesIO

# Adjusted the model path to the new project location
model = load_model('/root/insurance_predict/deployment_28042020_v2')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['prediction_label'][0]
    # Ensure prediction is not negative
    if predictions < 0:
        predictions = 0
    return predictions

def run():
    # Fetch the logo image from the URL
    response = requests.get('https://s3.ap-east-1.amazonaws.com/employee-churn.optimops.ai/logo_v1.png')
    image = Image.open(BytesIO(response.content))
    st.image(image, use_column_width=False)

    # Fetch the hospital image
    response_hospital = requests.get('https://s3.ap-east-1.amazonaws.com/employee-churn.optimops.ai/hospital.jpg')
    image_hospital = Image.open(BytesIO(response_hospital.content))

    # Display hospital image and PyCaret logo
    st.sidebar.image(image_hospital)
    
    response_pycaret = requests.get('https://s3.ap-east-1.amazonaws.com/employee-churn.optimops.ai/logo.png')
    image_pycaret = Image.open(BytesIO(response_pycaret.content))
    st.sidebar.image(image_pycaret)

    add_selectbox = st.sidebar.selectbox(
        "您想如何预测?",
        ("在线", "批量"))

    st.sidebar.info('此应用程序旨在预测患者的医院费用')
    st.sidebar.success('https://optimops.ai')

    st.title("保险费用预测应用程序")

    if add_selectbox == '在线':
        age = st.number_input('年龄', min_value=1, max_value=100, value=25)
        sex = st.selectbox('性别', ['男性', '女性'])
        bmi = st.number_input('BMI', min_value=10, max_value=50, value=10)
        children = st.selectbox('孩子数量', [0,1,2,3,4,5,6,7,8,9,10])
        smoker = 'yes' if st.checkbox('吸烟者') else 'no'
        region = st.selectbox('地区', ['西南', '西北', '东北', '东南'])

        # Map Chinese input to expected model input
        sex_mapping = {'男性': 'male', '女性': 'female'}
        smoker_mapping = {'是': 'yes', '否': 'no'}
        region_mapping = {'西南': 'southwest', '西北': 'northwest', '东北': 'northeast', '东南': 'southeast'}

        input_dict = {
            'age': age, 
            'sex': sex_mapping.get(sex), 
            'bmi': bmi, 
            'children': children, 
            'smoker': smoker, 
            'region': region_mapping.get(region)
        }

        input_df = pd.DataFrame([input_dict])

        if st.button("预测"):
            output = predict(model=model, input_df=input_df)
            output = '¥ ' + str(round(output, 2))
            st.success(f'预测的费用是 {output}')

        # Display input table
        st.write("用于预测的输入数据:")
        st.dataframe(input_df)

    if add_selectbox == '批量':
        file_upload = st.file_uploader("上传 csv 文件进行批量预测", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model, data=data)
            st.write(predictions)

if __name__ == '__main__':
    run()


import mlflow
import os

# Adding MLflow logging to the app logic
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

with mlflow.start_run():
    # Assuming app logic generates predictions
    prediction = model.predict(input_data)

    # Log predictions to MLflow
    mlflow.log_metric("prediction_result", prediction)

    print("Logged prediction result to MLflow.")

    # Evidently dashboard generation, if applicable in this app stage
    from evidently.dashboard import Dashboard
    from evidently.dashboard.tabs import DataDriftTab, RegressionPerformanceTab
    
    dashboard = Dashboard(tabs=[DataDriftTab(), RegressionPerformanceTab()])
    dashboard.calculate(train_data, input_data)
    
    dashboard.save("evidently_report.html")
    print("Evidently report generated.")
