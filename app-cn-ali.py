import streamlit as st
import pandas as pd
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from pycaret.regression import load_model, predict_model

# 调整模型路径到新项目位置
model = load_model('/root/insurance_predict/deployment_28042020_v2')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['prediction_label'][0]
    return predictions

def run():
    # 从URL获取logo和医院图片
    response = requests.get('https://s3.ap-east-1.amazonaws.com/employee-churn.optimops.ai/logo_v1.png')
    image = Image.open(BytesIO(response.content))
    st.image(image, use_column_width=False)

    response_hospital = requests.get('https://s3.ap-east-1.amazonaws.com/employee-churn.optimops.ai/hospital.jpg')
    image_hospital = Image.open(BytesIO(response_hospital.content))

    response_pycaret_logo = requests.get('https://s3.ap-east-1.amazonaws.com/employee-churn.optimops.ai/logo.png')
    image_pycaret_logo = Image.open(BytesIO(response_pycaret_logo.content))

    add_selectbox = st.sidebar.selectbox(
        "您想如何预测？",
        ("在线", "批量"))

    st.sidebar.info('此应用程序用于预测患者的医院费用')
    st.sidebar.success('https://optimops.ai')
    st.sidebar.image(image_hospital)
    st.sidebar.image(image_pycaret_logo)

    st.title("保险费用预测应用程序")

    if add_selectbox == '在线':
        # 表单元素
        age = st.number_input('年龄', min_value=1, max_value=100, value=25)
        sex = st.selectbox('性别', ['男性', '女性'])
        bmi = st.number_input('BMI', min_value=10, max_value=50, value=10)
        children = st.selectbox('孩子数量', [0,1,2,3,4,5,6,7,8,9,10])
        smoker = '是' if st.checkbox('吸烟者') else '否'
        region = st.selectbox('地区', ['西南', '西北', '东北', '东南'])

        input_dict = {'age': age, 'sex': sex, 'bmi': bmi, 'children': children, 'smoker': smoker, 'region': region}
        input_df = pd.DataFrame([input_dict])

        if st.button("预测"):
            output = predict(model=model, input_df=input_df)
            output = '¥ ' + str(round(output, 2))
            st.success(f'预测的费用是 {output}')

        # 显示输入数据表格
        st.write("用于预测的输入数据:")
        st.dataframe(input_df)

    if add_selectbox == '批量':
        file_upload = st.file_uploader("上传csv文件进行批量预测", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model, data=data)
            st.write(predictions)

if __name__ == '__main__':
    run()
