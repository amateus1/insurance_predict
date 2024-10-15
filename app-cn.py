from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
import requests  # Import requests for fetching images from URLs
from PIL import Image

# 加载模型
model = load_model('D:\\MLOPS_PROJECTS\\insurance-end-2-end\\deployment_28042020_v2')

# 定义列的中英文映射
column_mapping = {
    '年龄': 'age',
    '性别': 'sex',
    'BMI': 'bmi',
    '子女数量': 'children',
    '吸烟者': 'smoker',
    '地区': 'region'
}

def predict(model, input_df):
    # 将中文列名转换为模型所需的英文列名
    input_df = input_df.rename(columns=column_mapping)
    predictions_df = predict_model(estimator=model, data=input_df)
    # 获取预测结果，并保留两位小数
    predictions = round(predictions_df['prediction_label'][0], 2)
    return predictions

def run():
    # 使用 S3 URL 加载 logo
    image = Image.open(requests.get('https://s3.ap-east-1.amazonaws.com/employee-churn.optimops.ai/logo_v1.png', stream=True).raw)
    image_hospital = Image.open('hospital.jpg')

    st.image(image, use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
        "你想要如何进行预测？",
        ("在线", "批量"))

    st.sidebar.info('此应用程序用于预测患者的医院费用')
    # 修改 URL 到 optimops.ai
    st.sidebar.success('https://optimops.ai')
    st.sidebar.image(image_hospital)

    st.title("保险费用预测应用程序")

    if add_selectbox == '在线':
        age = st.number_input('年龄', min_value=1, max_value=100, value=25)
        sex = st.selectbox('性别', ['男', '女'])
        bmi = st.number_input('BMI', min_value=10, max_value=50, value=10)
        children = st.selectbox('子女数量', [0,1,2,3,4,5,6,7,8,9,10])
        smoker = '是' if st.checkbox('吸烟者') else '否'
        region = st.selectbox('地区', ['西南', '西北', '东北', '东南'])

        # 构建输入数据字典
        input_dict = {
            '年龄': age,
            '性别': sex,
            'BMI': bmi,
            '子女数量': children,
            '吸烟者': smoker,
            '地区': region
        }
        input_df = pd.DataFrame([input_dict])

        if st.button("预测"):
            # 获取预测结果
            output = predict(model=model, input_df=input_df)
            st.success(f'预测费用为 ¥ {output}')

            # 显示用于预测的输入表
            st.write("用于预测的输入表：")
            st.table(input_df)

    if add_selectbox == '批量':
        file_upload = st.file_uploader("上传 CSV 文件进行批量预测", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            # 将中文列名转换为英文列名
            data = data.rename(columns=column_mapping)
            predictions = predict_model(estimator=model, data=data)
            st.write(predictions)

if __name__ == '__main__':
    run()
