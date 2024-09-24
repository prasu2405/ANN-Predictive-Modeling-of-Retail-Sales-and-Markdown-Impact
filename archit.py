import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import datetime as dt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LeakyReLU
from streamlit_option_menu import option_menu
from PIL import Image
import joblib 


# Streamlit Page Configuration
st.set_page_config(
    page_title="Retail Sales Prediction",
    page_icon='🛒',
    layout="wide"
)

# Background Configuration
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] > .main {
background: rgb(220, 196, 248);
background-size: cover;
background-position: center center;
background-repeat: no-repeat;
background-attachment: local;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    select = option_menu("Main Menu", ["About Project", "Sales Prediction"])

if select == "About Project":
    image1 = Image.open(r"C:\Guvi project\Final Project\ANN Predictive Modeling of Retail Sales and Markdown Impact\Forecasting images.jpeg")
    st.image(image1)
    st.header("📊 ANN Predictive Modeling of Retail Sales and Markdown Impact")
    st.write("---")  # Adds a horizontal line
    st.subheader("🛒 Domain: Retail Analytics")
    st.write(
        """
        Develop a predictive **Artificial Neural Network (ANN)** model to forecast sales for each store's department over the next year.
        """
    )
    st.write(
        """
        Analyze the impact of markdowns on sales, particularly during holiday weeks, and provide actionable insights for optimizing markdown strategies and inventory management.
        """
    )
    st.write("### Project Goals:")
    st.markdown(
        """
        - **Forecast Sales**: Predict department-wide sales for 45 retail stores over the next year, considering factors like seasonality, holidays, and markdowns.
        - **Markdown Analysis**: Understand the effect of markdowns on sales during holiday weeks versus non-holiday weeks and recommend optimal markdown strategies to maximize revenue.
        """
    )

elif select == "Sales Prediction":
    st.markdown("<h1 style='text-align: center; color: red;'>Retail Weekly Sales Prediction</h1>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(['PREDICTION', 'CONCLUSION'])

    with tab1:
        col3, col4, col5, col6, col10 = st.columns(5, gap="large")

        # Input Fields
        with col3:
            st.header(":violet[Store info]")
            store = st.selectbox('Select the **:red[Store Number]**', [i for i in range(1, 46)])
            dept = st.selectbox('Select the **:red[Department Number]**', [1, 2, 3, 4, 5, ... , 98])
            Type_ = st.selectbox('Select the **:red[Store Type]**', ['A', 'B', 'C'])
            Type_ = {'A': 0, 'B': 1, 'C': 2}[Type_]
            size = st.selectbox('Select the **:red[Store Size]**', [151315, 202307, ... , 118221])

        with col4:
            temperature = st.number_input('Enter the **:red[Temperature]** in Fahrenheit -----> **:green[(min=5.54 & max=100.14)]**', value=90.0, min_value=5.54, max_value=100.14)
            fuel_price = st.number_input('Enter the **:red[Fuel Price]** ---> **:green[(min=2.472 & max=4.468)]**', value=3.67, min_value=2.472, max_value=4.468)
            cpi = st.number_input('Enter the **:red[CPI]** ----------> **:green[(min=126.0 & max=227.47)]**', value=211.77, min_value=126.0, max_value=227.47)
            unemployment = st.number_input('Enter the **:red[Unemployment Rate]** in percentage **:green[(min=3.879 & max=14.313)]**', value=8.106, min_value=3.879, max_value=14.313)

        with col5:
            
            # Markdown Inputs
            markdown1 = st.number_input('Enter the **:red[Markdown1]** in dollars -------- **:green[(min=0.27,max=88646.76)]**', value=2000.00, min_value=0.27, max_value=88646.76)
            markdown2 = st.number_input('Enter the **:red[Markdown2]** in dollars -------- **:green[(min=0.02,max=104519.54)]**', value=65000.00, min_value=0.02, max_value=104519.54)
            markdown3 = st.number_input('Enter the **:red[Markdown3]** in dollars -------- **:green[(min=0.01,max=141630.61)]**', value=27000.00, min_value=0.01, max_value=141630.61)
            markdown4 = st.number_input('Enter the **:red[Markdown4]** in dollars -------- **:green[(min=0.22,max=67474.85)]**', value=11200.00, min_value=0.22, max_value=67474.85)
            markdown5 = st.number_input('Enter the **:red[Markdown5]** in dollars -------- **:green[(min=135.06,max=108519.28)]**', value=89000.00, min_value=135.06, max_value=108519.28)

        with col6:
            isholiday = st.selectbox('Select the **:red[IsHoliday]**', ["YES", "NO"]) 
            isholiday = 1 if isholiday == "YES" else 0

            year = st.selectbox('Select the **:red[Year]**', [i for i in range(2010, 2025)])
            month = st.selectbox('Select the **:red[Month]**', [i for i in range(1, 13)])
            week = st.selectbox('Select the **:red[Week]**', [i for i in range(1, 53)])

        with col10:
            st.header(":red[Weekly Sales]")

            # Input Data Preparation
            input_data = [[store, dept, isholiday, temperature, fuel_price, markdown1, markdown2, markdown3, markdown4, markdown5, cpi, unemployment, Type_, size, year, month, week]]
            input_data = np.array(input_data)

            # Load Scaler
            scaler = joblib.load('scaler.pkl')
            scaled_features = scaler.fit_transform(input_data)

            if st.button('Predict'):
                custom_objects = {'LeakyReLU': LeakyReLU}
                loaded_model4 = load_model('ann_retail.h5', custom_objects=custom_objects)
                prediction = loaded_model4.predict(scaled_features)
                st.write(f'Predicted Sales: {prediction[0][0]:,.2f}')

    with tab2:
        st.subheader("My observation from analysis and prediction of this data...") 
        st.write(" * The **:red[Weekly Sales]** of the retail store is dependent on many factors.")
        st.write(" * These factors are directly or indirectly affecting Weekly Sales.")
        st.write(" * **:red[Size of the store]** plays a major role.")
        st.write(" * Combination of **:violet[Fuel Price]** and **:green[Unemployment rate]** significantly impacts Weekly Sales.")
        st.write(" * **:red[Temperature]** and **:green[Markdown]** are sometimes in direct, sometimes in indirect relation, both significantly impacting Weekly Sales.")
