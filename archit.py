import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from streamlit_option_menu import option_menu
from PIL import Image
from sklearn.preprocessing import StandardScaler


st.set_page_config(layout= "wide")
st.title("Predictive Modeling of Retail Sales")
st.write("")


model = tf.keras.models.load_model(r'C:\Guvi project\Final Project\ANN Predictive Modeling of Retail Sales and Markdown Impact\Final_retail_sales.h5')


scaler = joblib.load('scaler.pkl')

def predict_sales(inputs):
    inputs_scaled = scaler.transform(np.array(inputs).reshape(1, -1))
    prediction = model.predict(inputs_scaled)
    return prediction[0][0]

with st.sidebar:
    select= option_menu("Main Menu", ["About Project", "Sales Prediction"])

if select == "About Project":

    image1= Image.open(r"C:\Guvi project\Final Project\ANN Predictive Modeling of Retail Sales and Markdown Impact\Forecasting images.jpeg")
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


if select == "Sales Prediction":
    st.header("📈 Retail Price Analysis")
    st.write("---")  # Adds a horizontal line for separation

    # Creating tabs for the sales prediction section
    tab1 = st.tabs(["🛒 PREDICT WEEKLY SALES"])

    with st.form('my_form'):
        st.subheader("Enter the following details to predict weekly sales:")

        # Using columns to organize input fields side by side
        col1, col2 = st.columns([5, 5])
        
        with col1:
            st.markdown("### Store Information")
            store_id = st.number_input('🏬 Store ID (min:1 & max:45)', min_value=1, max_value=45)
            dept_id = st.number_input('🗂 Dept ID (min:1 & max:99)', min_value=1, max_value=99)
            holiday_flag = st.selectbox("🎉 Is it a Holiday Week?", ["No", "Yes"])
            temperature = st.number_input("🌡 Temperature (F) (min:-2.4 & max:100.1)", min_value=-2.4, max_value=100.1)
            fuel_price = st.number_input("⛽ Fuel Price (min:2.472 & max:4.468)", min_value=2.472, max_value=4.468)
            Type = st.number_input("🏷 Store Type (min:1 & max:3)", min_value=1, max_value=3)
            year = st.number_input("📅 Year (min:2010 & max:2012)", min_value=2010, max_value=2012)
            month = st.number_input("📆 Month (min:1 & max:12)", min_value=1, max_value=12)
            week_of_month = st.number_input("🗓 Week of Month (min:0 & max:4)", min_value=0, max_value=4)
            day_of_week = st.selectbox("🗓 Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

        with col2:
            st.markdown("### Markdown and Economic Factors")
            markdown1 = st.number_input("📉 Markdown 1 Amount (min:0 & max:100000)", min_value=0, max_value=100000)
            markdown2 = st.number_input("📉 Markdown 2 Amount (min:0 & max:100000)", min_value=0, max_value=100000)
            markdown3 = st.number_input("📉 Markdown 3 Amount (min:0 & max:100000)", min_value=0, max_value=100000)
            markdown4 = st.number_input("📉 Markdown 4 Amount (min:0 & max:100000)", min_value=0, max_value=100000)
            markdown5 = st.number_input("📉 Markdown 5 Amount (min:0 & max:100000)", min_value=0, max_value=100000)
            cpi = st.number_input("📊 Consumer Price Index (CPI) (min:126.0 & max:227.0)", min_value=126.0, max_value=227.0)
            unemployment = st.number_input("💼 Unemployment Rate (min:3.8 & max:14.33)", min_value=3.8, max_value=14.33)
            store_size = st.number_input('📏 Store Size (sqft) (min:34875 & max:219622)', min_value=34875, max_value=219622)
        
            # Submit button to trigger prediction
            submit_button = st.form_submit_button(label="🚀 Predict Sales")

    if submit_button:
        # Convert categorical inputs to numerical format if necessary
        holiday_flag = 1 if holiday_flag == "Yes" else 0
        day_of_week_num = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_of_week)

        # Collect all inputs into a list
        inputs = [
            store_id, dept_id, store_size, Type, holiday_flag, markdown1, markdown2,
            markdown3, markdown4, markdown5, temperature, fuel_price, cpi,
            unemployment, day_of_week_num, year, month, week_of_month
        ]

        # Predict sales
        prediction = predict_sales(inputs)
        st.success(f"📊 Predicted Sales: **${prediction:.2f}**")
