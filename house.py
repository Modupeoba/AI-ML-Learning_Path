import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import joblib 

# Add a title with icon
st.title(":house_with_garden: USA Housing Insights: Unveiling the Secrets of Real Estate Trends")
st.markdown("<br>", unsafe_allow_html=True)

# Add an image
st.image('modern-residential-district-with-green-roof-balcony-generated-by-ai-removebg-preview.png')


# Add image to sidebar
st.sidebar.image('3999989__1_-removebg-preview.png', width=300, caption='Welcome User')

# Add divider and spacing
# st.sidebar.divider()

# Add a header for project background information with styled divider
st.write(
    f"<h2 style='border-bottom:2px solid green; padding-bottom:10px;'>Project Background Information</h2>", 
    unsafe_allow_html=True
)

# Write project background information
st.write("The primary objective of this predictive model is to analyze USA housing data using machine learning algorithms. By leveraging demographic, socio-economic, and housing-related features, the model aims to predict housing prices accurately. This information can be valuable for various stakeholders, including home buyers, real estate agents, and policymakers.")

# Add spacing using Markdown
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Load the housing dataset
df = pd.read_csv('USA_Housing.csv')

# Drop the 'Address' column from the dataset
df = df.drop('Address', axis=1)

# Dataset Overview
st.write(
    f"<h2 style='border-bottom:2px solid green; padding-bottom:10px;'>Dataset Overview</h2>", 
    unsafe_allow_html=True
)

if st.checkbox('Show Raw Data'):
    st.write(df)

# Function to get user input
def get_user_input():
    input_choice = st.sidebar.radio('Select Your Input Type', ['Slider Input', 'Number Input'])
    
    if input_choice == 'Slider Input':
        area_income = st.sidebar.slider('Average Area Income', float(df['Avg. Area Income'].min()), float(df['Avg. Area Income'].max()))
        house_age = st.sidebar.slider('Average House Age', float(df['Avg. Area House Age'].min()), float(df['Avg. Area House Age'].max()))
        room_num = st.sidebar.slider('Average Number of Rooms', float(df['Avg. Area Number of Rooms'].min()), float(df['Avg. Area Number of Rooms'].max()))
        bedrooms = st.sidebar.slider('Average Number of Bedrooms', float(df['Avg. Area Number of Bedrooms'].min()), float(df['Avg. Area Number of Bedrooms'].max()))
        population = st.sidebar.slider('Area Population', float(df['Area Population'].min()), float(df['Area Population'].max()))
    else:
        area_income = st.sidebar.number_input('Average Area Income', float(df['Avg. Area Income'].min()), float(df['Avg. Area Income'].max()))
        house_age = st.sidebar.number_input('Average House Age', float(df['Avg. Area House Age'].min()), float(df['Avg. Area House Age'].max()))
        room_num = st.sidebar.number_input('Average Number of Rooms', float(df['Avg. Area Number of Rooms'].min()), float(df['Avg. Area Number of Rooms'].max()))
        bedrooms = st.sidebar.number_input('Average Number of Bedrooms', float(df['Avg. Area Number of Bedrooms'].min()), float(df['Avg. Area Number of Bedrooms'].max()))
        population = st.sidebar.number_input('Area Population', float(df['Area Population'].min()), float(df['Area Population'].max()))
    
    user_df = {
        'Avg. Area Income': area_income,
        'Avg. Area House Age': house_age,
        'Avg. Area Number of Rooms': room_num,
        'Avg. Area Number of Bedrooms': bedrooms,
        'Area Population': population
    }
    
    features = pd.DataFrame(user_df, index=[0])
    return features

# Get user input
user_input = get_user_input()

# Load the scalers and model
area_pop_scaler = joblib.load('Area Population_scaler.pkl')
area_inc_scaler = joblib.load('Avg. Area Income_scaler.pkl')
model = joblib.load('HousepriceModel.pkl')

# Transform user input
user_input['Area Population'] = area_pop_scaler.transform(user_input[['Area Population']])
user_input['Avg. Area Income'] = area_inc_scaler.transform(user_input[['Avg. Area Income']])

# Predict house price
predicted = model.predict(user_input)


# Tabs for prediction and interpretation
prediction, interpretation = st.tabs(["Prediction", "Interpretation"])

with prediction:
    pred = st.button('Predict')
    if pred:
        st.success(f'The Predicted price of your house is ${predicted[0]:,.2f} dollars')

with interpretation:
    st.header('Interpretation')
    st.write(f'The intercept of the model is: ${round(model.intercept_, 2):,.2f}')
    st.write(f'A unit change in the average area income causes the price to change by ${round(model.coef_[0], 2):,.2f}')
    st.write(f'A unit change in the average house age causes the price to change by ${round(model.coef_[1], 2):,.2f}')
    st.write(f'A unit change in the average rooms causes the price to change by ${round(model.coef_[2], 2):,.2f}')
    st.write(f'A unit change in the average bedrooms causes the price to change by ${round(model.coef_[3], 2):,.2f}')
    st.write(f'A unit change in the area population causes the price to change by ${round(model.coef_[4], 2):,.2f}')


# User Guide and Help Section
st.header('User Guide & Help')

if st.checkbox('Show User Guide'):
    st.subheader('User Guide')
    st.write("""
    - Use the sliders or number inputs to provide average area income, house age, number of rooms, bedrooms, and population.
    - Click 'Predict' to see the predicted house price.
    - The 'Interpretation' section explains how each feature affects the predicted price.
    - Check 'Show Raw Data'to explore the dataset.
    """)

    st.subheader('Need Help?')
    st.write("""
    - If you encounter any issues or have questions, please contact our support team at modupeobamuyi@gmail.com
    """)
