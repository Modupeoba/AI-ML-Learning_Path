import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import joblib 

import streamlit as st
import pandas as pd

# Add a title with icon
st.title(":house_with_garden: USA Housing Insights: Unveiling the Secrets of Real Estate Trends")
st.markdown("<br>", unsafe_allow_html=True)

# Add an image
st.image('modern-residential-district-with-green-roof-balcony-generated-by-ai-removebg-preview.png')

# Add image to sidebar
st.sidebar.image('3999989__1_-removebg-preview.png', width = 300, caption = 'Welcome User')

# Add divider and spacing
st.sidebar.divider()
st.sidebar.markdown("<br>", unsafe_allow_html= True)

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
data = pd.read_csv('USA_Housing.csv')

# Drop the 'Address' column from the dataset
data = data.drop('Address', axis=1)

# Display the modified housing dataset in the Streamlit app
st.dataframe(data)

# Declare user Input variables with styled divider
st.sidebar.write(
    f"<h3 style='border-bottom:2px solid green; padding-bottom:10px;'>Input Variables</h3>", 
    unsafe_allow_html=True
)
avg_area_income = st.sidebar.number_input('Avg. Area Income', min_value=data['Avg. Area Income'].min(), max_value=data['Avg. Area Income'].max())
avg_area_house_age = st.sidebar.number_input('Avg. Area House Age', min_value=data['Avg. Area House Age'].min(), max_value=data['Avg. Area House Age'].max())
avg_area_rooms = st.sidebar.number_input('Avg. Area Number of Rooms', min_value=data['Avg. Area Number of Rooms'].min(), max_value=data['Avg. Area Number of Rooms'].max())
avg_area_bedrooms = st.sidebar.number_input('Avg. Area Number of Bedrooms', min_value=data['Avg. Area Number of Bedrooms'].min(), max_value=data['Avg. Area Number of Bedrooms'].max())
area_population = st.sidebar.number_input('Area Population', min_value=data['Area Population'].min(), max_value=data['Area Population'].max())

# Display the user's input
input_var = pd.DataFrame()
input_var['Avg. Area Income'] = [avg_area_income]
input_var['Avg. Area House Age'] = [avg_area_house_age]
input_var['Avg. Area Number of Rooms'] = [avg_area_rooms]
input_var['Avg. Area Number of Bedrooms'] = [avg_area_bedrooms]
input_var['Area Population'] = [area_population]

st.markdown("<br>", unsafe_allow_html= True)

# display the users input variable 
st.subheader('Users Input Variables')
st.dataframe(input_var)

# Add a button to trigger prediction
predict_button = st.sidebar.button('Predict')

# Load the scaler models
avg_area_income_scaler = joblib.load('Avg. Area Income_scaler.pkl')
area_population_scaler = joblib.load('Area Population_scaler.pkl')

# Transform the users input with the imported scalers
input_var['Avg. Area Income'] = avg_area_income_scaler.transform(input_var[['Avg. Area Income']])
input_var['Area Population'] = area_population_scaler.transform(input_var[['Area Population']])

# # Display the scaled input variables along with their original values
# st.subheader('Scaled Input Variables')
# st.write(input_var)

# Display the predicted value
model = joblib.load('HousepriceModel.pkl')
predicted = model.predict(input_var)

# Add spacing using Markdown
st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

# Add a subheader for the predicted value
st.markdown("<h2 style='color: #00541A;'>Predicted House Price</h2>", unsafe_allow_html=True)

# Display the predicted value
st.write(f"The predicted house price is: ${predicted[0]:,.2f}")


