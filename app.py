import streamlit as st
import pandas as pd
import joblib

# Load the trained model
loaded_lr_model = joblib.load('model.sav')

st.title('Sales Prediction App')
st.write('Enter the advertising spending to predict sales.')

# Input fields for features
tv = st.number_input('TV Advertising (in thousands)', min_value=0.0, max_value=300.0, value=100.0, step=0.1)
radio = st.number_input('Radio Advertising (in thousands)', min_value=0.0, max_value=50.0, value=20.0, step=0.1)
newspaper = st.number_input('Newspaper Advertising (in thousands)', min_value=0.0, max_value=120.0, value=10.0, step=0.1)

# Make prediction
if st.button('Predict Sales'):
    features = pd.DataFrame([[tv, radio, newspaper]], columns=['TV', 'Radio', 'Newspaper'])
    prediction = loaded_lr_model.predict(features)[0]
    st.success(f'Predicted Sales: {prediction:.2f}')
