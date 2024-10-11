import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle

model = tf.keras.models.load_model(r'C:\Users\mwael\OneDrive\Desktop\after_cource\Uneeq_intern\Bank_Customer_Churn\Bank_customer_churn.h5')

with open(r'C:\Users\mwael\OneDrive\Desktop\after_cource\Uneeq_intern\Bank_Customer_Churn\scaler.pkl', 'rb') as f:
    scaler = pickle.load(f) 

st.title("نظام التنبؤ بانسحاب العملاء")

CreditScore = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
Geography = st.selectbox("Geography", options=['France', 'Germany', 'Spain'])
Gender = st.selectbox("Gender", options=['Female', 'Male'])
Age = st.number_input("Age", min_value=18, max_value=100, value=40)
Balance = st.number_input("Balance", min_value=0.0, value=50000.0)
NumOfProducts = st.selectbox("Number of Products", options=[1, 2, 3, 4])
IsActiveMember = st.selectbox("Is Active Member", options=['Yes', 'No'])

geography_map = {'France': 0, 'Germany': 1, 'Spain': 2}
gender_map = {'Female': 0, 'Male': 1}
is_active_member_map = {'Yes': 1, 'No': 0}

geography_encoded = geography_map[Geography]
gender_encoded = gender_map[Gender]
is_active_member_encoded = is_active_member_map[IsActiveMember]

input_data = np.array([[CreditScore, geography_encoded, gender_encoded, Age, Balance, NumOfProducts, is_active_member_encoded]])

if st.button('Prediction'):
    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)
    print(prediction)
    if prediction[0] > 0.5:
        st.write("العميل قد يخرج من البنك.")
    else:
        st.write("العميل سيبقى في البنك.")
