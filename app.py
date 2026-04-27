import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


model = tf.keras.models.load_model('model.h5')

#Load the label encoder, one-hot encoder, and scaler
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('onehot_encoder_geography.pkl', 'rb') as f:
    onehot_encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


# Define the Streamlit app
st.title('Customer Churn Prediction')

# Get user input
geography = st.selectbox('Geography', onehot_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age', 18, 100)
balance = st.slider('Balance', 0, 250000, step=1000)
credit_score = st.slider('Credit Score', min_value=300, max_value=850)
estimated_salary = st.slider('Estimated Salary', min_value=0, max_value=200000, step=1000)
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = {
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_of_products,
    "HasCrCard": has_cr_card,
    "IsActiveMember": is_active_member,
    "EstimatedSalary": estimated_salary
}

geo_encoded = onehot_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder.get_feature_names_out(['Geography']))

input_df = pd.DataFrame([input_data])
input_df = pd.concat([input_df.drop('Geography', axis=1), geo_encoded_df], axis=1)

input_df['Gender'] = label_encoder.transform(input_df['Gender'])
input_scaled = scaler.transform(input_df)

prediction = model.predict(input_scaled)
prediction_probability = prediction[0][0]
st.write(f'Churn Probability: {prediction_probability:.2f}')

if prediction_probability > 0.5:
    st.write('The customer is likely to churn.')
else:    
    st.write('The customer is not likely to churn.')
