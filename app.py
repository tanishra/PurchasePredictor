import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler from pickle files
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit UI
st.title("User Purchase Prediction")

st.write("This app predicts whether a user will purchase a product based on two features: Age and Estimated Salary.")

# Input fields for the user to provide values
age = st.number_input("Age", min_value=18, max_value=100, value=25)
estimated_salary = st.number_input("Estimated Salary", min_value=0, max_value=200000, value=50000)

# Prepare the input array for prediction with Age and Estimated Salary
input_array = np.array([age, estimated_salary]).reshape(1, -1)

# Button to trigger prediction
if st.button('Predict Purchase'):
    # Feature scaling (using the saved scaler)
    input_scaled = scaler.transform(input_array)

    # Predict the purchase behavior using the trained SVM model
    prediction = model.predict(input_scaled)

    # Display the result
    if prediction == 1:
        st.success("The user is likely to make a purchase!")
    else:
        st.warning("The user is unlikely to make a purchase.")
