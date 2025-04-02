import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
with open('heart_disease_model.pkl', 'rb') as file:
    model, scaler = pickle.load(file)

# Define min and max values for each parameter
parameter_ranges = {
    "age": (20, 100),
    "sex": (0, 1),
    "cp": (0, 3),
    "trestbps": (80, 200),
    "chol": (100, 600),
    "fbs": (0, 1),
    "restecg": (0, 2),
    "thalach": (60, 220),
    "exang": (0, 1),
    "oldpeak": (0.0, 6.2),
    "thal": (1, 3)
}

# Streamlit UI
st.title("Heart Disease Prediction App")
st.write("Enter your details to predict heart disease risk")

# User Input
user_data = []
for param, (min_val, max_val) in parameter_ranges.items():
    if isinstance(min_val, int):  # For integers and categorical values
        value = st.slider(param.capitalize(), min_val, max_val, min_val)
    else:  # For float values
        value = st.slider(param.capitalize(), min_val, max_val, min_val, step=0.1)
    user_data.append(value)

# Predict Button
if st.button("Predict"):
    input_data = np.array(user_data).reshape(1, -1)
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)[0]
    result = "High Risk of Heart Disease" if prediction == 1 else "Low Risk of Heart Disease"
    
    # Display Prediction
    st.markdown(f"## Result: :{'red' if prediction == 1 else 'green'}[{result}]")
    
    # Healthy person's sample data for comparison
    healthy_person = np.array([50, 0, 1, 120, 200, 0, 1, 150, 0, 1.0, 2]).reshape(1, -1)
    healthy_person = scaler.transform(healthy_person)[0]

    # Plot Comparison
    labels = list(parameter_ranges.keys())
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, np.abs(input_data[0]), width, label='User')
    ax.bar(x + width/2, np.abs(healthy_person), width, label='Healthy')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()
    st.pyplot(fig)
