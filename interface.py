import streamlit as st
import joblib
import numpy as np

# Load the machine learning model
model = joblib.load("rf_class.pkl")

def main():
    st.title("Machine Learning Model Deployment")

    # Getting input components for features
    sepal_length = st.slider("Sepal Length", min_value=0.0, max_value=10.0, value=1.0)
    petal_width = st.slider("Petal Width", min_value=0.0, max_value=10.0, value=1.0)

    if st.button("Make Prediction"):
        result = make_prediction(sepal_length, petal_width)
        st.success(f"The Prediction is: {result}")

def make_prediction(sepal_length, petal_width):
    # Using the model to make predictions
    input_features = np.array([[sepal_length, petal_width]])
    prediction = model.predict(input_features)
    return prediction[0]

if __name__ == "__main__":
    main()
