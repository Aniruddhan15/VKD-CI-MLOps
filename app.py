# app.py
import streamlit as st
import joblib
import numpy as np

st.title("Iris Flower Classifier 🌸")
st.write("Enter the values to predict the Iris species:")

sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width", 0.1, 2.5, 0.2)

model = joblib.load("iris_model.pkl")
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)

iris_classes = ["Setosa", "Versicolor", "Virginica"]
# adding submit button
if st.button("Predict"):
    st.write("Processing...")
    st.success(f"Predicted Class: {iris_classes[prediction[0]]}")
