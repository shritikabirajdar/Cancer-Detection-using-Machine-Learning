import streamlit as st
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

st.title("Cancer Detection App")

data = load_breast_cancer()
model = LogisticRegression(max_iter=1000)
model.fit(data.data, data.target)

inputs = []
for i in range(5):
    val = st.number_input(f"Feature {i}")
    inputs.append(val)

if st.button("Predict"):
    prediction = model.predict([inputs + [0]*(data.data.shape[1]-5)])
    st.write("Prediction:", "Malignant" if prediction[0]==0 else "Benign")
