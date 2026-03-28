import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("Copy of sonar data.csv", header=None)

X = data.drop(columns=60, axis=1)
Y = data[60]

# -----------------------------
# Train Model
# -----------------------------
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Rock vs Mine Prediction App")

st.write("Enter **60 sonar values separated by comma**")

input_data = st.text_area("Input Features")

if st.button("Predict"):

    try:
        input_list = [float(i) for i in input_data.split(",")]

        if len(input_list) != 60:
            st.error("Please enter exactly 60 values")
        else:
            input_array = np.asarray(input_list)
            input_reshaped = input_array.reshape(1, -1)

            prediction = model.predict(input_reshaped)

            if prediction[0] == "R":
                st.success("The object is a **ROCK**")
            else:
                st.success("The object is a **MINE**")

    except:
        st.error("Invalid input format")