import streamlit as st
import tensorflow as tf
import joblib
import numpy as np

# Load Model
model = tf.saved_model.load('saved_model')
scaler = joblib.load("scaler_iris.pkl")


def make_prediction(input):
    # Scaling
    new_input_scaled = scaler.transform(input)

    # Ambil signature serving
    infer = model.signatures["serving_default"]

    # Konversi ke tensor
    input_tensor = tf.convert_to_tensor(new_input_scaled, dtype=tf.float32)

    # Predict
    prediction = infer(input_tensor)

    # Ambil output (biasanya hanya 1 key)
    prediction_values = list(prediction.values())[0].numpy()

    predicted_class = np.argmax(prediction_values, axis=1)

    class_names = ['setosa', 'versicolor', 'virginica']

    return class_names[predicted_class[0]]



# Untuk membuat judul halaman web dan juga iconnya
st.set_page_config(
    page_title="Iris Classification",
    page_icon="🌸",
    layout="centered"
)

st.badge("DEMO")
st.header("Iris Classification")
st.write("Sistem Klasifikasi Bunga Iris dengan Neural Network")

with st.form('iris_feature_input'):
    sepal_length = st.number_input("Masukkan nilai sepal length")
    sepal_width = st.number_input("Masukkan nilai sepal width")
    petal_length = st.number_input("Masukkan nilai petal length")
    petal_width = st.number_input("Masukkan nilai petal width")

    submit = st.form_submit_button('Submit')


if submit:
    data = [[sepal_length, sepal_width, petal_length, petal_width]]
    result = make_prediction(data)
    st.markdown(f'Hasil Klasifikasi: **{result}**')
