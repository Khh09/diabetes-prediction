import streamlit as st
import joblib
import pandas as pd
from PIL import Image

image1= Image.open('undraw_medicine_b1ol.png')

@st.cache(allow_output_mutation=True)
def load(scaler_path, model_path):
    sc = joblib.load(scaler_path)
    model = joblib.load(model_path)
    return sc , model

def inference(row, scaler, model, feat_cols):
    df = pd.DataFrame([row], columns = feat_cols)
    X = scaler.transform(df)
    features = pd.DataFrame(X, columns = feat_cols)
    if (model.predict(features)==0):
        st.balloons()
        return "Congratulations: you are a healthy person!"

    else:
        return "**Warning**: You have high chances of being diabetics!"





st.title('**Diabetes Prediction**')
st.header('Kinan Hassounah')
st.header('Health care Analytics')
st.image(image1, caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
st.header('In this small app, I would like to explore a very important subject regarding diabetes diagnosis and prediction.')
st.write('This app will take into account the listed arguments in the navigation bar and predict whether a person is healthy or has of chance of being diabetic.')
st.header('Note that this app is powered by machine learning algorithm that are pipelined by Joblib .')
st.header('What is Diabetes?')
st.markdown('Diabetes is a chronic (long-lasting) health condition that affects how your body turns food into energy. Most of the food you eat is broken down into sugar (also called glucose) and released into your bloodstream. When your blood sugar goes up, it signals your pancreas to release insulin..')
st.write('The data for the following example is originally from the National Institute of Diabetes and Digestive and Kidney Diseases and contains information on females at least 21 years old of Pima Indian heritage.')

st.write('**Please fill in your health information below!**')

age =           st.number_input("Age", 1, 150, 25, 1)
pregnancies =   st.number_input("Number of Pregnancies", 0, 20, 0, 1)
glucose =       st.slider("Glucose Level", 0, 200, 25, 1)
skinthickness = st.slider("Skin Thickness", 0, 99, 20, 1)
bloodpressure = st.slider('Blood Pressure', 0, 122, 69, 1)
insulin =       st.slider("Insulin", 0, 846, 79, 1)
bmi =           st.number_input("BMI", 0.0, 67.1, 31.4, 0.1)
dpf =           st.slider("Diabetics Pedigree Function", 0.000, 2.420, 0.471, 0.001)

row = [pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age]

if (st.button('Predict Health Status')):
    feat_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    sc, model = load('models/scaler.joblib', 'models/model.joblib')
    result = inference(row, sc, model, feat_cols)
    st.write(result)
