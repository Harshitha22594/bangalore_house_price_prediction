# streamlit 
import pickle 
import pandas as pd
import streamlit as st
# model de serialization (loading model)
with open("Linear_model.pkl","rb") as file:
    model = pickle.load( file)

# model.predict(data)

# import joblib
# # model de serialization (loading model) using joblib
# file = 'model.pkl'
# Model = pickle.load(file)

with open("label_encoder.pkl","rb") as file1:
    encoder = pickle.load(file1)

df = pd.read_csv("cleaned_data.csv")
st.set_page_config(
    page_title= "Banglore House price Predictin "
)

st.title("Banglore House price Predictin ")

df

location = st.selectbox("Location:", options=df['location'].unique())
BHK = st.selectbox("BHK:", options=sorted(df['BHK'].unique()))
sqft = st.number_input("Total square feet:", min_value=300)
bath= st.selectbox("bathroom:", options=sorted(df['BHK'].unique()))

encoded_loc = encoder.transform(df["location"])

new_data = [[BHK,sqft,bath,encoded_loc[0]]]

if st.button("Prdect the house price"):
    pred = model.predict(new_data)[0]
    pred = pred * 100000
    st.subheader(f"Predicted Price: Rs.{pred}")