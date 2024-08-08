import pandas as pd
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

model = pk.load(open('model.pkl', 'rb'))
scaler = pk.load(open('scaler.pkl', 'rb'))

st.title("Movie Review Sentiment Analysis")

st.write("""
This app uses a machine learning model to predict whether a movie review is positive or negative.
Simply enter a movie review in the text box below and click on the 'Predict' button.
""")

review = st.text_area('Enter Movie Review:', height=200)

if st.button('Predict'):
    if review:
        review_scaled = scaler.transform([review])
        result = model.predict(review_scaled)
        if result[0] == 0:
            st.markdown("<h2 style='color: red;'>Negative Review</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='color: green;'>Positive Review</h2>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a review to get a prediction.")

