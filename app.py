import streamlit as st
import pickle
from src.preprocess import clean_tweet
model = pickle.load(open("src/sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("src/vectorizer.pkl", "rb"))

st.set_page_config(page_title="Twitter Sentiment Analysis", layout="centered")

st.title("🐦 Twitter Sentiment Analysis")
st.write("Analyze sentiment of tweets using NLP and Machine Learning")

tweet = st.text_area("Enter a tweet:")

if st.button("Analyze Sentiment"):
    if tweet.strip() == "":
        st.warning("Please enter some text")
    else:
        cleaned = clean_tweet(tweet)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == "positive":
            st.success(" Positive Sentiment")
        elif prediction == "negative":
            st.error(" Negative Sentiment")
        else:
            st.info("Neutral Sentiment")
