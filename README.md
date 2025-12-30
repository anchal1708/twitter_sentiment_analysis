# Twitter Sentiment Analysis

This project performs sentiment analysis on Twitter data to classify tweets as **Positive**, **Negative**, or **Neutral** using Natural Language Processing (NLP) techniques.

## Tech Stack
- Python
- NLTK
- Scikit-learn
- Pandas

## Project Structure
twitter_sentiment_analysis/
├── data/
│ └── tweets.csv
├── src/
│ ├── preprocess.py
│ ├── model.py
│ └── predict.py
├── requirements.txt
└── README.md



## How to Run
1. Install dependencies  
   `pip install -r requirements.txt`
2. Train the model  
   `python src/model.py`
3. Predict sentiment  
   `python src/predict.py`


   Key Features

🧹 Tweet cleaning and preprocessing (URLs, mentions, stopwords removal)

🧠 NLP-based feature extraction using TF-IDF

🤖 Machine Learning model using Logistic Regression

📈 Model evaluation with accuracy and classification report

🔮 Predict sentiment of custom user input

📁 Clean and modular project structure



### Model Details

1.Algorithm: Logistic Regression

2.Feature Extraction: TF-IDF (max_features=5000)

3.Evaluation Metrics: Accuracy, Precision, Recall, F1-score


## Future Improvements

Add Streamlit web interface

Use LSTM / BERT for better accuracy

Live Twitter API integration

Hyperparameter tuning

Multilingual sentiment support


