# Twitter-Sentiment-Analysis
Overview
This project aims to detect hate speech in tweets using machine learning techniques. The classification task is binary: tweets are labeled as either Hate Speech or Not Hate Speech. The model employs text preprocessing, TF-IDF vectorization, and Logistic Regression with class balancing for prediction.

#Features
Text preprocessing: removal of Twitter handles, URLs, punctuation, and conversion to lowercase.

TF-IDF vectorization incorporating unigrams and bigrams.

Handling of class imbalance via logistic regression with balanced class weights.

Exploratory Data Analysis (EDA) including class distribution plots and word clouds.

Evaluation metrics: Accuracy, F1-Score, and full classification reports.

Persistence of model and vectorizer using pickle for deployment.

Prediction function for quick inference on new tweets.

Simple Flask or Streamlit app setup for user interaction (optional).

#Getting Started
Prerequisites
Python 3.7+

Packages: pandas, numpy, scikit-learn, matplotlib, seaborn, wordcloud, flask or streamlit (if using UI)

Install dependencies with:

bash
pip install pandas numpy scikit-learn matplotlib seaborn wordcloud flask streamlit
Dataset
The dataset Twitter Sentiments.csv should contain the following columns:

tweet: Text of the tweet

label: Binary label (0 = Not Hate Speech, 1 = Hate Speech)

Running the Training Pipeline
Load and preprocess the data (removes handles, URLs, and punctuation).

Vectorize tweets using TF-IDF.

Split dataset with stratification for balanced train/test sets.

Train Logistic Regression with balanced class weights.

Evaluate using Accuracy, F1-score, and classification report.

Save the trained model and vectorizer.
