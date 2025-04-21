#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import necessary dependencies

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
import seaborn as sns


# In[5]:


# Load the saved model and vectorizer

model = joblib.load("news_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit app
st.title("Real-Time News Text Analysis Dashboard")

# Text input from user
user_input = st.text_area("Enter a news article or headline: ")

if user_input:
    # Preprocessed Text
    processed_text = [user_input.lower()]

    # Convert to TF-IDF
    vectorized_text = vectorizer.transform(processed_text)

    # Predict category
    prediction = model.predict(vectorized_text)[0]

    # Sentiment Analysis
    sentiment_score = TextBlob(user_input).sentiment.polarity
    sentiment = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"

    # Display results
    st.subheader("📌Prediction:")
    st.write(f"**Category:** {prediction}")
    st.write(f"**Sentiment:** {sentiment} (Score: {sentiment_score:.2f})")

    # Real-Time Text Word Count and Sentiment Plot - Getting visual feedback for each submitted text
    st.subheader("📈 Your Text Stats:")

    # Word Count
    word_count = len(user_input.split())
    st.metric("📝 Word Count", word_count)

    # Sentiment Line
    st.write("#### Sentiment Score")
    st.line_chart(pd.DataFrame({"Sentiment Score": [sentiment_score]}))

    # Generate Word Cloud
    wordcloud = WordCloud(width=600, height=300, background_color="White").generate(user_input)

    st.subheader("📌Word Cloud:")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    # Show data summary
    df = pd.read_excel("bbc-text.xlsx")
    st.subheader("📊News Data Summary")

    # Category Distribution Bar Chart
    st.write("### Distribution of News Categories")
    fig, ax = plt.subplots()
    sns.countplot(x = "category", data=df, palette="viridis", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Pie Chart of Category proportions
    st.subheader("🥧 Category Proportion (in dataset)")
    fig1, ax1 = plt.subplots()
    category_counts = df["category"].value_counts()
    ax1.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
    ax1.axis('equal')    # Equal aspect ratio ensures pie is circular
    st.pyplot(fig1)

    # Article Length by Category
    df["text_length"] = df["text"].apply(lambda x: len(x.split()))

    st.write("### 📦 Text Length Distribution by Category")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.boxplot(x="category", y="text_length", data=df, palette="Set3", ax=ax2)
    plt.xticks(rotation=45)
    st.pyplot(fig2)




