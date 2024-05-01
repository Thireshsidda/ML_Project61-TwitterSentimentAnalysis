# ML_Project61-TwitterSentimentAnalysis

# Twitter Sentiment Analysis

## Overview
This project aims to classify tweets into two categories: tweets containing hate speech (racist or sexist sentiment) and tweets that do not contain hate speech. The objective is to build a model that can automatically detect hate speech in tweets. The dataset used for training and evaluation contains labeled tweets, where label '1' denotes hate speech and label '0' denotes non-hate speech.

## Dataset Information
The dataset consists of 31,962 tweets provided in a CSV file format. Each tweet is accompanied by its unique ID, label (0 for non-hate speech, 1 for hate speech), and the tweet text itself.

## Code Structure
### 1. Data Loading and Preprocessing
- Load the dataset using pandas.
- Preprocess the text data by removing Twitter handles, special characters, numbers, and punctuations.
- Tokenize the tweets into individual words and apply stemming to reduce words to their root forms.

### 2. Exploratory Data Analysis (EDA)
- Visualize the most frequent words in both hate speech and non-hate speech tweets using word clouds.
- Extract hashtags from tweets and analyze their frequency in both categories.

### 3. Input Split
- Perform feature extraction using Bag-of-Words (BoW) representation.
- Split the dataset into training and testing sets.

### 4. Model Training and Evaluation
- Train a logistic regression model on the training data using BoW features.
- Evaluate the model's performance using F1-score and accuracy metrics on the testing data.
- Experiment with different probability thresholds for class prediction to optimize the model's performance.

## Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- re
- nltk
- scikit-learn
- wordcloud

## Usage
1. Ensure that the dataset file 'Twitter Sentiments.csv' is available.
2. Run the provided code in a Python environment with the required dependencies installed.
3. Analyze the EDA visualizations to understand the distribution of hate speech and non-hate speech tweets.
4. Train the logistic regression model using the provided code.
5. Evaluate the model's performance using F1-score and accuracy metrics.
6. Experiment with different preprocessing techniques and model architectures to improve performance if needed.

## Future Improvements
- Experiment with advanced natural language processing (NLP) techniques such as word embeddings and deep learning models (e.g., LSTM, BERT) for better classification accuracy.
- Fine-tune hyperparameters and explore different feature representations to enhance model performance.
- Incorporate additional features such as sentiment analysis and contextual information to improve hate speech detection.

## Acknowledgments
- Acknowledge any additional resources, libraries, or datasets used in the project.
