# Disaster Tweet Classification Using NLP

## Overview
This project aims to build a machine learning model to distinguish between tweets that are about real disasters and those that are not. With the ubiquity of smartphones, people can report emergencies they're observing in real-time via platforms like Twitter. 
However, distinguishing between actual disaster-related tweets and metaphorical or non-disaster related tweets can be challenging for automated systems. This project tackles that challenge using Natural Language Processing (NLP).

## Model
The project utilizes the BERT (Bidirectional Encoder Representations from Transformers) model for tweet classification. BERT has shown great success in various NLP tasks and is well-suited for understanding the context of words in a sentence.

## Implementation
The implementation involves several steps:
- Data loading and preprocessing
- Tokenization of tweets using the BERT tokenizer
- Training a BERT model for sequence classification
- Evaluating the model's performance on a validation set

### Libraries 
- PyTorch
- Transformers library
- Pandas for data manipulation
- Scikit-learn for calculating metrics

### Usage
The main steps of the project are implemented in a Jupyter Notebook, making it easy to follow along and experiment with different approaches.

1. **Data Preparation**: Split the dataset into training and validation sets.
2. **Tokenization**: Tokenize the text data using the BERT tokenizer.
3. **Model Training**: Train the BERT model on the processed data.
4. **Evaluation**: Evaluate the model's performance on the validation set using metrics like accuracy and F1 score.

## Results
The model's effectiveness is measured using the F1 score, which balances precision and recall. The training process is detailed in the provided Jupyter Notebook, including the calculation of loss and F1 score after each epoch.

### Dataset Source
The dataset for this project was obtained from the Kaggle competition: [NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started/data).
