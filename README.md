# Resume_summarizer - NLP Based


## Overview:
This project leverages Natural Language Processing (NLP) and Machine Learning to automatically analyze and classify resumes based on their content. It mimics a real-world HR automation tool that can efficiently parse, clean, and predict the relevant job category from resumes using text classification techniques.

## Problem Statement:
Manual resume screening is inefficient and error-prone. This project aims to automate the process by training an NLP pipeline that classifies resumes into predefined job roles using machine learning and n-gram-based features.

## Steps Involved in the Project:

### 1. Resume Data Collection

Raw resume files in .txt format are collected and labeled by job category (e.g., Data Scientist, Developer, Blockchain).

### 2. Text Preprocessing (using NLTK):

Tokenization

Lowercasing

Stopword removal

Lemmatization

### 3. Feature Engineering:

Transforming cleaned text using TF-IDF Vectorization

Applying n-grams (unigram, bigram) for better feature representation

### 4. Model Building:

Training a Logistic Regression classifier on the TF-IDF features

Optional: Swap with other models (SVM, Random Forest)

### 5. Model Evaluation:

Metrics used: Accuracy, Precision, Recall, F1-Score

Confusion Matrix for class-wise insights

### 6. Prediction:

Classify new/unseen resumes into the most probable job domain

Output job label based on resume content

### 7. Visualization:

Plots for confusion matrix and evaluation metrics

Option to extend with WordClouds or feature importance

##  Tech Stack

Python

NLTK for NLP tasks

Scikit-learn for machine learning

Pandas, NumPy, Matplotlib

## How to Run

1. Clone the repository

2. Add .txt resume files to the data/ folder

3. Run the notebook step-by-step

4. View predictions and evaluation results

## Use Case
Perfect for HR systems, job portals, and recruitment firms looking to implement AI-powered resume screening and classification.

<img width="1837" height="525" alt="image" src="https://github.com/user-attachments/assets/d74e3281-3994-470c-962a-09c1e6c13528" />
