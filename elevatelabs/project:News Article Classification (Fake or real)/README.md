# Project 6: Fake News Article Classification – AI & ML Internship

This project demonstrates how to classify news articles as **Fake** or **Real** using Natural Language Processing (NLP) and machine learning models.

## 🎯 Objective

- Build a classifier to detect fake news articles
- Apply text preprocessing and vectorization
- Train and evaluate machine learning models
- Save the model for deployment

## 🛠 Tools Used

- Google Colab
- Python
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - nltk
  - scikit-learn
  - joblib

## 📄 Dataset Used

- **Fake and Real News Dataset**
- Files: `Fake.csv`, `True.csv`
- Source: [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

## 🚀 Steps Performed

### 1. Data Loading & Labeling
- Loaded `Fake.csv` and `True.csv`
- Labeled Fake news as `0` and Real news as `1`
- Combined datasets and shuffled rows

### 2. Text Preprocessing
- Converted text to lowercase
- Removed punctuation and stopwords
- Applied stemming using NLTK

### 3. Feature Engineering
- Converted text to numerical form using TF-IDF Vectorizer

### 4. Model Training
- Trained **Logistic Regression** and **Naive Bayes** classifiers
- Evaluated models using accuracy, precision, recall, and F1-score

### 5. Evaluation
- Visualized confusion matrices
- Compared Logistic Regression vs Naive Bayes

### 6. Saving Model
- Saved the trained model (`news_classifier_model.pkl`)
- Saved the TF-IDF vectorizer (`tfidf_vectorizer.pkl`) for deployment

## 📂 Output Files

- `news_article_classification.ipynb` – Colab notebook
- `news_classifier_model.pkl` – Trained model file
- `tfidf_vectorizer.pkl` – Saved vectorizer
- `README.md` – This documentation file

## 👤 Author

shrey sakhiya  
AI & ML Internship Participant

## 📌 Submission

All files are uploaded to this repository as part of Project 6 submission.
