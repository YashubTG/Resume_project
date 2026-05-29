# Resume Screening System using Machine Learning and NLP

## Project Overview

This project is an end-to-end Resume Screening System built using Machine Learning, Natural Language Processing (NLP), and Streamlit. The system analyzes resume text, predicts the most suitable job category, and recommends the top matching job roles based on semantic similarity.

The project was developed as a practical data science and NLP portfolio project to demonstrate:

* Text preprocessing and cleaning
* NLP techniques
* Resume classification
* TF-IDF vectorization
* Cosine similarity-based recommendation systems
* Streamlit deployment
* End-to-end machine learning workflow

---

# Features

* Predicts the job category of a resume
* Recommends top 3 matching job roles
* Uses NLP preprocessing and text normalization
* Interactive Streamlit web application
* Real-time resume screening
* Clean and user-friendly interface

---

# Technologies Used

## Programming Language

* Python

## Libraries and Frameworks

* Streamlit
* Pandas
* NumPy
* Scikit-learn
* NLTK
* XGBoost
* LightGBM
* Pickle
* Regular Expressions (re)

---

# Machine Learning and NLP Workflow

## 1. Data Preprocessing

The resume dataset was cleaned and preprocessed using:

* Lowercasing
* Removal of special characters
* Removal of stopwords
* Lemmatization
* Regex-based text cleaning
* Removal of unnecessary sections and noise

## 2. Feature Engineering

TF-IDF Vectorization was used to convert textual resume data into numerical feature vectors.

## 3. Model Training and Evaluation

Multiple machine learning algorithms were trained and evaluated for resume classification:

* Logistic Regression
* Linear Support Vector Classifier (Linear SVC)
* Naive Bayes
* Random Forest
* XGBoost
* LightGBM

The models were compared using classification performance metrics, and LightGBM achieved the best overall results. Therefore, LightGBM was selected as the final model for deployment in the Resume Screening System.

## 4. Resume Classification

The final LightGBM classification model predicts the most suitable job category based on resume content.

## 5. Job Recommendation System

A cosine similarity-based recommendation system was implemented to identify the top matching job roles based on semantic similarity between resumes and job descriptions.

---

# Streamlit Application

The project was deployed as an interactive Streamlit web application.

Users can:

1. Paste resume text
2. Click the screening button
3. Receive:

   * Predicted job category
   * Top 3 recommended job roles

---

# Project Structure

```bash
Resume_project/
│
├── app.py
├── requirements.txt
├── README.md
│
├── resume_classifier.pkl
├── tfidf_vectorizer.pkl
├── label_encoder.pkl
├── tfidf2.pkl
├── job_desc_vectors.pkl
├── job_roles.pkl
│
├── notebooks/
│   ├── classification_part.ipynb
│   ├── recommendation_system.ipynb
│   └── combined_code.ipynb
│
└── data/
```

---

# How to Run the Project Locally

## 1. Clone the Repository

```bash
git clone <repository-link>
```

## 2. Open Project Folder

```bash
cd Resume_project
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run Streamlit App

```bash
streamlit run app.py
```

---

# Future Improvements

* Resume PDF upload functionality
* Improved UI design
* Advanced recommendation system
* Deep learning-based NLP models
* Skill extraction and visualization
* Deployment optimization

---

# Learning Outcomes

Through this project, the following concepts were practiced:

* Machine Learning workflow
* NLP preprocessing techniques
* Feature engineering using TF-IDF
* Model comparison and evaluation
* Classification algorithms
* Ensemble learning methods
* Model serialization using Pickle
* Cosine similarity
* Streamlit web app development
* End-to-end deployment workflow
* Model integration into web applications

---

## Live Demo

https://your-app-name.streamlit.app

# Conclusion

This project demonstrates how machine learning and NLP can be used to automate resume screening and job role recommendation. Multiple machine learning models were evaluated, including Logistic Regression, Linear SVC, Naive Bayes, Random Forest, XGBoost, and LightGBM. LightGBM delivered the best performance and was selected for deployment. The project combines classification and semantic similarity techniques into a complete end-to-end application with an interactive web interface.

