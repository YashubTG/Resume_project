

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

#nltk.download('stopwords')
#nltk.download('wordnet')

try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

try:
    WordNetLemmatizer()
except:
    nltk.download('wordnet')

# Load all saved files
with open("resume_classifier.pkl", "rb") as f:
    loaded_model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    loaded_tfidf = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    loaded_le = pickle.load(f)

with open("tfidf2.pkl", "rb") as f:
    loaded_tfidf2 = pickle.load(f)

with open("job_desc_vectors.pkl", "rb") as f:
    loaded_job_desc_vectors = pickle.load(f)

with open("job_roles.pkl", "rb") as f:
    loaded_job_roles = pickle.load(f)

# Cleaning functions
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_resume(text):
    text = re.sub(r'(linkedin|github)\S+', '', text)
    text = re.sub(r'Exprience\s*-\s*[\w\s<]+months?', '', text)
    text = re.sub(r'(Skill Details|Company Details|Education Details|company\s*-|description\s*-)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b', '', text)
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\b\w{1,2}\b', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def lemmatize_text(text):
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# Combined prediction function
def predict(resume_text):
    cleaned = clean_resume(resume_text)
    lemmatized = lemmatize_text(cleaned)

    # Model 1 - Classification
    vectorized_classify = loaded_tfidf.transform([lemmatized])
    vectorized_classify = pd.DataFrame(
        vectorized_classify.toarray(),
        columns=loaded_tfidf.get_feature_names_out()
    )
    prediction = loaded_model.predict(vectorized_classify)
    category = loaded_le.inverse_transform(prediction)[0]

    # Model 2 - NLP Similarity
    vectorized_top3 = loaded_tfidf2.transform([lemmatized])
    similarity_scores = cosine_similarity(
        vectorized_top3,
        loaded_job_desc_vectors
    ).flatten()
    top3_indices = similarity_scores.argsort()[::-1][:3]

    top3 = []
    for ind in top3_indices:
        top3.append({
            "Job Role": loaded_job_roles[ind],
            "Score": round(float(similarity_scores[ind]), 4)
        })

    return category, top3

# Streamlit UI
st.title("Resume Screening System")
st.write("Paste your resume below to get job category and top matching job roles")

resume_input = st.text_area("Paste Resume Here", height=300)

if st.button("Screen Resume"):
    if resume_input.strip() == "":
        st.warning("Please paste a resume first!")
    else:
        with st.spinner("Analyzing resume..."):
            category, top3 = predict(resume_input)

        st.success("Analysis Complete!")

        st.subheader("Predicted Job Category:")
        st.info(category)

        st.subheader("Top 3 Matching Job Roles:")
        df_results = pd.DataFrame(top3)
        st.table(df_results)