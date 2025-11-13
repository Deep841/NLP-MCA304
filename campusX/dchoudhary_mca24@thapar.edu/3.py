"""
Offline Chat Reply Recommendation System
Round 4 – AI/ML Developer Intern
Author: Deep Chaudhary
"""

import pandas as pd
import numpy as np
import joblib
import os, re, argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

# ----------------------------
# Utility: Clean text
# ----------------------------
def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9' ]", " ", text)
    tokens = [w.lower() for w in word_tokenize(text) if w.lower() not in STOPWORDS]
    return " ".join(tokens)

# ----------------------------
# Build dataset
# ----------------------------
def build_dataset(pathA, pathB):
    dfA = pd.read_csv(pathA)
    dfB = pd.read_csv(pathB)
    df = pd.concat([dfA, dfB]).sort_values(['Conversation ID', 'Timestamp']).reset_index(drop=True)
    df['Message'] = df['Message'].astype(str).apply(clean_text)
    print(f"Dataset loaded: {len(df)} messages")
    return df

# ----------------------------
# Build QA pairs
# ----------------------------
def build_pairs(df):
    pairs = []
    for cid, group in df.groupby('Conversation ID'):
        group = group.sort_values('Timestamp').reset_index(drop=True)
        for i in range(len(group)-1):
            cur_sender = group.loc[i, 'Sender']
            nxt_sender = group.loc[i+1, 'Sender']
            if cur_sender != nxt_sender:
                pairs.append({
                    'ConversationID': cid,
                    'question': group.loc[i, 'Message'],
                    'answer': group.loc[i+1, 'Message']
                })
    qa = pd.DataFrame(pairs)
    print(f"QA pairs extracted: {len(qa)}")
    return qa

# ----------------------------
# Train TF-IDF model
# ----------------------------
def train_model(qa):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(qa['question'])
    model = {'vectorizer': vectorizer, 'answers': qa['answer'].values}
    joblib.dump(model, 'Model.joblib')
    print("Model saved -> Model.joblib")
    return model

# ----------------------------
# Predict reply
# ----------------------------
def predict_reply(model, query):
    query_clean = clean_text(query)
    vect = model['vectorizer'].transform([query_clean])
    similarities = cosine_similarity(vect, model['vectorizer'].transform(model['vectorizer'].get_feature_names_out().reshape(-1, 1)))
    # fallback for TFIDF matrix form
    sims = cosine_similarity(vect, model['vectorizer'].transform([clean_text(q) for q in model['vectorizer'].get_feature_names_out()]))
    idx = np.argmax(sims)
    return model['answers'][idx]

# ----------------------------
# Main run
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--userA', default='/Desktop/Dataset/UserA_chats.csv')
    parser.add_argument('--userB', default='/Desktop/Dataset/UserB_chats.csv')
    parser.add_argument('--query', type=str, default="How was the movie?")
    args = parser.parse_args()

    df = build_dataset(args.userA, args.userB)
    qa = build_pairs(df)
    model = train_model(qa)
    reply = predict_reply(model, args.query)
    print(f"User Query: {args.query}")
    print(f"Predicted Reply: {reply}")