# Step 3: TF-IDF Retriever + Cosine Similarity
# Deep Blue Project – Accounting Helpdesk

# This script retrieves top-k relevant documents for a user query

# =============================
# 1. REQUIRED LIBRARIES
# =============================
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =============================
# 2. LOAD PREPROCESSED DATA
# =============================
# We reuse the preprocessing logic from Step 2
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Uncomment only if needed
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

with open('data/accounting_docs.json', 'r', encoding='utf-8') as f:
    documents = json.load(f)

# =============================
# 3. PREPROCESSING FUNCTION
# =============================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    clean_tokens = []
    for word in tokens:
        if word not in stop_words:
            clean_tokens.append(lemmatizer.lemmatize(word))
    return ' '.join(clean_tokens)

# =============================
# 4. BUILD CORPUS
# =============================
corpus = []
answers = []

for doc in documents:
    combined = doc['question'] + ' ' + doc['answer']
    corpus.append(preprocess_text(combined))
    answers.append(doc['answer'])

# =============================
# 5. TF-IDF hi
# =============================
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# =============================
# 6. RETRIEVER FUNCTION
# =============================
def retrieve_context(user_query, top_k=3):
    query_clean = preprocess_text(user_query)
    query_vec = vectorizer.transform([query_clean])

    similarities = cosine_similarity(query_vec, X)[0]
    top_indices = similarities.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            'score': similarities[idx],
            'answer': answers[idx]
        })
    return results

# =============================
# 7. TEST RETRIEVER
# =============================
if __name__ == '__main__':
    print("TF-IDF Retriever Ready. Type 'exit' to quit.\n")
    while True:
        query = input("User: ")
        if query.lower() == 'exit':
            break

        retrieved = retrieve_context(query)
        print("\nTop Retrieved Answers:")
        for r in retrieved:
            print(f"- ({r['score']:.3f}) {r['answer']}")
        print()