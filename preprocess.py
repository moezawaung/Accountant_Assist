# Text Preprocessing for Accounting Helpdesk
# This script loads accounting_docs.json and prepares clean text for TF-IDF


# REQUIRED LIBRARIES

import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# =============================
# NLTK DOWNLOADS (RUN ONCE)
# =============================
# If you get errors, uncomment these lines and run once
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


# LOAD DATASET

with open('data/accounting_docs.json', 'r', encoding='utf-8') as f:
    documents = json.load(f)

print(f"Loaded {len(documents)} documents")


# INITIALIZE NLP TOOLS

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# CLEANING FUNCTION

def preprocess_text(text):
    # Lowercase
    text = text.lower()

    # Remove special characters & numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords & lemmatize
    clean_tokens = []
    for word in tokens:
        if word not in stop_words:
            lemma = lemmatizer.lemmatize(word)
            clean_tokens.append(lemma)

    return ' '.join(clean_tokens)

# APPLY PREPROCESSING

corpus = []          # cleaned text for TF-IDF
original_answers = []  # keep original answers

for doc in documents:
    combined_text = doc['question'] + ' ' + doc['answer']
    cleaned_text = preprocess_text(combined_text)
    corpus.append(cleaned_text)
    original_answers.append(doc['answer'])


# 7. PREVIEW RESULTS

print("\nSample cleaned text:")
print(corpus[0])
