# Step 6: Evaluation & Testing
# Deep Blue Project – Accounting Helpdesk RAG

import json
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# =============================
# 1. LOAD DATA
# =============================
with open('data/accounting_docs.json', 'r', encoding='utf-8') as f:
    documents = json.load(f)

# =============================
# 2. PREPROCESSING
# =============================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# =============================
# 3. BUILD DATASET
# =============================
texts = []
labels = []

for doc in documents:
    combined = doc['question'] + ' ' + doc['answer']
    texts.append(preprocess_text(combined))
    labels.append(doc['answer'])

# =============================
# 4. TF-IDF
# =============================
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(texts).toarray()

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)
y = to_categorical(y_encoded)

# =============================
# 5. TRAIN MODEL
# =============================
model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10, batch_size=16, verbose=0)

# =============================
# 6. EVALUATION
# =============================
pred_probs = model.predict(X)
pred_labels = np.argmax(pred_probs, axis=1)
true_labels = np.argmax(y, axis=1)

acc = accuracy_score(true_labels, pred_labels)
print(f"Overall Accuracy: {acc:.2f}")

print("\nClassification Report (sample):")
print(classification_report(true_labels, pred_labels, zero_division=0))

# =============================
# 7. MANUAL TEST CASES
# =============================
test_questions = [
    "What is debit?",
    "Explain credit in accounting",
    "What is a balance sheet?",
    "How to record an expense?"
]

print("\nManual Test Results:")
for q in test_questions:
    q_vec = vectorizer.transform([preprocess_text(q)]).toarray()
    pred = model.predict(q_vec)
    ans = label_encoder.inverse_transform([np.argmax(pred)])[0]
    print(f"Q: {q}\nA: {ans}\n")