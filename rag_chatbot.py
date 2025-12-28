# Step 5: RAG Integration (Retriever + Neural Generator)
# Deep Blue Project – Accounting Helpdesk

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
# 3. BUILD CORPUS
# =============================
corpus = []
answers = []

for doc in documents:
    combined = doc['question'] + ' ' + doc['answer']
    corpus.append(preprocess_text(combined))
    answers.append(doc['answer'])

# =============================
# 4. TF-IDF RETRIEVER
# =============================
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(corpus)

# =============================
# 5. NEURAL NETWORK (CLASSIFIER)
# =============================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(answers)
y = to_categorical(y_encoded)

model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Training RAG neural generator...")
model.fit(X.toarray(), y, epochs=15, batch_size=16)

# =============================
# 6. RAG CHAT FUNCTION
# =============================
def rag_chat():
    print("\nRAG Chatbot Ready (type 'exit' to quit)")
    while True:
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            break

        query_clean = preprocess_text(user_input)
        query_vec = vectorizer.transform([query_clean])

        similarities = cosine_similarity(query_vec, X)[0]
        best_index = np.argmax(similarities)

        predicted = model.predict(query_vec.toarray())
        final_index = np.argmax(predicted)

        print("Bot:", label_encoder.inverse_transform([final_index])[0])


if __name__ == '__main__':
    rag_chat()