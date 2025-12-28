import json
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# Load Data
# ----------------------------
with open('data/accounting_docs.json', 'r', encoding='utf-8') as f:
    documents = json.load(f)

# ----------------------------
# Preprocessing
# ----------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

questions = []
answers = []

for doc in documents:
    questions.append(preprocess_text(doc['question']))
    answers.append(doc['answer'])

# ----------------------------
# Tokenization
# ----------------------------
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(questions)

sequences = tokenizer.texts_to_sequences(questions)
max_len = max(len(seq) for seq in sequences)

X = pad_sequences(sequences, maxlen=max_len, padding='post')

# ----------------------------
# Encode Labels
# ----------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(answers)
y = to_categorical(y_encoded)

# ----------------------------
# Build LSTM Model
# ----------------------------
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=max_len),
    LSTM(128, return_sequences=False),
    Dropout(0.3),
    Dense(y.shape[1], activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Training LSTM model...")
model.fit(X, y, epochs=15, batch_size=16)

# ----------------------------
# Chat Function
# ----------------------------
def chat():
    print("\nLSTM Accounting Chatbot Ready (type 'exit')")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        processed = preprocess_text(user_input)
        seq = tokenizer.texts_to_sequences([processed])
        padded = pad_sequences(seq, maxlen=max_len, padding='post')

        pred = model.predict(padded)
        answer_index = np.argmax(pred)

        print("Assistant:", label_encoder.inverse_transform([answer_index])[0])

if __name__ == '__main__':
    chat()
