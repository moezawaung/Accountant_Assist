## Accounting Helpdesk AI Chatbot

### 📌 Project Overview
This Project is an AI-based accounting helpdesk chatbot.
It answers accounting-related questions using a classical
Retrieval-Augmented Generation (RAG) approach without using
any external APIs or large language models.

---

### 🎯 Objective
- Build a real-life usable chatbot
- Use classical NLP and neural networks
- Work fully offline
- Avoid APIs and LLMs

---

### 🧠 System Pipeline
User Query
→ Text Preprocessing (NLTK)
→ TF-IDF Vectorization
→ Cosine Similarity Retriever
→ Neural Network Generator
→ Final Answer

---

### 🛠 Technologies Used
- Python
- NLTK
- Scikit-learn
- TensorFlow

---

### 📂 How to Run (Step by Step)
-pip install -r requirements.txt
-python preprocess.py
-python retriever.py
-python rag_chatbot.py
-python lstm_chatbot.py
-python evaluation.py

