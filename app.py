from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

app = Flask(__name__)

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

# Load tokenizer and model for QA
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Load sentence transformer model for embedding
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Read corpus from corpus.txt
def read_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        corpus_text = file.read()
    return corpus_text.split('\n\n')  # Split by paragraphs

# Path to your corpus.txt file
corpus_file = 'corpus.txt'
corpus = read_corpus(corpus_file)
corpus_chunks = corpus

# Embed corpus chunks
corpus_embeddings = embedder.encode(corpus_chunks)

# Create FAISS index and add corpus embeddings
dimension = corpus_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(np.array(corpus_embeddings))

# Store conversation history
conversation_history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        user_message = request.json['message']
        logging.debug(f"User message: {user_message}")
        conversation_history.append(user_message)

        # Encode user query and search for similar chunks
        user_embedding = embedder.encode([user_message])
        D, I = faiss_index.search(user_embedding, k=3)  # Retrieve top 3 similar chunks
        logging.debug(f"Retrieved indices: {I[0]}")

        combined_context = ""
        for idx in I[0]:
            if idx != -1:
                combined_context += corpus_chunks[idx] + " "
        
        combined_context = combined_context.strip()
        if not combined_context:
            combined_context = "I'm sorry, I couldn't find any relevant information in the corpus."
        
        logging.debug(f"Combined context: {combined_context[:500]}")  # Log the first 500 characters of context

        result = qa_pipeline(question=user_message, context=combined_context)
        best_answer = result['answer']
        logging.debug(f"Answer: {best_answer}, Score: {result['score']}")

        # Check answer length and confidence score
        if best_answer is None or result['score'] < 0.5 or len(best_answer) < 5:
            best_answer = "Please contact us directly for more information."

        conversation_history.append(best_answer)
        return jsonify({'message': best_answer})

    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({'message': "An error occurred. Please try again."})

if __name__ == '__main__':
    app.run(debug=True)
