from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import random, spacy, json, os
import torch
import torch.nn.functional as NNF
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

app = Flask(__name__)
app.secret_key = 'your_secret_key'
nlp = spacy.load("en_core_web_sm")

# Load Transformer model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Load intents from the JSON file
with open('About us.json', 'r') as file:
    intents = json.load(file)['intents']

users = {}

def load_users():
    global users
    if os.path.exists('users.json'):
        with open('users.json', 'r') as file:
            users = json.load(file)

def save_users():
    with open('users.json', 'w') as file:
        json.dump(users, file)

load_users()

# Mean pooling for sentence embeddings
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Function to compute similarity scores using Transformer embeddings
def compute_similarity(input_sentence, sentences):
    encoded_input = tokenizer(input_sentence, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        input_model_output = model(**encoded_input)
    input_sentence_embedding = mean_pooling(input_model_output, encoded_input['attention_mask'])
    input_sentence_embedding = NNF.normalize(input_sentence_embedding, p=2, dim=1)

    encoded_sentences = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        sentences_model_output = model(**encoded_sentences)
    sentences_embeddings = mean_pooling(sentences_model_output, encoded_sentences['attention_mask'])
    sentences_embeddings = NNF.normalize(sentences_embeddings, p=2, dim=1)

    similarities = cosine_similarity(input_sentence_embedding, sentences_embeddings)
    sentences_with_scores = list(zip(sentences, similarities[0]))
    sorted_sentences = sorted(sentences_with_scores, key=lambda x: x[1], reverse=True)

    return sorted_sentences[0][0]  # Return the best matching sentence

# Route for the chatbot
@app.route('/chatbot')
def chatbot():
    session['conversation'] = []
    return render_template('index.html')

# Chatbot interaction route
@app.route('/ask', methods=['POST'])
def ask():
    user_message = request.form['message'].lower()
    session['conversation'].append({'user': user_message})

    # Collect patterns from the intents to use them for sentence matching
    predefined_patterns = []
    intent_responses = {}
    for intent in intents:
        predefined_patterns.extend(intent['patterns'])
        for pattern in intent['patterns']:
            intent_responses[pattern] = random.choice(intent['responses'])

    # Find the best matching sentence from predefined patterns
    best_match = compute_similarity(user_message, predefined_patterns)

    # Retrieve response based on the best matching pattern
    response = intent_responses.get(best_match, "I'm not sure how to respond to that, but I'm learning!")
    
    session['conversation'].append({'bot': response})
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)
