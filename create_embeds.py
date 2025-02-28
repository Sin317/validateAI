import pandas as pd
import numpy as np
import json
import ollama
from numpy.linalg import norm
from llama_index.embeddings.ollama import OllamaEmbedding

# Load CSV
df = pd.read_csv("startups.csv").head(1000)

# Convert each row into a document text
documents = [
    f"Name: {row['name']}\nCity: {row['city']}\nTagline: {row['tagline']}\nDescription: {row['description']}"
    for _, row in df.iterrows()
]

print("starting...")

# Initialize embedding model
embedding_model = OllamaEmbedding(model_name="llama3.2")  # Change model if needed

print("model initialized")

# Generate embeddings for each startup description
embeddings = [embedding_model.get_text_embedding(doc) for doc in documents]

print("done generating")
# Store embeddings in a JSON file
embedding_data = [{"text": doc, "embedding": emb} for doc, emb in zip(documents, embeddings)]


with open("embeddings_startups_1k.json", "w") as f:
    json.dump(embedding_data, f)

print("Embeddings stored successfully!")
