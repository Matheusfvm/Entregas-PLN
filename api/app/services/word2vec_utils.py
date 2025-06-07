import os
from gensim.models import Word2Vec
import numpy as np

MODEL_PATH = "app/data/models/word2vec.model"
VECTORS_PATH = "app/data/documents_vectors.npy"

def train_model(sentences, vector_size=100, window=5, min_count=1, epochs=10):
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        workers=4
    )

    return model

def save_model(model, model_path=MODEL_PATH):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Model saved to {model_path}")

def load_model(model_path=MODEL_PATH):
    if os.path.exists(model_path):
        return Word2Vec.load(model_path)
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
def generate_vector(sentence, model):
    vectors = []
    for word in sentence:
        if word in model.wv:
            vectors.append(model.wv[word])
    
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)
    
def save_documents_vectors(vectors, vectors_path=VECTORS_PATH):
    os.makedirs(os.path.dirname(vectors_path), exist_ok=True)
    np.save(vectors_path, vectors)

def load_documents_vectors(vectors_path=VECTORS_PATH):
    if os.path.exists(vectors_path):
        return np.load(vectors_path)
    else:
        raise FileNotFoundError(f"Vectors file not found at {vectors_path}")