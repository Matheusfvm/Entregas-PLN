import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def search_similar_documents(query_vector, document_vectors):
    similarities = cosine_similarity(query_vector, document_vectors)[0]
    
    similar_indices = np.argsort(similarities)[::-1]
    
    return [(idx, similarities[idx]) for idx in similar_indices]

def format_search_results(similar_docs, documents):
    results = []
    for idx, similarity in similar_docs:
        if similarity > 0:
            results.append({
                'doc': documents[idx],
                'similarity': float(similarity)
            })
        if len(results) >= 5:
            break
    return results