from sklearn.metrics.pairwise import cosine_similarity

from preprocess import preprocess_text

def predict_intent(user_input, threshold, vectorizer, pair_vectors, pair_responses):
    # 1) Pré-processa + stem + remove acento/ponto
    u = preprocess_text(user_input)
    # 2) Vetoriza
    u_vec = vectorizer.transform([u]).toarray()
    # 3) Cosine similarity contra todos os padrões
    sims = cosine_similarity(u_vec, pair_vectors)[0]
    max_sim = sims.max()
    if max_sim < threshold:
        return "Desculpa, não entendi sua pergunta"
    # 4) Índice do padrão mais próximo → retorna a resposta exata
    idx = sims.argmax()
    return pair_responses[idx]