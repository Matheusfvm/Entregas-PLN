from flask import Blueprint, request, jsonify
from app.services.preprocessing import preprocess_text
from app.services.word2vec_utils import load_model, load_documents_vectors, generate_vector
from app.services.search import search_similar_documents, format_search_results
from app.data.documents import documents

bp = Blueprint('semantic-search', __name__)

@bp.route('/search', methods=['POST'])
def semantic_search():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'error': 'Termos de busca não fornecidos'}), 400
    
    query = data['query']
    preprocessed_query = preprocess_text(query)

    model = load_model()
    query_vector = generate_vector(preprocessed_query, model).reshape(1, -1)

    document_vectors = load_documents_vectors()
    similar_docs = search_similar_documents(query_vector, document_vectors)

    results = format_search_results(similar_docs, documents)

    return jsonify({
        'query': query,
        'results': results
    }), 200