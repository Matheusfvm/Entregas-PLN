from app.data.documents import documents
from app.data.training_sentences import training_sentences
from app.services.preprocessing import preprocess_text
from app.services.word2vec_utils import (
    train_model, 
    save_model,
    generate_vector,
    save_documents_vectors
)
def train_word2vec_model():
    try:
        print("Starting training of Word2Vec model...")

        all_training_data = documents + training_sentences
        print(f"Total training sentences: {len(all_training_data)}")

        if not all_training_data:
            raise ValueError("No documents available for training the model.")
        
        preprocessed_docs = preprocess_text(all_training_data)
        model = train_model(sentences=preprocessed_docs)
        save_model(model)

        print("Generating document vectors...")
        original_docs = preprocess_text(documents)
        vectors = [generate_vector(doc, model) for doc in original_docs]
        save_documents_vectors(vectors)
        
        print("Word2Vec model trained and saved successfully.")
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    train_word2vec_model()