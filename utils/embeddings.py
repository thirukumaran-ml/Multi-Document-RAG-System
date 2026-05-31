from sentence_transformers import SentenceTransformer


def load_embedding_model():
    """
    Load embedding model.
    """

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    return model