from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)
