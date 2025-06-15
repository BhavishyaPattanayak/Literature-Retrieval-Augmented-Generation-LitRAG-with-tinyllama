from embedding import load_embedding_model
from splitter import split_text
from indexing import build_faiss_index, query_index
from prompt import build_prompt

class LitRAGEngine:
    def __init__(self):
        self.model = load_embedding_model()
        self.documents = []
        self.index = None

    def load_text(self, text: str):
        self.documents = split_text(text)

    def build_index(self):
        embeddings = self.model.encode(self.documents, show_progress_bar=True)
        self.index = build_faiss_index(embeddings)

    def retrieve(self, query: str, k=5):
        q_vector = self.model.encode([query])
        indices = query_index(self.index, q_vector, k)
        return [self.documents[i] for i in indices]

    def generate_prompt(self, query: str):
        docs = self.retrieve(query)
        return build_prompt(query, docs)
