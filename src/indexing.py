import faiss
import numpy as np

def build_faiss_index(embeddings):
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

def query_index(index, query_vector, k=5):
    distances, indices = index.search(query_vector, k)
    return indices[0]
