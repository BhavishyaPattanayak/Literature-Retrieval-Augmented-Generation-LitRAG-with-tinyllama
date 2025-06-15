import sys
import os
import pickle
import faiss

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from engine import LitRAGEngine
from embedding import load_embedding_model
from indexing import build_faiss_index

BOOK_PATH = os.path.join(os.path.dirname(__file__), "books", "frankenstein.txt")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "data", "frankenstein.index")
DOCS_PATH = os.path.join(os.path.dirname(__file__), "data", "frankenstein_docs.pkl")

with open(BOOK_PATH, "r", encoding="utf-8") as f:
    book_text = f.read()

engine = LitRAGEngine()

if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
    print("Index already exists.")
else:
    print("Building new index...")
    engine.load_text(book_text)
    engine.build_index()
    os.makedirs("data", exist_ok=True)
    pickle.dump(engine.documents, open(DOCS_PATH, "wb"))
    faiss.write_index(engine.index, INDEX_PATH)
    print("Saved index and documents.")
