# Literature-Retrieval-Augmented-Generation-(LitRAG)-with-tinyllama

LitRAG (Literature Retrieval-Augmented Generation) is a lightweight, local RAG (Retrieval-Augmented Generation) pipeline for question answering on literary texts. This version uses FAISS for retrieval and TinyLlama-1.1B-Chat for generation — all running offline once set up.

## Project Structure

```plaintext
LitRAG/
  ├── data/
  │   ├── frankenstein_docs.pkl      -> Pre-split text chunks from Frankenstein
  │   └── frankenstein.index         -> FAISS index built on document embeddings
  ├── src/
  │   ├── engine.py                  -> RAG engine that combines retrieval and prompt generation
  │   ├── embedding.py               -> Embedding logic using sentence-transformers
  │   ├── splitter.py                -> Splits raw text into overlapping chunks
  │   ├── indexing.py                -> Builds and loads FAISS indexes
  │   └── prompt.py                  -> Formats context and query into a LLM prompt
  ├── main.py                   -> Interactive QA loop (TinyLlama + FAISS)
  ├── TinyLLama_main.py            -> Loads TinyLlama and starts answering questions
  └── saved_model/                   -> (Optional) model cache from Hugging Face (excluded from GitHub)
```
## Dataset 

Text Corpus: Frankenstein; Source: Project Gutenberg [ID: 84]

## Training Setup

- Method: LangChain’s RecursiveCharacterTextSplitter

- Embedding Model: all-MiniLM-L6-v2 from sentence-transformers (384-dimensional vector embeddings)

- Indexing: FAISS flat index (IndexFlatL2)

- Document Store: Chunked documents stored as frankenstein_docs.pkl (pickle format)
