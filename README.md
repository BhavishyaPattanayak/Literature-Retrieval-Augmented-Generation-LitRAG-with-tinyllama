# Literature-Retrieval-Augmented-Generation-(LitRAG)-with-tinyllama

LitRAG (Literature Retrieval-Augmented Generation) is a lightweight, local RAG (Retrieval-Augmented Generation) pipeline for question answering on literary texts. This version uses FAISS for retrieval and TinyLlama-1.1B-Chat for generation, all running offline once set up.

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

## Key Features
- Offline-compatible: No external API calls required
- Lightweight: Uses TinyLlama-1.1B for fast, local inference
- Modular: Embeddings, prompt templates, and retrieval are all decoupled
- Extendable: Can swap text corpus, embeddings, or LLM with minimal changes

## Dataset 

Text Corpus: Frankenstein; Source: Project Gutenberg [ID: 84]

## Training Setup

- Method: LangChain’s RecursiveCharacterTextSplitter

- Embedding Model: all-MiniLM-L6-v2 from sentence-transformers (384-dimensional vector embeddings)

- Indexing: FAISS flat index (IndexFlatL2)

- Document Store: Chunked documents stored as frankenstein_docs.pkl (pickle format)

## Quickstart

```bash
git clone https://github.com/BhavishyaPattanayak/Literature-Retrieval-Augmented-Generation-LitRAG-with-tinyllama.git
cd LitRAG
pip install -r requirements.txt
```
## Setup Dependencies


    pip install sentence-transformers faiss-cpu transformers

## Training the Model

1. Run the FAISS Index builder:

       !python LitRAG/main.py
   
3. Start the interactive QA chatbot:

       !python LitRAG/TinyLLama_main.py

What It Does

- Loads and splits Frankenstein into ~800-token chunks with overlaps.
- Encodes the chunks using all-MiniLM-L6-v2.
- Builds a FAISS index for dense retrieval.
- Accepts a user query, retrieves top-k relevant chunks.
- Builds a context-rich prompt for TinyLlama.
- Generates a final answer using the TinyLlama-1.1B-Chat model.

## Example Queries

Question: who is frankenstein?

Answer: frankenstein is a monster, a being created by a madman, who would have us believe that he is a man.

Question: what is the monster's name?

Answer: the monster is called frankenstein.

Question: what is the monster's appearance?

Answer: the monster is a tall, thin, and gaunt man, with a long, thin face, and a long, thin nose. His eyes are black, and his hair is black, and his skin is pale.



