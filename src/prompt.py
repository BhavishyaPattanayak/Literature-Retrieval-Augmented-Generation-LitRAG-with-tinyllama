def build_prompt(query: str, docs: list) -> str:
    context = "\n\n".join(docs)
    return f"""Answer the following question based only on the context below.

Context:
{context}

Question: {query}

Answer:"""
