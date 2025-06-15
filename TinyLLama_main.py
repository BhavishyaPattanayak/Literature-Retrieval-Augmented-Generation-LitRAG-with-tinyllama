from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import sys
import pickle
import faiss

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)
qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

def answer_with_tinyllama(prompt: str, max_tokens=150):
    response = qa_pipeline(
        prompt,
        max_new_tokens=max_tokens,
        do_sample=False,
        temperature=0.2,
        return_full_text=False
    )
    return response[0]['generated_text'].strip()

sys.path.append("LitRAG/src")
from engine import LitRAGEngine

engine = LitRAGEngine()
engine.documents = pickle.load(open("LitRAG/data/frankenstein_docs.pkl", "rb"))
engine.index = faiss.read_index("LitRAG/data/frankenstein.index")

print("Ask anything about Frankenstein (type 'exit' to quit)")
while True:
    query = input("\nYour question: ")
    if query.strip().lower() == "exit":
        break
    prompt = engine.generate_prompt(query)
    print("\nAnswer:\n")
    print(answer_with_tinyllama(prompt))
