from dotenv import load_dotenv
load_dotenv()

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

model = SentenceTransformer("all-MiniLM-L6-v2")
client = OpenAI()                          # reads OPENAI_API_KEY from env

index = pickle.load(open("index.pkl", "rb"))
CHUNKS, EMB = index["chunks"], index["embeddings"]

def retrieve(question, k=8):
    q = model.encode([question])[0]
    sims = EMB @ q / (np.linalg.norm(EMB, axis=1) * np.linalg.norm(q))
    top = np.argsort(sims)[::-1][:k]
    return [CHUNKS[i] for i in top]

SYSTEM = """You answer questions about the UK Innovator Founder visa.
Use ONLY the provided context. If the answer is not in the context, say:
"I don't know based on the official guidance." Always cite the source filename."""

def answer(question):
    hits = retrieve(question)
    context = "\n\n---\n\n".join(f"[source: {h['source']}]\n{h['text']}" for h in hits)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=600,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return resp.choices[0].message.content, hits

if __name__ == "__main__":
    while True:
        q = input("\nQuestion (blank to quit): ")
        if not q:
            break
        text, hits = answer(q)
        print("\n" + text)
        print("\nSources used:", [h["source"] for h in hits])
        
        
        
        
        
        
        