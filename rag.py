from dotenv import load_dotenv
load_dotenv()

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer , CrossEncoder
from openai import OpenAI

model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
client = OpenAI()                          # reads OPENAI_API_KEY from env

index = pickle.load(open("index.pkl", "rb"))
CHUNKS, EMB = index["chunks"], index["embeddings"]

def retrieve(question, k=4, candidates=20):
    # STAGE 1 — quick skim: fast vector search, grab a wide net of candidates
    q = model.encode([question])[0]
    sims = EMB @ q / (np.linalg.norm(EMB, axis=1) * np.linalg.norm(q))
    top_candidates = np.argsort(sims)[::-1][:candidates]
    shortlist = [CHUNKS[i] for i in top_candidates]

    # STAGE 2 — careful read: re-rank the shortlist by scoring (question, chunk) together
    pairs = [[question, c["text"]] for c in shortlist]
    scores = reranker.predict(pairs)
    reranked = [c for _, c in sorted(zip(scores, shortlist), key=lambda x: x[0], reverse=True)]

    return reranked[:k]   # keep only the best k after the careful read

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
        
        
        
        
        
        
        