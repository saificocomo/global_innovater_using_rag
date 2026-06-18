import os, glob, pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

def chunk_text(text, source):
    chunks, start = [], 0
    while start < len(text):
        chunks.append({"source": source, "text": text[start:start + CHUNK_SIZE]})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def read_pdf(path):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    all_chunks = []
    for path in glob.glob("data/*.txt") + glob.glob("data/*.pdf"):
        text = read_pdf(path) if path.endswith(".pdf") else open(path, encoding="utf-8").read()
        all_chunks += chunk_text(text, os.path.basename(path))
    print(f"Created {len(all_chunks)} chunks")

    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    with open("index.pkl", "wb") as f:
        pickle.dump({"chunks": all_chunks, "embeddings": np.array(embeddings)}, f)
    print("Saved index.pkl")

if __name__ == "__main__":
    main()