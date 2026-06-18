# UK Innovator Founder Visa — RAG Assistant

A small Retrieval-Augmented Generation (RAG) app that answers questions about the
**UK Innovator Founder visa** using only the official GOV.UK guidance as its source.

It embeds the guidance locally with [`sentence-transformers`](https://www.sbert.net/),
retrieves the most relevant chunks for a question, and asks an OpenAI chat model to
answer **strictly from the retrieved context** — citing the source file and refusing
to answer when the context doesn't cover it.

## How it works

1. **`ingest.py`** — reads the documents in `data/`, splits them into overlapping
   chunks, embeds them with `all-MiniLM-L6-v2`, and saves the vectors to `index.pkl`.
2. **`rag.py`** — embeds a question, retrieves the top matching chunks by cosine
   similarity, and generates a grounded answer with `gpt-4o-mini`.
3. **`evals.py`** — runs the questions in `golden.json` through the pipeline and uses
   an LLM judge to score answers for correctness and faithfulness.

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your OpenAI API key
cp .env.example .env
# then edit .env and paste your real key
```

## Usage

```bash
# Build the search index from the documents in data/
python ingest.py

# Ask questions interactively
python rag.py

# Run the evaluation suite
python evals.py
```

## Project structure

```
.
├── data/            # Source documents (official GOV.UK guidance)
├── ingest.py        # Build the embedding index -> index.pkl
├── rag.py           # Retrieve + answer
├── evals.py         # LLM-as-judge evaluation
├── golden.json      # Q&A reference set for evals
├── requirements.txt
└── .env.example     # Template for your OPENAI_API_KEY
```

## Notes

- `index.pkl` is a generated artifact and is git-ignored — run `python ingest.py` to
  create it.
- Your API key lives only in `.env`, which is git-ignored and never committed.
