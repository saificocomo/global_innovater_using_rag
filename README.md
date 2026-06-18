# UK Innovator Founder Visa — RAG Assistant

A retrieval-augmented generation (RAG) app that answers questions about the UK Innovator Founder visa using only the official gov.uk guidance, with inline source citations and a "don't know" guard against hallucination.

Built from scratch (no LangChain) so every part of the pipeline — chunking, embedding, retrieval, generation — is explicit and easy to reason about. The project includes an **evaluation harness** that measures answer quality, and the README documents a measured before/after improvement from tuning retrieval.

---

## Results

The app is evaluated against a hand-built "golden set" of 16 questions covering easy lookups, multi-part questions whose answers span multiple pages, paraphrased questions, and out-of-scope "trick" questions that should be refused.

Two metrics are tracked:

- **Correct** — does the answer match the known reference facts?
- **Faithful** — does the answer stay grounded in the retrieved source, without inventing anything?

| Change | Correct | Faithful |
|---|---|---|
| Baseline (retrieve top 4 chunks) | 81% (13/16) | 100% (16/16) |
| Retrieve top 8 chunks | 94% (15/16) | 100% (16/16) |
| Two-stage retrieval + cross-encoder re-ranker | 94% (15/16) | 100% (16/16) |

> Re-ranking did not move the headline score, but it changed the *kind* of failure on the remaining question for the better — see "The last question, and a note on measurement" below.

### Failure analysis

The baseline's three misses were not generation errors — the model answered correctly whenever it received the right text. They were **retrieval** problems:

- **Facts split across pages.** "How many years to settle for the applicant vs. dependants?" needs two facts from different sections; with only 4 chunks retrieved, one of them fell outside the window.
- **Topic ambiguity.** "Decision time inside the UK" collided with the more frequent "3 weeks" (outside-UK) phrasing repeated throughout the document.

Increasing retrieval depth from 4 to 8 chunks resolved both, lifting correctness to 94% while keeping faithfulness at 100%.

### The last question, and a note on measurement

One question — *"How much cash do I need to put into the business?"* — is still scored incorrect, but the story behind it is the most interesting result in the project.

It is a **semantic collision**: "cash into the business" (investment funds) embeds very close to the personal-savings requirement (£1,270), so single-vector search favours the savings passage. Increasing `k` did not help, because the problem is not retrieval depth — the two concepts simply look alike as embeddings.

Adding the cross-encoder re-ranker changed the *behaviour*, even though the score stayed the same:

- **Before re-ranking:** the app confidently answered with the wrong figure (the £1,270 savings amount).
- **After re-ranking:** the re-ranker correctly rejected the savings chunk as a poor match, and with no clearly-correct chunk to promote, the app honestly returned *"I don't know based on the official guidance."*

So the re-ranker traded a **confidently-wrong** answer for an **honest refusal** — which, for a compliance domain like immigration, is the safer failure. The eval still marks it incorrect because the answer doesn't match the reference, but this is arguably defensible: the source guidance does not state a fixed investment figure (funding is assessed case-by-case by the endorsing body), so "I don't know" is a reasonable response.

**A note on measurement:** the LLM-as-judge shows minor run-to-run variance — repeated runs occasionally flip a single borderline verdict. Single-point metric changes are therefore treated cautiously rather than over-interpreted; for a small eval set, the *direction* and the *qualitative inspection of failures* matter more than a one-question delta.

---

## Architecture

The app uses **separate models for separate jobs**, which keeps it cheap and provider-agnostic:

1. **Retrieval, stage 1 (local, free):** a `sentence-transformers` model (`all-MiniLM-L6-v2`) embeds text into vectors. A fast cosine-similarity search in NumPy retrieves a wide set of candidate chunks — no paid API, no external vector database.
2. **Retrieval, stage 2 — re-ranking (local, free):** a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores those candidates by looking at the question and each chunk *together*, and keeps only the best few. This resolves semantic collisions that single-vector search cannot.
3. **Generation (hosted API):** an LLM reads the final chunks and writes the answer. Because only this step touches the API, swapping the provider (OpenAI ↔ Anthropic ↔ local) is a one-function change.

```
Question
   │
   ▼
Embed question
   │
   ▼
Stage 1: vector search        →  ~20 candidate chunks   (fast, rough)
   │
   ▼
Stage 2: cross-encoder rerank →  best k chunks           (slow, precise)
   │
   ▼
Chunks + question  ──►  Generate grounded answer         (LLM API)
                              │
                              ▼
                        Answer + cited source
```

### How it works

**Ingestion (`ingest.py`, run once):** extracts text from the source PDF, splits it into overlapping chunks, embeds each chunk into a vector, and saves everything to `index.pkl`. The slow work happens once, so querying is instant.

**Query (`rag.py`):** embeds the user's question with the same model, retrieves a wide set of candidate chunks by cosine similarity, re-ranks them with a cross-encoder to surface the most relevant few, and passes those to the LLM with a system prompt that instructs it to answer **only** from the provided context, cite the source, and say "I don't know based on the official guidance" if the answer isn't present.

**Evaluation (`evals.py`):** runs the real app against the golden set and uses an LLM-as-judge to score each answer for correctness and faithfulness, printing an overall percentage. This is what makes "did my change help?" an answerable question.

---

## Setup

```bash
git clone <your-repo-url>
cd <repo-folder>

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Add your API key in a `.env` file in the project root:

```
OPENAI_API_KEY=your-key-here
```

Place the source PDF(s) in the `data/` folder.

## Usage

```bash
python ingest.py     # build the searchable index (run once)
python rag.py        # ask questions interactively
python evals.py      # measure quality against the golden set
```

Example:

```
Question: How much does it cost to apply from outside the UK?
It costs £1,357 per person if you apply outside the UK.
Sources used: ['Innovator_Founder_visa - GOV.UK.pdf']
```

---

## Project structure

```
.
├── data/             # source PDF(s)
├── ingest.py         # PDF → chunks → embeddings → index.pkl
├── rag.py            # retrieve + generate (interactive Q&A)
├── evals.py          # quality measurement (LLM-as-judge)
├── golden.json       # 16-question evaluation set with reference answers
├── requirements.txt
└── .env              # API key (gitignored, not committed)
```

---

## Future work

- **Heading-based chunking** — split on the document's numbered section headings so each chunk is one coherent topic, and measure against the current fixed-size chunks.
- **Show retrieved snippets** in the `Sources used` output, not just filenames, for better transparency.
- **Expand the golden set** to ~30 questions and add retrieval hit-rate as a separate metric.
- **Simple web UI** and deployment.

---

## Notes

Source content is UK government guidance, available under the Open Government Licence v3.0. This project is for demonstration and learning; it is not legal or immigration advice.
