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
| Retrieve top 8 chunks | **94% (15/16)** | 100% (16/16) |

### Failure analysis

The baseline's three misses were not generation errors — the model answered correctly whenever it received the right text. They were **retrieval** problems:

- **Facts split across pages.** "How many years to settle for the applicant vs. dependants?" needs two facts from different sections; with only 4 chunks retrieved, one of them fell outside the window.
- **Topic ambiguity.** "Decision time inside the UK" collided with the more frequent "3 weeks" (outside-UK) phrasing repeated throughout the document.

Increasing retrieval depth from 4 to 8 chunks resolved both, lifting correctness to 94% while keeping faithfulness at 100%.

### Known limitation

One question still fails: *"How much cash do I need to put into the business?"* This is a **semantic collision** — "cash into the business" (investment funds) is embedded very close to the personal-savings requirement (£1,270), so retrieval favours the savings passage. Increasing `k` did not fix it, which indicates the issue is at the embedding level rather than retrieval depth. Documented here rather than over-fitting the prompt to a single eval question. See *Future work*.

---

## Architecture

The app uses **two different models for two different jobs**, which keeps it cheap and provider-agnostic:

1. **Retrieval (local, free):** a `sentence-transformers` model (`all-MiniLM-L6-v2`) embeds text into vectors. Searching is done with cosine similarity in NumPy — no paid API, no external vector database.
2. **Generation (hosted API):** an LLM reads the retrieved chunks and writes the final answer. Because only this step touches the API, swapping the provider (OpenAI ↔ Anthropic ↔ local) is a one-function change.

```
Question
   │
   ▼
Embed question  ──►  Retrieve top-k chunks   (local: MiniLM + cosine similarity)
                          │
                          ▼
                 Chunks + question  ──►  Generate grounded answer   (LLM API)
                                              │
                                              ▼
                                        Answer + cited source
```

### How it works

**Ingestion (`ingest.py`, run once):** extracts text from the source PDF, splits it into overlapping chunks, embeds each chunk into a vector, and saves everything to `index.pkl`. The slow work happens once, so querying is instant.

**Query (`rag.py`):** embeds the user's question with the same model, ranks all chunks by cosine similarity, takes the top `k`, and passes them to the LLM with a system prompt that instructs it to answer **only** from the provided context, cite the source, and say "I don't know based on the official guidance" if the answer isn't present.

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

- **Heading-based chunking** — split on the document's numbered section headings so each chunk is one coherent topic, which may resolve the investment-funds vs. personal-savings collision.
- **Show retrieved snippets** in the `Sources used` output, not just filenames, for better transparency.
- **Expand the golden set** to ~30 questions and add retrieval hit-rate as a separate metric.
- **Simple web UI** and deployment.

---

## Notes

Source content is UK government guidance, available under the Open Government Licence v3.0. This project is for demonstration and learning; it is not legal or immigration advice.
