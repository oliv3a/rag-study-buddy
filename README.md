# 📚 RAG Study Buddy

Chat with your lecture notes, get quizzed on them, and see which topics you're weakest at. Every answer is grounded in your own PDFs, with page citations.

**🔗 Live demo:** https://rag-study-buddy-fpoc.onrender.com/ 
*(Free tier: the first load may take ~1 min while the server wakes up.)*

<!-- Add a demo GIF here: chat question → quiz → stats -->

## Features

- **💬 Chat**: ask questions about your notes. Answers cite source pages, and the bot says "not covered" instead of making things up.
- **📝 Quiz**: generates exam-style questions from random parts of your notes, grades free-text answers, and shows which pages to review.
- **🎯 Weak-topic practice**: leave the topic blank and it quizzes you on your lowest-accuracy topic.
- **📊 Stats & session summary**: accuracy per topic, plus an end-of-session recap.

## How it works

```
PDFs ──► ingest.py ──► split into chunks ──► Gemini embeddings ──► Chroma vector DB
                                                                         │
Question ──► retrieve top 5 chunks (rag.py) ─────────────────────────────┘
                     │
                     ▼
         Gemini answers using ONLY those chunks ──► answer + page sources
```

- Each page is tagged with a **topic** (from a `Topic:` line in the PDF, or the file name), which powers quiz targeting and weak-topic tracking.
- Quiz attempts are logged to **SQLite** (`tracker.py`).
- The vector DB is **built automatically on startup** if it doesn't exist, so the app deploys without committing generated data.

**Stack:** Python · LangChain · Gemini (LLM + embeddings) · Chroma · Gradio · SQLite · Render

## Evaluation

Two small eval scripts, run after every change to catch regressions:

| Eval | What it checks | Result |
|---|---|---|
| `eval.py` | Does retrieval fetch the right page? (incl. paraphrased questions) | **13/13** |
| `eval_answers.py` | Does it answer in-scope questions and refuse out-of-scope ones? | **10/10** |

Manual review is still needed: the answer eval only checks *answered vs refused*. Reading the outputs caught one factually inverted answer that the score marked as a pass.

## Engineering notes

Problems found while testing on real lecture slides, and how they were fixed:

1. **Multi-part questions were refused.** Debugged by printing the retrieved chunks. Raised `k` from 2 to 5, and changed the prompt from all-or-nothing refusal to allowing partial answers, while confirming out-of-scope questions are still refused.
2. **Every page was tagged "general".** Topic detection relied on a `Topic:` line that real slides don't have. Now it falls back to the file name.
3. **The quiz kept asking the same question.** Similarity search always returned the same top chunks. It now seeds from a random chunk within the topic.
4. **Double-clicking Submit logged an answer twice.** Each question now locks after one submission.
5. **Weak-topic mode picked topics not in the notes.** Stale stats (and a self-test writing fake data into the real DB) leaked in. Stats are now filtered to current topics.
6. **Module name hardcoded in several places.** Moved to config (`MODULE` env var).

## Run locally

```bash
git clone https://github.com/oliv3a/rag-study-buddy.git
cd rag-study-buddy
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:
```
GOOGLE_API_KEY=your_key_here
MODULE=SC2005
DOCS_DIR=demo_docs
```

Then run `python app.py` and open http://localhost:7860.

To use your own notes, put PDFs in a folder (e.g. `docs/`), set `DOCS_DIR=docs`, and run `python ingest.py`.

## Limitations

- Topics are per-PDF or per-`Topic:` line. Auto-tagging sub-topics with an LLM is a planned upgrade.
- On the hosted demo, quiz stats are shared by all visitors and reset when the server restarts.
- Text extracted from slide tables and diagrams can be garbled, which limits answers on those slides.
- Runs on the Gemini free tier, so heavy use can hit the API rate limit. The app shows a "busy, try again" message instead of crashing.

## Roadmap

- Upload your own PDF in the browser
- Telegram bot front-end (via n8n)
- Per-user stats
