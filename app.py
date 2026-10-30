import os
from dotenv import load_dotenv
load_dotenv()
if not os.path.exists("chroma_db"):
    import ingest
    ingest.main()

MODULE = os.getenv("MODULE", "SC2005")
import gradio as gr
from rag import ask
from quiz import generate_question, grade_answer, ALL_TOPICS
from tracker import new_session_id, log_attempt, weakest_topic
import pandas as pd
from tracker import topic_stats
from summary import summarize_session


def chat(message, history):
    answer, sources = ask(message)
    if "NOT_IN_NOTES" in answer:
        return f"🤷 That's not covered in your notes. Try asking about one of your {MODULE} topics."
    return f"{answer}\n\n📄 *Sources: {' | '.join(sources)}*"


def new_question(topic, state):
    typed = topic.strip()          # what the user actually typed
    topic = typed
    if not topic or topic.lower() == "weak":
        topic = weakest_topic(ALL_TOPICS)
    q, docs, real_topic = generate_question(topic)
    state = {
        "q": q, "docs": docs, "topic": real_topic,
        "session": state.get("session") or new_session_id(),  # keep the same session
    }
    note = ""
    # only warn if the user typed a topic that isn't in the notes
    if typed and typed.lower() != "weak" and not any(typed.lower() in t for t in ALL_TOPICS):
        note = f"⚠️ *'{typed}' isn't a topic in your notes. Closest match: **{real_topic}***\n\n"
    return f"{note}### ❓ [{real_topic}] {q.question}", "", "", state

def submit_answer(answer, state):
    if "q" not in state:
        return "👆 Click **New question** first!", state
    if state.get("answered"):
        return "✋ Already submitted. Click **🎲 New question** for another one.", state
    q = state["q"]
    g = grade_answer(q, answer)
    log_attempt(state["session"], state["topic"], q.question, g.correct)
    state["answered"] = True   # lock this question
    pages = ", ".join(sorted({f"p.{d.metadata['page']}" for d in state["docs"]}))
    verdict = "✅ **Correct!**" if g.correct else "❌ **Not quite.**"
    return (f"{verdict}\n\n**Feedback:** {g.feedback}\n\n"
            f"**Model answer:** {q.answer}\n\n📄 Review: {pages}"), state
def load_stats():
    df = pd.DataFrame(topic_stats(), columns=["topic", "attempts", "correct"])
    df["accuracy"] = (100 * df["correct"] / df["attempts"]).round()
    return df, df


def end_session(state):
    if "session" not in state:
        return "Answer some quiz questions first!"
    return summarize_session(state["session"])


with gr.Blocks(title=f"{MODULE} Study Buddy") as demo:
    gr.Markdown(f"# 📚 {MODULE} Study Buddy")
    with gr.Tab("💬 Chat"):
        gr.ChatInterface(
            chat,
            chatbot=gr.Chatbot(latex_delimiters=[
                {"left": "$$", "right": "$$", "display": True},
                {"left": "$", "right": "$", "display": False},
            ]),
        )
    with gr.Tab("📝 Quiz"):
        quiz_state = gr.State({})
        topic_box = gr.Textbox(label="Topic", placeholder=f"e.g. {ALL_TOPICS[0]} (leave empty to practise your weakest topic in {MODULE})")
        new_btn = gr.Button("🎲 New question")
        question_md = gr.Markdown()
        answer_box = gr.Textbox(label="Your answer", lines=3)
        submit_btn = gr.Button("Submit answer", variant="primary")
        result_md = gr.Markdown()

        new_btn.click(new_question, [topic_box, quiz_state],
                      [question_md, answer_box, result_md, quiz_state])
        submit_btn.click(submit_answer, [answer_box, quiz_state],
                         [result_md, quiz_state])

    with gr.Tab("📊 Stats"):
        refresh_btn = gr.Button("🔄 Refresh stats")
        chart = gr.BarPlot(x="topic", y="accuracy", title="Accuracy by topic (%)", y_lim=[0, 100])
        table = gr.Dataframe()
        summary_btn = gr.Button("🏁 End session & summarise", variant="primary")
        summary_md = gr.Markdown()

        refresh_btn.click(load_stats, None, [chart, table])
        summary_btn.click(end_session, quiz_state, summary_md)

        
demo.launch()