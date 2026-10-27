import gradio as gr
from rag import ask
from quiz import generate_question, grade_answer, ALL_TOPICS
from tracker import new_session_id, log_attempt, weakest_topic


def chat(message, history):
    answer, sources = ask(message)
    return f"{answer}\n\n📄 *Sources: {' | '.join(sources)}*"



def new_question(topic, state):
    topic = topic.strip()
    if not topic or topic.lower() == "weak":
        topic = weakest_topic(ALL_TOPICS)
    q, docs, real_topic = generate_question(topic)
    state = {
        "q": q, "docs": docs, "topic": real_topic,
        "session": state.get("session") or new_session_id(),  # keep the same session
    }
    return f"### ❓ [{real_topic}] {q.question}", "", "", state


def submit_answer(answer, state):
    if "q" not in state:
        return "👆 Click **New question** first!", state
    q = state["q"]
    g = grade_answer(q, answer)
    log_attempt(state["session"], state["topic"], q.question, g.correct)
    pages = ", ".join(sorted({f"p.{d.metadata['page']}" for d in state["docs"]}))
    verdict = "✅ **Correct!**" if g.correct else "❌ **Not quite.**"
    return (f"{verdict}\n\n**Feedback:** {g.feedback}\n\n"
            f"**Model answer:** {q.answer}\n\n📄 Review: {pages}"), state



with gr.Blocks(title="SC2001 Study Buddy") as demo:
    gr.Markdown("# 📚 SC2001 Study Buddy")
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
        topic_box = gr.Textbox(label="Topic", placeholder="e.g. quicksort (leave empty to practise your weakest topic)")
        new_btn = gr.Button("🎲 New question")
        question_md = gr.Markdown()
        answer_box = gr.Textbox(label="Your answer", lines=3)
        submit_btn = gr.Button("Submit answer", variant="primary")
        result_md = gr.Markdown()

        new_btn.click(new_question, [topic_box, quiz_state],
                      [question_md, answer_box, result_md, quiz_state])
        submit_btn.click(submit_answer, [answer_box, quiz_state],
                         [result_md, quiz_state])

        
demo.launch()