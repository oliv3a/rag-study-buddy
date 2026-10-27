import gradio as gr
from rag import ask


def chat(message, history):
    answer, sources = ask(message)
    return f"{answer}\n\n📄 *Sources: {' | '.join(sources)}*"


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

demo.launch()