from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag import llm
from tracker import session_stats

COACH_PROMPT = PromptTemplate.from_template("""
You are a friendly study coach. These are a student's quiz results this session:
{results}

In 2-3 sentences: say what they did well, and which topic to review next.
Plain text only, no LaTeX.
""")

coach_chain = COACH_PROMPT | llm | StrOutputParser()


def summarize_session(session_id):
    stats = session_stats(session_id)
    if not stats:
        return "No questions answered this session."

    total = sum(n for _, n, _ in stats)
    right = sum(r for _, _, r in stats)
    lines = [f"  {t}: {r}/{n} ({100 * r // n}%)" for t, n, r in stats]
    results = "\n".join(lines)

    coach = coach_chain.invoke({"results": results})
    return (
        "\n===== 📊 Session Summary =====\n"
        f"Questions: {total} | Score: {right}/{total} ({100 * right // total}%)\n"
        f"{results}\n\n"
        f"🤖 Coach: {coach}\n"
        "=============================="
    )
