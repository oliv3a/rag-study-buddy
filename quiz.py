from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from rag import retriever, llm, format_docs
from tracker import new_session_id, log_attempt, topic_stats

# --- The "shapes" Gemini must reply in ---
class QuizQuestion(BaseModel):
    question: str = Field(description="One short exam-style question")
    answer: str = Field(description="The correct answer in 1-2 sentences")


class Grade(BaseModel):
    correct: bool = Field(description="True if the student's answer has the right idea")
    feedback: str = Field(description="1-2 sentences explaining why")


QUESTION_PROMPT = PromptTemplate.from_template("""
You are an exam tutor. Using ONLY the notes below, write ONE short question
that tests understanding, not just memorising a single word.

Notes:
{context}
""")

GRADE_PROMPT = PromptTemplate.from_template("""
Grade the student's answer fairly. Accept answers with the right idea
even if worded differently. Use plain text, no LaTeX.

Question: {question}
Correct answer: {answer}
Student's answer: {student_answer}
""")

question_chain = QUESTION_PROMPT | llm.with_structured_output(QuizQuestion)
grade_chain = GRADE_PROMPT | llm.with_structured_output(Grade)


# --- Functions ---
def generate_question(topic):
    docs = retriever.invoke(topic)
    q = question_chain.invoke({"context": format_docs(docs)})
    real_topic = docs[0].metadata.get("topic", "general")  # from the PDF, not the AI
    return q, docs, real_topic


def grade_answer(q, student_answer):
    return grade_chain.invoke({
        "question": q.question,
        "answer": q.answer,
        "student_answer": student_answer,
    })


if __name__ == "__main__":
    session_id = new_session_id()
    print("--- Quiz Mode --- (type 'stats' for progress, 'exit' to quit)")
    while True:
        try:
            topic = input("\nTopic (e.g. quicksort, dijkstra): ").strip()
            if topic.lower() in ["exit", "quit"]:
                break
            if not topic:
                continue
            if topic.lower() == "stats":
                for t, total, right in topic_stats():
                    print(f"  {t}: {right}/{total} ({100 * right // total}%)")
                continue

            q, docs, real_topic = generate_question(topic)
            print(f"\n❓ [{real_topic}] {q.question}")
            student = input("Your answer: ").strip()

            g = grade_answer(q, student)
            log_attempt(session_id, real_topic, q.question, g.correct)  
            print("✅ Correct!" if g.correct else "❌ Not quite.")
            print(f"Feedback: {g.feedback}")
            print(f"Model answer: {q.answer}")
            pages = sorted({f"p.{d.metadata['page']}" for d in docs})
            print(f"Review: {', '.join(pages)}")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Something went wrong: {e}")