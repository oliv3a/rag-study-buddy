from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

DB_DIR = "chroma_db"

PROMPT = PromptTemplate.from_template("""
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Context: {context}

Question: {question}

Answer:
""")

# Open the SAVED database (no re-embedding!)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", timeout=30, max_retries=2)
chain = PROMPT | llm | StrOutputParser()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def ask(question):
    """Return (answer, sources) for a question."""
    docs = retriever.invoke(question)
    answer = chain.invoke({"context": format_docs(docs), "question": question})
    sources = sorted({f"{d.metadata['source']}, p.{d.metadata['page']}" for d in docs})
    return answer, sources


if __name__ == "__main__":
    print("--- Study Buddy --- (type 'exit' to quit)")
    while True:
        try:
            q = input("\nYour Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        if not q:
            continue
        if q.lower() in ["exit", "quit"]:
            break
        try:
            answer, sources = ask(q)
            print(f"Answer: {answer}")
            print(f"Sources: {', '.join(sources)}")
        except Exception as e:
            print(f"Something went wrong: {e}")