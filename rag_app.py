import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from the .env file
load_dotenv()

print("Loading PDF document...")
loader = PyPDFLoader("TechCorp_Official_Employee_Handbook.pdf")
document = loader.load()

print(document[0].page_content)

print("Chunking text...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(document)
print("chunk1")
print(chunks[0].page_content)
print("chunk2")
print(chunks[1].page_content)

# convert chunks into vector coordinates
print("Creating vector database...")

#sets up a translator that turns text into a list of numbers
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001") 

#Chroma is a vector database
vector_db = Chroma.from_documents(
    documents=chunks, #your text, already split into small pieces
    embedding=embeddings, #which translator to use
    persist_directory="./chroma_db" #saves the database to a folder on disk, so you don’t have to re-embed everything
)

# Configure the database to act as a document retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 2})
# converts the vector database into a retriever object that can search through stored document embeddings and return the most relevant chunks for a user's question 
# setting k = 2 : tells the database to only pull the top two most relevant chunks for any given question
# this helps to keep things clean and efficient 




# Define the hidden prompt structure for the LLM
template = """
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Context: {context}

Question: {question}

Answer:
"""
prompt = PromptTemplate.from_template(template)

# Initializing the LLM and Constructing the RAG Chain 

# Initialize the free Gemini model tier
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0,
    timeout=30,      # fail after 30s instead of hanging
    max_retries=2,   # retry twice, then give up
)
#temperature=0 makes answers consistent and factual (less “creative”)
#ask same qn , same output ans
# Helper function to stitch retrieved chunks into a single text block
# Joins the retrieved chunks into one block of text
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Connect everything together using LangChain Expression Language (LCEL)
rag_chain = (
    {
        # question → retriever gets chunks → format_docs turns them into one text block
        "context": retriever | format_docs,
        # keeps the question as is
        "question": RunnablePassthrough(),
    }
    | prompt   # fills {context} and {question} into the template
    | llm      # Gemini writes the answer
    | StrOutputParser()   # pulls out just the answer text
)

#| simply means “take what came out of the last step and feed it into the next one


user_question = "What days can I work from home?"
print(f"\nQuestion: {user_question}")

response = rag_chain.invoke(user_question)
print(f"Answer: {response}")


# conversation loop 
# Chat with your PDF in a continuous loop
print("\n--- PDF Chatbot Initialized ---")
print("Type 'exit' or 'quit' to stop.")

while True:
    # 1. Wait for the user to type a question
    user_question = input("\nYour Question: ").strip()

    # 2. Skip empty input (e.g. user just pressed Enter)
    if not user_question:
        continue

    # 3. Allow the user to exit
    if user_question.lower() in ["exit", "quit"]:
        print("Shutting down chatbot. Goodbye!")
        break

    # 4. Send the question through the RAG chain
    #    (StrOutputParser in the chain already returns plain text)
    try:
        response = rag_chain.invoke(user_question)
        print(f"Answer: {response}")
    except Exception as e:
        print(f"Something went wrong: {e}")

## test case ##
# 1. Did the chunks get stored?
print("Chunks stored:", vector_db._collection.count())

# 2. Does the embedding model work?
test_vec = embeddings.embed_query("hello world")
print("Vector length:", len(test_vec))   # gemini-embedding-001 → 3072
print("First 5 numbers:", test_vec[:5])

# 3. Does search return relevant chunks?
results = vector_db.similarity_search_with_score("YOUR QUESTION HERE", k=3)
for doc, score in results:
    print(f"\nScore: {score:.4f}")   # lower = closer match
    print(doc.page_content[:200])