from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0,
    timeout=30,      # fail after 30s instead of hanging
    max_retries=2,   # retry twice, then give up
)
print(llm.invoke("Say hi").content)