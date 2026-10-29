import re
import os
import shutil
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DOCS_DIR = "docs"       # where your PDFs live
DB_DIR = "chroma_db"    # where the vector database is saved


def load_pdfs(folder):
    """Read every PDF in the folder, one Document per page."""
    docs = []
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".pdf"):
            continue
        reader = PdfReader(os.path.join(folder, filename))
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                match = re.search(r"Topic:\s*(\S+)", text)
                topic = match.group(1) if match else os.path.splitext(filename)[0].lower()
                docs.append(Document(
                    page_content=text,
                    metadata={"source": filename, "page": page_num, "topic": topic},
                ))
    return docs


def main():
    # Delete the old database so we never store duplicates
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

    docs = load_pdfs(DOCS_DIR)
    print(f"Loaded {len(docs)} pages")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)
    print(f"Saved to {DB_DIR}/")


if __name__ == "__main__":
    main()