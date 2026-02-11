import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_core.documents import Document
from pathlib import Path
from langchain_ollama import OllamaEmbeddings


load_dotenv()

def load_documents(docs_path="docs"):
    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist.")

    documents = []

    for file_path in Path(docs_path).glob("*.txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        documents.append(
            Document(
                page_content=text,
                metadata={"source": str(file_path)}
            )
        )

    if not documents:
        raise FileNotFoundError(f"No .txt files found in {docs_path}.")

    # Debug preview
    # for i, doc in enumerate(documents[:2]):
    #     print(f"\nDocument {i+1}:")
    #     print(f" Source: {doc.metadata['source']}")
    #     print(f" Content length: {len(doc.page_content)} characters")
    #     print(f" Content preview: {doc.page_content[:100]}...")
    #     print(f" Metadata: {doc.metadata}")

    return documents

def split_documents(documents, chunk_size=800, chunk_overlap=100):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:
        # for i,chunk in enumerate(chunks[:5]):
        #     print(f"\nChunk {i+1}:")
        #     print(f" Source: {chunk.metadata['source']}")
        #     print(f" Content length: {len(chunk.page_content)} characters")
        #     print(f" Content :")
        #     print(chunk.page_content)
        #     print(f"-"*50) 

        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")  

    return chunks   

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")

    embedding_model = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents = chunks,
        embedding = embedding_model,
        persist_directory = persist_directory,
        collection_metadata = {"hnsw:space": "cosine"}
    )  
    print("--- Finished creating vector store ---")

    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore


def main():
    documents = load_documents(docs_path="docs")

    chunks = split_documents(documents)

    vectorstore = create_vector_store(chunks)

if __name__ == "__main__":
    main()


