from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

persistent_directory = "db/chroma_db"

embedding_model = OllamaEmbeddings(model = "nomic-embed-text")

db = Chroma(
    persist_directory = persistent_directory,
    embedding_function = embedding_model,
    collection_metadata = {"hnsw:space": "cosine"}
)
# set the query and retriever before running debug checks
query = "What was NVIDIA's first graphics accelerator called?"

retriever = db.as_retriever(search_kwargs={"k": 3})
relevant_docs = retriever.invoke(query)


# Use the retriever's proper retrieval API. Different langchain/adapter
# versions expose different methods, so try common ones and fall back
# to a direct similarity search on the DB.
# try:
#     relevant_docs = retriever.get_relevant_documents(query)
# except Exception:
#     try:
#         relevant_docs = retriever.retrieve(query)
#     except Exception:
#         relevant_docs = db.similarity_search(query, k=5)

print(f"User query: {query}")

print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}: \n{doc.page_content}\n")



# Synthetic Questions: 

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"

if not relevant_docs:
    print("No relevant documents found.")
    exit()

combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

model = ChatOllama(
    model = "llama3.2:3b",
    temperature=0
)

messages = [
    SystemMessage(content="You are a helpful assistant strictly using the provided documents. Do not use outside knowledge."),
    HumanMessage(content=combined_input)
]

result = model.invoke(messages)

print("\n--- Generated Response ---")

print("Content only:")
print(result.content)