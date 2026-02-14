from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

persistent_directory = "db/chroma_db"

embedding_model = OllamaEmbeddings(model = "nomic-embed-text")

db = Chroma(
    persist_directory = persistent_directory,
    embedding_function = embedding_model,
    collection_metadata = {"hnsw:space": "cosine"}
)

model = ChatOllama(
    model = "llama3.2:3b",
    temperature=0
)
# set the query and retriever before running debug checks
# query = "What was NVIDIA's first graphics accelerator called?"

chat_history = []

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")

    if chat_history:
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content = f"New question: {user_question}")
        ]

        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

    retriever = db.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(search_question)


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

    print(f"User query: {search_question}")

    print("--- Context ---")
    for i, doc in enumerate(relevant_docs, 1):
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"Document {i}: \n{preview}...\n")



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

    combined_input = f"""Based on the following documents, please answer this question: {search_question}

    Documents:
    {chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """



    messages = [
        SystemMessage(content="You are a helpful assistant strictly using the provided documents. Do not use outside knowledge."),
        HumanMessage(content=combined_input)
    ]

    result = model.invoke(messages)
    answer = result.content
    
    chat_history.append(HumanMessage(content = user_question))
    chat_history.append(AIMessage(content=answer))

    print("\n--- Generated Response ---")
    print("Content only:")
    print(answer)
    return answer

def start_chat():
    print("Ask me questions! Type quit to exit.")

    while True:
        question = input("\nYour question: ")

        if question.lower() == 'quit':
            print("Goodbye!")
            break

        ask_question(question)

if __name__ == "__main__":
    start_chat()
