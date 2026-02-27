from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import BaseModel
from typing import List

persistent_directory = "db/chroma_db"
embedding_model = OllamaEmbeddings(model = "nomic-embed-text")
llm = ChatOllama(model = "llama3.2:3b", temperature=0)

db = Chroma(
    persist_directory = persistent_directory,
    embedding_function = embedding_model,
    collection_metadata = {"hnsw:space":"cosine"}
)

#Pydantic model for structured output
class QueryVariations(BaseModel):
    queries: List[str]

# Original query
original_query = "What was NVIDIA's first graphics accelerator called?"
print(f"Original Query: {original_query}\n")

# Generate multiple query variations

llm_with_tools = llm.with_structured_output(QueryVariations)

prompt = f"""Generate 3 different variations of this query that would help retrieve relevant documents:

Original query: {original_query}

Return 3 alternative queries that rephrase or approach the same question from different angles."""

response = llm_with_tools.invoke(prompt)
query_variations = response.queries

print("Generate Query Variations: ")
for i, variation in enumerate(query_variations, 1):
    print(f"{i}. {variation}")

print("\n"+"="*60)

# Step 2: Search with each query variation & Store Results

retriever = db.as_retriever(search_kwargs={"k": 4})
all_retrieval_results = []

for i,query in enumerate(query_variations, 1):
    print(f"\n=== RESULTS FOR QUERY {i}: {query} ===")

    docs = retriever.invoke(query)
    all_retrieval_results.append(docs)

    print(f"Retrieved {len(docs)} documents:\n")

    for j, doc in enumerate(docs, 1):
        print(f"Document {j}:")
        print(f"{doc.page_content[:150]}...\n")

    print("-"*50)

print("\n"+"="*50)
print("Multi-Query Retrieval Complete!")
