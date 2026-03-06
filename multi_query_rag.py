from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import BaseModel
from typing import List
from collections import defaultdict

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

# Step 3: Apply RRF

def reciprocal_rank_fusion(chunks_list, k = 60, verbose = True):
    if verbose:
        print("\n" + "="*60)
        print("Applying RRF")
        print("="*60)
        print("Calculating RRF scores...\n")

    rrf_scores = defaultdict(float)
    all_unique_chunks = {}
    chunk_id_map = {}
    chunk_counter = 1
    for query_idx, chunks in enumerate(chunks_list,1):
        if verbose:
            print(f"Processing Query {query_idx} results:")

        for position, chunk in enumerate(chunks, 1):
            chunk_content = chunk.page_content

            if chunk_content not in chunk_id_map:
                chunk_id_map[chunk_content] = f"Chunk_{chunk_counter}"
                chunk_counter += 1

            chunk_id = chunk_id_map[chunk_content]
            all_unique_chunks[chunk_content] = chunk
            position_score = 1/(k+position)

            rrf_scores[chunk_content] += position_score

            if verbose:
                print(f" Position {position}: {chunk_id}+{position_score:.4f} (running total: {rrf_scores[chunk_content]:.4f})")
                print(f" Preview: {chunk_content[:80]}...")

        if verbose:
            print()
    
    sorted_chunks = sorted(
        [(all_unique_chunks[chunk_content], score) for chunk_content, score in rrf_scores.items()],
        key = lambda x:x[1],
        reverse = True
    )

    if verbose:
        print(f"✅RRF complete! Processes {len(sorted_chunks)} unique chunks from {len(chunks_list)} queries.")
    return sorted_chunks

fused_results = reciprocal_rank_fusion(all_retrieval_results, k = 60, verbose = True)

# Step 4: Display final fused results

print("\n"+"="*60)
print("Final RRF Ranking")
print("="*60)

print(f"\nTop {min(10, len(fused_results))} documents after RRF fusion:\n")

for rank, (doc, rrf_score) in enumerate(fused_results[:10], 1):
    print(f"RANK {rank} (RRF Score: {rrf_score:.4f})")
    print(f"{doc.page_content[:200]}...")
    print("-"*50)

print(f"\n✅ RRF Complete! Fused {len(fused_results)} unique documents from {len(query_variations)} query variations.")