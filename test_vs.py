from vector_store import create_vector_store

chunks = [
    "Machine Learning allows computers to learn from data.",
    "Deep Learning uses neural networks with multiple layers.",
    "Natural Language Processing helps computers understand human language."
]

vector_store = create_vector_store(chunks)

print("FAISS Vector Store Created Successfully!")

results = vector_store.similarity_search(
    "What is Machine Learning?",
    k=2
)

for result in results:
    print("\nRetrieved Chunk:")
    print(result.page_content)