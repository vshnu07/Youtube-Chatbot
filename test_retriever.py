from vector_store import create_vector_store
from retriever import retrieve_relevant_chunks

chunks = [
    "Machine Learning allows computers to learn from data.",
    "Deep Learning uses neural networks with multiple layers.",
    "Natural Language Processing helps computers understand human language.",
    "Transformers are the architecture behind modern Large Language Models."
]

vector_store = create_vector_store(chunks)

question = "What is Machine Learning?"

results = retrieve_relevant_chunks(
    vector_store,
    question,
    k=3
)

print("Question:", question)

print("\nRetrieved Relevant Chunks:")

for index, result in enumerate(results):
    print(f"\nChunk {index + 1}:")
    print(result.page_content)