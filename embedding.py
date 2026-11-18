
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [
    "Machine Learning allows computers to learn from data.",
    "Deep Learning uses neural networks.",
    "Python is a programming language."
]

embeddings = model.encode(texts)

print("Embeddings:")
print(embeddings)

print("\nEmbedding Shape:")
print(embeddings.shape)
