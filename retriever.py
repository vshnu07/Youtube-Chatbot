
def retrieve_relevant_chunks(vector_store, question, k=5):

    results = vector_store.similarity_search_with_score(
        query=question,
        k=k
    )

    print("\n========== RETRIEVAL RESULTS ==========\n")

    documents = []

    for i, (document, score) in enumerate(results):

        print(f"Chunk {i + 1}")
        print(f"Similarity Distance: {score}")
        print(f"Content: {document.page_content}")
        print("-" * 60)

        documents.append(document)

    return documents
