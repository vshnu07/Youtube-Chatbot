
from langchain_ollama import ChatOllama
from retriever import retrieve_relevant_chunks


llm = ChatOllama(
    model="llama3.2:latest",
    temperature=0
)


def generate_answer(vector_store, question):

    results = retrieve_relevant_chunks(
        vector_store,
        question,
        k=5
    )

    print("\n========== RETRIEVED CONTEXT ==========\n")

    context = "\n\n".join(
        document.page_content for document in results
    )

    print(context)

    print("\n========== USER QUESTION ==========\n")
    print(question)

    prompt = f"""
You are an intelligent YouTube video assistant.

Your task is to answer questions using the provided video transcript.

STRICT RULES:

1. Use only the information available in the transcript.
2. Never invent facts or add information from your general knowledge.
3. Understand the transcript even if it contains Telugu, English,
   or a mixture of both languages.
4. Always respond in English, regardless of the transcript language.
5. Translate Telugu information into natural English when necessary.
6. Give clear, accurate and well-structured answers.
7. Include important details and examples when they are present
   in the transcript.
8. If the transcript does not contain enough information to answer,
   clearly say:
   "The video transcript does not contain enough information
   to answer this question."
9. Do not confuse suggestions mentioned in the transcript
   with facts or completed actions.

VIDEO TRANSCRIPT:
{context}

USER QUESTION:
{question}

ANSWER IN ENGLISH:
"""

    response = llm.invoke(prompt)

    return response.content
