
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_transcript(transcript_text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(transcript_text)

    return chunks
