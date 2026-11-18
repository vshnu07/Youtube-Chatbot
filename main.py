
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi

from rag_pipeline import split_transcript
from vector_store import create_vector_store
from qa_chain import generate_answer

from fastapi.middleware.cors import CORSMiddleware



app = FastAPI(
    title="YouTube AI Companion",
    description="AI assistant to interact with YouTube videos",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.youtube.com"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

video_vector_stores = {}


class VideoRequest(BaseModel):
    video_id: str


class ChatRequest(BaseModel):
    video_id: str
    question: str


@app.get("/")
def home():
    return {
        "message": "YouTube AI Companion Backend is running!"
    }


@app.post("/transcript")
def get_transcript(request: VideoRequest):

    try:
        api = YouTubeTranscriptApi()

        transcript_list = api.list(request.video_id)

        transcript = next(iter(transcript_list))

        fetched_transcript = transcript.fetch()

        transcript_text = " ".join(
            snippet.text for snippet in fetched_transcript
        )

        if not transcript_text.strip():
            raise HTTPException(
                status_code=404,
                detail="Transcript is empty."
            )

        chunks = split_transcript(transcript_text)

        vector_store = create_vector_store(chunks)

        video_vector_stores[request.video_id] = vector_store

        return {
            "video_id": request.video_id,
            "language": transcript.language,
            "total_chunks": len(chunks),
            "vector_store_created": True,
            "message": "Transcript processed and vector store created successfully."
        }

    except StopIteration:
        raise HTTPException(
            status_code=404,
            detail="No transcript available for this video."
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/ask")
def ask_question(request: ChatRequest):

    vector_store = video_vector_stores.get(request.video_id)

    if vector_store is None:
        raise HTTPException(
            status_code=404,
            detail="Video transcript not found. Fetch the transcript first."
        )

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        answer = generate_answer(
            vector_store,
            request.question
        )

        return {
            "video_id": request.video_id,
            "question": request.question,
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
