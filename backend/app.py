from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.model.config import settings


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Computer Science Interview Preparation and Custom LLM API",
    debug=settings.DEBUG,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="User's message",
    )
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    message: str
    response: str
    conversation_id: str | None
    model: str
    model_status: str


@app.get("/")
async def home():
    return {
        "application": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "message": "PrepZio API is running",
        "documentation": "/docs",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "application": settings.APP_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/v1/status")
async def system_status():
    return {
        "application": settings.APP_NAME,
        "model": settings.MODEL_NAME,
        "model_status": "not_trained",
        "message": "The new Custom LLM has not been trained or loaded yet.",
    }


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # The actual trained model will be connected in the inference stage.
    return ChatResponse(
        message=request.message,
        response=(
            "The PrepZio Custom LLM is not connected yet. "
            "Complete the tokenizer, model training and inference stages "
            "before generating AI responses."
        ),
        conversation_id=request.conversation_id,
        model=settings.MODEL_NAME,
        model_status="not_trained",
    )


@app.get("/api/v1/subjects")
async def get_subjects():
    return {
        "subjects": [
            "Python",
            "Java",
            "C",
            "C++",
            "Data Structures",
            "DBMS",
            "Operating Systems",
            "Computer Networks",
            "Artificial Intelligence",
            "Machine Learning",
            "Web Development",
            "Blockchain",
        ]
    }