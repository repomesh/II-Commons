from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from . import handler

class TextRequest(BaseModel):
    query: str
    max_results: int = 5

    class Config:
        json_schema_extra = {
            "example": {
                "query": "I want to know information about documentaries related to World War II.",
                "max_results": 5,
            }
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        handler.init()
    except Exception as e:
        print(f"Failed to initialize services: {str(e)}")
        raise e

    yield

    handler.clean()

app = FastAPI(
    lifespan=lifespan,
    title="Retrieval API",
    description="API for retrieval of documents from the knowledge base",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    root_path="/api/v1"
)


@app.post("/search", tags=["Search"])
async def search_text(request: TextRequest):
    """
    Search for similar images using a text description.

    Args:
        request (TextRequest): The search request containing text query and pagination parameters

    Returns:
        dict: Search results containing similar images with their metadata

    Raises:
        HTTPException: If services are not initialized or search fails
    """
    try:
        # Generate embedding for the input text
        results = await handler.query(request.query, request.max_results)
        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")