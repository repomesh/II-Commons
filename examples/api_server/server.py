from typing import List
from dotenv import load_dotenv
load_dotenv()
from fastmcp import FastMCP
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from contextlib import asynccontextmanager
import handler

class TextRequest(BaseModel):
    query: str
    max_results: int = 20
    options: handler.QueryConfiguration = handler.QueryConfiguration()

    class Config:
        json_schema_extra = {
            "example": {
                "query": "I want to know information about documentaries related to World War II.",
                "max_results": 20,
                "options": {
                    "refine_query": True,
                    "rerank": True,
                    "vector_weight": 0.9,
                    "bm25_weight": 0.1,
                    "search_type": "all"
                },
            }
        }

class SearchResultTextItem(BaseModel):
    score: float
    url: str
    title: str
    text: str

class SearchResultImageItem(BaseModel):
    score: float
    url: str
    caption: str
    processed_storage_id: str
    aspect_ratio: float
    exif: dict
    source: List[str]
    distance: float

class SearchResp(BaseModel):
    results: List[SearchResultTextItem]
    images: List[SearchResultImageItem]

    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "score": 0.5678,
                        "url": "http://example.com",
                        "title": "Example Title",
                        "text": "Example text content related to the query."
                    }
                ],
                "images": [
                    {
                        "score": 0.1234,
                        "url": "http://example.com/image.jpg",
                        "caption": "Example image caption."
                    }
                ]
            }
        }

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await handler.init()
    except Exception as e:
        print(f"Failed to initialize services: {str(e)}")
        raise e

    yield

    await handler.clean()

app = FastAPI(
    lifespan=lifespan,
    title="Retrieval API",
    description="API for retrieval of documents from the knowledge base",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    root_path="/api/v1"
)

@app.post("/search", response_model=SearchResp, tags=["Search"], operation_id="cg_search")
async def search_text(request: TextRequest):
    """
    Seek common ground knowledge using a text query.

    Args:
        request (TextRequest): The search request containing text query and pagination parameters

    Returns:
        dict: Search results containing similar images

    Raises:
        HTTPException: If services are not initialized or search fails
    """
    try:
        # Generate embedding for the input text
        results, images = await handler.query(request.query, request.max_results, request.options)
        return {"results": results, "images": images}

    except Exception as e:
        print(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/search_image", response_model=SearchResp, tags=["Search"], operation_id="cg_search_image")
async def search_image(
    image_file: UploadFile = File(...),
    max_results: int = Form(20)
):
    """
    Seek common ground knowledge using an image query.

    Args:
        image_file (UploadFile): The image file to search with.
        max_results (int): Maximum number of results to return.

    Returns:
        dict: Search results containing similar images

    Raises:
        HTTPException: If services are not initialized or search fails
    """
    try:
        results, images = await handler.image_query(
            image_file=image_file,
            max_results=max_results
        )
        return {"results": results, "images": images}

    except Exception as e:
        print(f"Image search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image search failed: {str(e)}")
    
# Generate an MCP server directly from the FastAPI app
mcp_server = FastMCP.from_fastapi(app)

if __name__ == "__main__":
    import os
    import signal
    from multiprocessing import Process
    import uvicorn
    port = os.getenv("API_SERVER_PORT", 8080)
    port = int(port)
    mcp_port = os.getenv("API_MCP_SERVER_PORT", port + 1)
    mcp_port = int(mcp_port)


    def run_mcp_server():
        import asyncio
        print(f"Starting MCP server on port {mcp_port}...")
        asyncio.run(mcp_server.run_sse_async(host="0.0.0.0", port=mcp_port))

    def run_uvicorn_server():
        print(f"Starting Uvicorn server on port {port}...")
        uvicorn.run(app, host="0.0.0.0", port=port)

# Create separate processes for MCP server and Uvicorn server
    mcp_process = Process(target=run_mcp_server)
    uvicorn_process = Process(target=run_uvicorn_server)

    # Function to handle SIGINT (Ctrl+C)
    def handle_sigint(signal_number, frame):
        print("\nShutting down gracefully...")
        mcp_process.terminate()
        uvicorn_process.terminate()
        mcp_process.join()
        uvicorn_process.join()
        print("All processes terminated.")
        exit(0)

    # Register the signal handler
    signal.signal(signal.SIGINT, handle_sigint)

    # Start both processes
    mcp_process.start()
    uvicorn_process.start()

    # Wait for both processes to complete
    mcp_process.join()
    uvicorn_process.join()
