from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from contextlib import asynccontextmanager
import torch
from transformers import AutoModel

# Load the model
embedding_model_name = 'Snowflake/snowflake-arctic-embed-m-v2.0'
embedding_model = None
rerank_model_name = 'jinaai/jina-reranker-m0'
rerank_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        global rerank_model, embedding_model
        if torch.cuda.is_available():
            print('> 🐧 Using CUDA...')
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            # https://github.com/pytorch/pytorch/issues/77764
            print('>  Using MPS...')
            device = torch.device('mps')
        else:
            print('> ⚠️ Using CPU...')
            device = torch.device('cpu')

        embedding_model = SentenceTransformer(embedding_model_name, trust_remote_code=True)
        embedding_model.to(device)
        embedding_model.eval()

        # comment out the flash_attention_2 line if you don't have a compatible GPU
        # attn_implementation = "flash_attention_2"
        attn_implementation = "eager"
        rerank_model = AutoModel.from_pretrained(
            rerank_model_name,
            torch_dtype="auto",
            trust_remote_code=True,
            attn_implementation=attn_implementation
        )
        rerank_model.to(device)
        rerank_model.eval()
    except Exception as e:
        print(f"Failed to initialize services: {str(e)}")
        raise e

    yield

    if embedding_model:
        del embedding_model
        embedding_model = None
    if rerank_model:
        del rerank_model
        rerank_model = None

# Initialize FastAPI app
app = FastAPI(    
    lifespan=lifespan,
    title="Model Serving API",
    description="API for embedding and reranking",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# Define request model
class EmbeddingRequest(BaseModel):
    queries: List[str]
    prompt_name: Optional[str] = "query"

class RerankRequest(BaseModel):
    query: str
    documents: List[str]

@app.post("/embedding")
def get_embeddings(request: EmbeddingRequest):
    try:
        # Compute embeddings
        embeddings = embedding_model.encode(request.queries, prompt_name=request.prompt_name)
        return embeddings.tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/rerank")
def rerank(request: RerankRequest):
    try:
        # Compute embeddings
        def rerank(query, documents):
            text_pairs = [[query, doc] for doc in documents]
            return rerank_model.compute_score(text_pairs, max_length=8192, doc_type="text")
        scores = rerank(request.query, request.documents)
        return scores
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import os

    if __name__ == "__main__":
        
        port = os.getenv("MODEL_API_PORT", 8001)
        port = int(port)
        
        uvicorn.run(app, host="0.0.0.0", port=port)
