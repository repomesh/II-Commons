from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer

# Initialize FastAPI app
app = FastAPI()

# Load the model
model_name = 'Snowflake/snowflake-arctic-embed-m-v2.0'
model = SentenceTransformer(model_name, trust_remote_code=True)

# Define request model
class EmbeddingRequest(BaseModel):
    queries: List[str]
    prompt_name: Optional[str] = "query"

@app.post("/embedding")
def get_embeddings(request: EmbeddingRequest):
    try:
        # Compute embeddings
        embeddings = model.encode(request.queries, prompt_name=request.prompt_name)
        return embeddings.tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    import uvicorn
    import os

    if __name__ == "__main__":
        
        port = os.getenv("MODEL_API_PORT", 8001)
        port = int(port)
        
        uvicorn.run(app, host="0.0.0.0", port=port)
