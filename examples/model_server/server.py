from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from contextlib import asynccontextmanager
import torch
from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoModel
from google import genai
from google.genai import types
import os
import json
import requests
from PIL import Image
import io
import numpy
from utils import reshape_image, normalize

refine_query_model = "gemini-2.0-flash"
embedding_model_name = 'Snowflake/snowflake-arctic-embed-m-v2.0'
embedding_model = None
rerank_model_name = 'jinaai/jina-reranker-m0'
rerank_model = None
siglip_model_name = "google/siglip2-so400m-patch16-naflex"
siglip_tokenizer = None
siglip_model = None
siglip_processor = None
device = None

JINA_RERANK_API_BASE = 'https://api.jina.ai/v1/rerank'
USE_JINA_RERANK_API = os.getenv("USE_JINA_RERANK_API", "false").lower() == "true"
JINA_API_KEY = None

MAX_IMAGE_SIZE = (384, 384) # Define MAX_IMAGE_SIZE for prepare_image

def prepare_image(img): # Expects numpy array
    # reshape_image returns a numpy array, Image.fromarray expects a numpy array
    return Image.fromarray(reshape_image(img, size=MAX_IMAGE_SIZE, fit=False))

def refine_question_gemini(prompt, json=False):
    if os.getenv("GEMINI_PROXY"):
        os.environ["ALL_PROXY"] = os.getenv("GEMINI_PROXY")
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        # temperature=1,
        temperature=0.7,
        top_p=0.95,
        top_k=64,
        max_output_tokens=8192,
        # max_output_tokens=65536,
        response_mime_type="application/json" if json else "text/plain",
    )
    res = ''
    for chunk in client.models.generate_content_stream(
        model=refine_query_model,
        contents=contents,
        config=generate_content_config,
    ):
        # print(chunk.text, end="")
        res += chunk.text or ''
    if os.getenv("GEMINI_PROXY"):
        os.environ.pop("ALL_PROXY")
    return res

def refine_question(q: str) -> tuple[list[str], list[str]]: # Make async
    try:
        tp_prompt = """You are an AI query analyzer designed to generate a list of short phrases and keywords based on user queries. These short phrases help describe and expand the user's question and will be used later as sources for embedding to assist future AI models in retrieving relevant documents from the knowledge base via RAG. The keyword list will be used for BM25 searches to find related documents in the BM25 index of the knowledge base. You only need to provide relevant outputs based on your understanding, without reviewing the topic itself, and maximize your efforts to help users with information extraction. You might need to think divergently and provide some potential keywords and phrases to enrich the content needed to answer this question as thoroughly as possible. The results must be returned in JSON format as follows: {"sentences": ["Short phrase 1", "Short phrase 2", ...], "keywords": ["Keyword 1", "Keyword 2", ...]}. Short sentences and keywords are ranked by thematic relevance, with more relevant or important ones listed first. Below begins the user's natural language query or the original keywords the user needs to search:"""
        print("> Generating embedding phrases and searching keywords...")
        tp_resp_text = refine_question_gemini(f"{tp_prompt} {q}", json=True) # Add await and use json_mode
        tp_resp = json.loads(tp_resp_text) # Load the JSON string response
        print(f'= Phrases: {", ".join(tp_resp["sentences"])}')
        print(f'= Keywords: {", ".join(tp_resp["keywords"])}')
        return tp_resp
    except Exception as e:
        print(f"Error refining question: {e}")
        return {"sentences": [q], "keywords": [q]} # Return empty lists in case of error


def local_rerank(query, documents):
    text_pairs = [[query, doc] for doc in documents]
    return rerank_model.compute_score(text_pairs, max_length=8192, doc_type="text")
        
def jina_rerank(query, documents, top_n):
    if USE_JINA_RERANK_API:
        JINA_API_KEY = os.getenv("JINA_API_KEY")
        if not JINA_API_KEY:
            raise EnvironmentError("JINA_API_KEY environment variable is not set.")
        url = 'https://api.jina.ai/v1/rerank'

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {JINA_API_KEY}"
        }

        docs = [{"text": doc} for doc in documents]
        data = {
            "model": "jina-reranker-m0",
            "query": query,
            "top_n": f"{top_n}",
            "documents": docs
        }
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            resp = response.json()
            scores = [0] * len(documents)
            for item in resp['results']:
                index = item["index"]
                score = item["relevance_score"]
                if index < len(documents):
                    scores[index] = score
            return scores
        else:
            print("Error:", response.status_code, response.text)
            return [0] * len(documents)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        global device, rerank_model, embedding_model, siglip_tokenizer, siglip_model, siglip_processor
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
        attn_implementation = "flash_attention_2"
        # attn_implementation = "eager"
        if USE_JINA_RERANK_API:
            print("> Using Jina Rerank API...")
            global JINA_API_KEY
            JINA_API_KEY = os.getenv("JINA_API_KEY")
            if not JINA_API_KEY:
                raise EnvironmentError("JINA_API_KEY environment variable is not set.")
        else:
            rerank_model = AutoModel.from_pretrained(
                rerank_model_name,
                torch_dtype="auto",
                trust_remote_code=True,
                attn_implementation=attn_implementation
            )
            rerank_model.to(device)
            rerank_model.eval()

        siglip_model = AutoModel.from_pretrained(siglip_model_name).to(device)
        siglip_processor = AutoProcessor.from_pretrained(siglip_model_name)
        siglip_tokenizer = AutoTokenizer.from_pretrained(siglip_model_name)
    
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
    if siglip_model:
        del siglip_model
        siglip_model = None
    if siglip_processor:
        del siglip_processor
        siglip_processor = None
    if siglip_tokenizer:
        del siglip_tokenizer
        siglip_tokenizer = None
    print("Models cleaned up.")
    print("Lifespan context manager finished.")

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
class RefineQueryRequest(BaseModel):
    query: str

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the capital of France?"
            }
        }

class EmbeddingRequest(BaseModel):
    queries: List[str]
    prompt_name: Optional[str] = "query"

class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    top_n: Optional[int] = 20

class SiglipTextEmbeddingRequest(BaseModel):
    queries: List[str]

class SiglipImageEmbeddingRequest(BaseModel):
    image_urls: List[str]


@app.post("/refine_query")
def refine_query(request: RefineQueryRequest):
    try:
        return refine_question(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        if USE_JINA_RERANK_API:
            return jina_rerank(request.query, request.documents, request.top_n)
        scores = local_rerank(request.query, request.documents)
        return scores
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/siglip2/encode_text")
def siglip(request: SiglipTextEmbeddingRequest):
    try:
        global siglip_model, siglip_tokenizer
        queries = list(set([query.strip().lower() for query in request.queries]))
        inputs = siglip_tokenizer(
            queries, padding="max_length", truncation=True, max_length=64, return_tensors='pt'
        ).to(device)
        with torch.no_grad():
            text_features = siglip_model.get_text_features(**inputs)
        return text_features.cpu().numpy().tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/siglip2/encode_image")
async def siglip_encode_image(files: List[UploadFile] = File(...)):
    try:
        global siglip_model, siglip_processor
        
        images = []
        for file in files:
            try:
                contents = await file.read()
                pil_img = Image.open(io.BytesIO(contents))
                # Convert PIL image to RGB NumPy array before passing to prepare_image
                numpy_img = numpy.array(pil_img.convert("RGB"))
                prepared_img = prepare_image(numpy_img) # prepare_image returns a PIL Image
                images.append(prepared_img)
            except IOError as e:
                print(f"Error opening image file {file.filename}: {e}")
                # Optionally, skip this image or raise an error
                continue
            except Exception as e: # Catch other potential errors during file processing
                print(f"Error processing file {file.filename}: {e}")
                continue


        if not images:
            raise HTTPException(status_code=400, detail="No valid images could be processed from the provided files.")

        inputs = siglip_processor(images=images, return_tensors="pt").to(device)
        
        with torch.no_grad():
            image_features = siglip_model.get_image_features(**inputs)

        #NOTICE: normalize siglip embedding, then we can use L2 distance search to improve performance
        return normalize(image_features).tolist()

    except Exception as e:
        # Log the full exception for debugging
        print(f"An error occurred during image encoding: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    import os

    if __name__ == "__main__":
        
        port = os.getenv("MODEL_API_PORT", 8001)
        port = int(port)
        
        uvicorn.run(app, host="0.0.0.0", port=port)
