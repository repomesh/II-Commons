# Model Server Example

This directory contains an example server responsible for handling model-related tasks, such as reranking and potentially other AI model interactions (like Gemini).

## Configuration

Create a `.env` file in this directory by copying `.env.example`:

```bash
cp .env.example .env
```

Then, fill in the required values in the `.env` file.

### Environment Variables

-   `JINA_API_KEY`: Your API key for Jina AI services. Required if `USE_JINA_RERANK_API` is true.
-   `USE_JINA_RERANK_API`: Set to `"true"` to use the Jina Rerank API for reranking tasks. Defaults to `"true"`. If set to false or another value, a local reranker model will be used (jinaai/jina-reranker-m0).
-   `GEMINI_API_KEY`: Your API key for Google Gemini services. Required for functionalities utilizing Gemini models(for query rewriting).
-   `MODEL_API_PORT`: The port on which this model server will run. Defaults to `8001`.
-   `HF_ENDPOINT`: (Optional) The endpoint URL for accessing Hugging Face models, potentially a mirror. Like `https://hf-mirror.com`.
-   `GEMINI_PROXY`: (Optional) A proxy URL to use for accessing Google Gemini services. Like `http://localhost:2080`.
-   `NLTK_PROXY`: (Optional) A proxy URL to use for accessing NLTK data. Like `http://localhost:2080`.

## Running the Server

(Instructions on how to run the server would typically go here, e.g., using Docker or Python directly. This needs to be added based on how the server is intended to be run.)

```bash
python server.py
```