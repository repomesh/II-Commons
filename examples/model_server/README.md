# Model Server Example

This directory contains an example server responsible for handling model-related tasks, such as reranking and potentially other AI model interactions (like Gemini).

## Configuration

Create a `.env` file in this directory by copying `.env.example`:

```bash
cp .env.example .env
```

Then, fill in the required values in the `.env` file.

### Environment Variables

-   `JINA_API_KEY`: Your API key for Jina AI services. Required if `USE_RERANK` is `jina_api`.
-   `USE_RERANK`: Set to `"jina_api"` to use the Jina Rerank API for reranking tasks, `"local"` for local inference and `none` to disable the reranker.
-   `GEMINI_API_KEY`: Your API key for Google Gemini services. Required for functionalities utilizing Gemini models(for query rewriting).
-   `MODEL_API_PORT`: The port on which this model server will run. Defaults to `8001`.
-   `HF_ENDPOINT`: (Optional) The endpoint URL for accessing Hugging Face models, potentially a mirror. Like `https://hf-mirror.com`.
-   `GEMINI_PROXY`: (Optional) A proxy URL to use for accessing Google Gemini services. Like `http://localhost:2080`.
-   `NLTK_PROXY`: (Optional) A proxy URL to use for accessing NLTK data. Like `http://localhost:2080`.

## Running the Server

```bash
python server.py
```
