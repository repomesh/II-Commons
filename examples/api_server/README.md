# Retrieval API Server

This project provides a FastAPI-based API server for retrieving documents and related images from a knowledge base using text queries. It leverages a PostgreSQL database utilizing [VectorChord](https://github.com/tensorchord/VectorChord) and [pg_search](https://github.com/paradedb/paradedb/tree/dev/pg_search) for efficient similarity search and connects to a separate model server for generating embeddings. Additionally, it automatically exposes its search functionality through an integrated MCP (Model Context Protocol) server.

## Features

*   Text-based search for relevant documents and images.
*   Configurable search options (e.g., reranking).
*   Integration with PostgreSQL, utilizing [VectorChord](https://github.com/tensorchord/VectorChord) and [pg_search](https://github.com/paradedb/paradedb/tree/dev/pg_search) for vector storage and search.
*   Requires connection to a separate Model Server for generating embeddings.
*   Built with Python and FastAPI, providing interactive API documentation (Swagger UI and ReDoc).
*   Includes a Dockerfile for easy containerization and deployment.
*   Provides an integrated MCP server exposing the search functionality.

## Prerequisites

*   Python 3.12+
*   Access to a PostgreSQL database instance with the `VectorChord` and `pg_search` extensions enabled/installed.
*   Access to a running Model Server (URL specified in configuration).
*   (Optional) Docker for containerized deployment.

## Installation

1.  **Clone the repository:**
    ```bash
    # Assuming you are in the root of the main project
    cd examples/api_server
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

The server is configured using environment variables. A template file `.env.example` is provided.

1.  **Create a `.env` file:**
    ```bash
    cp .env.example .env
    ```

2.  **Edit the `.env` file** with your specific settings:

    ```dotenv
    # PostgreSQL Connection Details
    POSTGRES_HOST="your_postgres_host"
    POSTGRES_PORT="5432" # Or your postgres port
    POSTGRES_USER="your_postgres_user"
    POSTGRES_PASSWORD="your_postgres_password"
    POSTGRES_DB="your_database_name"

    # URL of the Model Server (for embeddings)
    MODEL_SERVER_URL_BASE="http://your_model_server_host:port"

    # Optional NLTK Proxy (if needed for downloading NLTK data behind a proxy)
    NLTK_PROXY="http://your_proxy_host:port"

    # Port for the API server
    PORT=18889 # Or any port you prefer
    ```

## Running the Server

You can run the server directly using Python or via Docker.

### Directly with Python

```bash
python server.py
```

This command starts two processes:
1.  The FastAPI/Uvicorn server on the port specified by `PORT` in your `.env` file (default: 18889).
2.  The MCP server on the next consecutive port (default: `PORT + 1`, e.g., 18890).

*   API Base URL: `http://localhost:{PORT}/api/v1`
*   Interactive API Docs (Swagger): `http://localhost:{PORT}/docs`
*   Alternative API Docs (ReDoc): `http://localhost:{PORT}/redoc`
*   MCP Server SSE Endpoint: `http://localhost:{PORT+1}/sse`

### Using Docker

1.  **Build the Docker image:**
    ```bash
    docker build -t retrieval-api-server .
    ```

2.  **Run the container:**
    Make sure your `.env` file is configured correctly in the `examples/api_server` directory.
    ```bash
    # Replace 18889 with the PORT specified in your .env if different
    docker run --rm -p 18889:18889 --env-file .env retrieval-api-server
    ```
    This will start the container, and the API server will be accessible on `http://localhost:18889` (or the mapped port) on your host machine. The MCP server will run inside the container on port `PORT + 1`.

## MCP Server

The application automatically starts an MCP server that mirrors the functionality of the `/search` API endpoint.

*   **Host:** `0.0.0.0`
*   **Port:** Listens on the port defined by the `RAG_MCP_SERVER_PORT` environment variable, which defaults to `PORT + 1` (e.g., if `PORT` is 18889, MCP runs on 18890).
*   **Tool:** Exposes the search functionality, likely named based on the FastAPI operation ID (`cg_search`). You can connect to this server using an MCP client to utilize the search capabilities programmatically.
