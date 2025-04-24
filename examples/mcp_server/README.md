# Common Ground MCP Server Example

This directory contains an example implementation of a Model Context Protocol (MCP) server using the `fastmcp` library.

This server provides tools to search a Common Ground Knowledge Base via an external API.

## Features

- **`cg_search_deep`**: Performs a search using query rewriting and result reranking for potentially better relevance.
- **`cg_search`**: Performs a direct hybrid search (embedding + BM25) without rewriting or reranking.

## Prerequisites

- Python 3.12+
- `uv` package manager (or `pip`)

## Setup

1.  **Clone the repository (if you haven't already).**
2.  **Navigate to this directory:**
    ```bash
    cd examples/mcp_server
    ```
3.  **Install dependencies:**
    ```bash
    uv sync
    ```
    (or `pip install -r requirements.txt` if you generate one from `pyproject.toml`)

4.  **Configure Environment Variables:**
    Create a `.env` file by copying the example:
    ```bash
    cp .env.example .env
    ```
    Edit the `.env` file and set the `API_SERVER_BASE_URL` to the correct URL of your Common Ground API server.
    ```dotenv
    API_SERVER_BASE_URL="http://your-api-server-address:port"
    ```

## Running the Server

Execute the server script:

```bash
uv run python server.py
```

Or simply:

```bash
python server.py
```

The MCP server will start and be ready to accept connections from compatible clients.

## Usage

Once running, MCP clients can connect to this server and utilize the `cg_search` and `cg_search_deep` tools. Refer to the MCP client documentation for connection details.
