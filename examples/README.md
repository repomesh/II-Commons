# ii Common Ground Examples

This directory contains examples demonstrating the usage of the "ii common ground" components.

## Overview

These examples showcase a system designed for multimodal search, leveraging text and image embeddings, BM25 search, and reranking capabilities. The system is composed of several microservices that can be orchestrated using Docker Compose.

## Getting Started

You can easily start all the example services using Docker Compose:

```bash
docker compose up -d
```

## Components

The example setup includes the following components:

### 1. Database (`db/`)

-   **Description:** Contains the PostgreSQL database and necessary plugins (like `pgvector`) used by the system.
-   **Deployment:** Can be started directly using Docker via the provided `docker-compose.yaml`.

### 2. Model Server (`model_server/`)

-   **Description:** Provides HTTP APIs for core AI functionalities:
    -   Text Embedding: Generates vector representations for text.
    -   Image Embedding: Generates vector representations for images.
    -   Reranking: Reorders search results for improved relevance.
-   **Deployment:** Requires deployment on a machine with significant computational resources (e.g., GPU support) to handle the model inference efficiently.

### 3. API Server (`api_server/`)

-   **Description:** Offers HTTP APIs for performing text-to-multimodal search. It combines:
    -   BM25: A keyword-based search algorithm.
    -   Vector Similarity Search: Finds items based on the closeness of their embeddings.
-   **Dependencies:** Needs network access to the database (`db/`) for direct querying and the `model_server/` for embedding generation if not pre-computed.

### 4. MCP Server (`mcp_server/`)

-   **Description:** A lightweight server designed to run client-side or as a local proxy. It acts as a bridge, translating requests using the Model Context Protocol (MCP) into standard HTTP requests directed to the `api_server`.
-   **Configuration:** Requires the endpoint (URL) of the `api_server` to be configured.
-   **Purpose:** Facilitates interaction with the backend services through a standardized protocol, potentially simplifying client implementations.

## Configuration

Ensure you configure the necessary environment variables, particularly the endpoints for communication between the `mcp_server`, `api_server`, and `model_server`. Refer to the `.env.example` files within each component's directory for guidance.
