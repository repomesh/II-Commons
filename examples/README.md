# II-Commons Examples

This directory contains examples demonstrating the usage of the "II-Commons" components.

## Overview

These examples showcase a system designed for multimodal search, leveraging text and image embeddings, BM25 search, and reranking capabilities. The system is composed of several microservices that can be orchestrated using Docker Compose.

## Getting Started

If you already have an available database that contains data and indexes. You can easily start all the example services using Docker Compose.

```bash
docker compose up -d
```

## Components

The example setup includes the following components:

### 1. [Model Server](model_server/)

-   **Description:** Models inference api provider:
    -   Text Embedding: Generates vector representations for text.
    -   Image Embedding: Generates vector representations for images.
    -   Reranking: Reorders search results for improved relevance.
-   **Deployment:** Requires deployment on a machine with significant computational resources (e.g., GPU support) to handle the model inference efficiently.
-   **Tips:** With only a CPU, query embedding is feasible, but the rerank model will be too slow on CPU inference. Your alternatives are to use a rerank model service from API providers, or to disable reranking and use embedding similarity for ranking (note: rerank models usually have a better ranking quality).

### 3. [API and MCP Server](api_server/)

-   **Description:** Offers HTTP APIs for performing text-to-multimodal search, and Model Context Protocol (MCP) bridge to the API. It combines:
    -   BM25: A keyword-based search algorithm.
    -   Vector Similarity Search: Finds items based on the closeness of their embeddings.
-   **Dependencies:** Needs network access to the embedding database for direct querying and the `model_server/` for query embedding generation, and reranking

### 4. [Demo web](demosite/)
-   **Description:** A simple web page for demo search apis.
-   **Dependencies:** Needs network access to the `api_server`

## Configuration

Ensure you configure the necessary environment variables.  Copy the .env.example file to .env and modify the configuration as needed.

```bash
cp .env.example .env
```
