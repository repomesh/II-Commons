# Custom PostgreSQL Docker Image with ParadeDB and vchord

This directory contains a Dockerfile to build a custom PostgreSQL image based on ParadeDB. This image is specifically configured to utilize **`vchord` for vector similarity search** and ParadeDB's **`pg_search` for BM25 sparse vector search**.

## Included Extensions

This image includes and preloads the following key PostgreSQL extensions:

*   **ParadeDB Extensions:**
    *   `pg_search`: Provided by the base ParadeDB image. In this project, it is used for **BM25 sparse vector search**.
    *   `pg_cron`: Provided by the base ParadeDB image for job scheduling.
*   **vchord:**
    *   Added from the `ghcr.io/tensorchord/vchord-postgres` image. In this project, it is used for **vector similarity search**.

The `shared_preload_libraries` setting in `postgresql.conf` is configured to load `pg_search`, `pg_cron`, and `vchord` when the PostgreSQL server starts.

## Usage

To run a container using the pre-built image:

```bash
docker run --rm -it \
  --name postgres-test \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=testdb \
  -p 5432:5432 \
  us-central1-docker.pkg.dev/backend-alpha-97077/iirepo/postgres-17-parade-vchord
```

## Enabling Extensions in PostgreSQL

While the extension libraries are included and preloaded, you still need to explicitly enable `vchord` within a specific database before using its functions and types. Connect to your target database (e.g., using `psql`) and run:

```sql
-- Connect to your target database, e.g., testdb
\c testdb

CREATE EXTENSION IF NOT EXISTS vchord;
CREATE EXTENSION IF NOT EXISTS pg_search;

-- Verify enabled extensions (optional)
\dx
```