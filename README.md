# 🐿️ Chipmunk

<img src="https://github.com/user-attachments/assets/eb5ea0c1-17e4-4b2a-bccb-48dcb20b4344" alt="Chipmunk" width="400">

This repository [Chipmunk](https://en.wikipedia.org/wiki/Chipmunk) contains tools for managing text and image datasets, including loading, fetching, and embedding large datasets.

The dataset processed by these tools are suitable for model training, fine-tuning, RAG, and other applications.

## Requirements

- [PostgreSQL](https://www.postgresql.org/) for metadata and vector storage
- [VectorChord](https://github.com/tensorchord/vectorchord) for vector indexing
- [pg_search](https://github.com/paradedb/paradedb/tree/dev/pg_search#overview) for [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) indexing

## Installation

```bash
$ git clone https://github.com/Intelligent-Internet/dataset-tools.git
$ cd dataset-tools
$ pip install -r requirements.txt
```

## Configuration

Create a `.env` file from [sample.env](./sample.env) and configure the necessary parameters.

Be sure to configure the [PostgreSQL](https://www.postgresql.org/) and [S3](https://aws.amazon.com/s3/) related environment variables. Most of the features are dependent on them.

## Prepare a Image Dataset

`Chipmunk` supports multiple image datasets, for example [PD12M](https://huggingface.co/datasets/Spawning/PD12M), [CC12M](https://github.com/google-research-datasets/conceptual-12m), [
cc12m-cleaned](https://huggingface.co/datasets/opendiffusionai/cc12m-cleaned), and so on. It also supports custom datasets in parquet, jsonl, or csv format. In this demonstration, we will use a [sample mini dataset](https://github.com/Intelligent-Internet/Chipmunk/tree/main/meta/PD12M) which is the [first 100,000 entries from PD12M](https://huggingface.co/datasets/Spawning/PD12M/tree/main/metadata?show_file_info=metadata%2Fpd12m.000.parquet) for the sake of speed.

### 1. Load Metadata to Database

First the dataset meta must be loaded into the database.

```bash
$ python . -w load -d pd12m -p ./meta/PD12M/metadata
```

### 2. Fetch Data from Source

Then we need to fetch raw data items and save them to object storage. It supports [S3](https://aws.amazon.com/s3/) and S3-compatible object storage services. For local deployments, [SeaweedFS](https://github.com/seaweedfs/seaweedfs) is recommended.

```bash
$ python . -w fetch -d pd12m
```

### 3. Embed Images in a Dataset

After the data items are fetched, we can embed the images.

We use [google/siglip2-so400m-patch16-naflex](https://huggingface.co/google/siglip2-so400m-patch16-naflex) as default image embedding model.

```bash
$ python . -w embed_image -d pd12m
```

You can run the above command multiple times parallelly to speed up the embedding process in a single machine or in a distributed environment. `Chipmunk` will automatically divide the dataset into multiple parts and embed them in parallel. And also, a worker can be up and down dynamically, `Chipmunk` will automatically manage the workers and the dataset parts, you don't need to care about it.

## Prepare a Text Dataset

`Chipmunk` is designed to support text based datasets like wikipedia, arXiv and so on. We will use the [Wikipedia English](https://dumps.wikimedia.org/) dataset for demonstration. Full support for arXiv is coming soon.

### 1. Get the Raw Dataset

Navigate to the [wikipedia dump](https://dumps.wikimedia.org/enwiki/20250501/) directory. Download the dump file `pages-articles-multistream` in `xml.bz2` format, like [enwiki-20250501-pages-articles-multistream.xml.bz2](https://dumps.wikimedia.org/enwiki/20250501/enwiki-20250501-pages-articles-multistream.xml.bz2). Extract the `xml` file from the `bz2` archive.

You can use the [sample mini dataset](https://github.com/Intelligent-Internet/Chipmunk/tree/main/meta/wikipedia_en) for testing, jump to the [Load the Dataset to Database](#load-the-dataset-to-database) section.

### 2. Extract Pages from the Raw Dataset

The best way to extract pages from the raw dataset is to use the [wikiextractor](https://github.com/attardi/wikiextractor) tool.

Besure to apply this [patch](https://github.com/attardi/wikiextractor/commit/ab8988ebfa9e4557411f3d4c0f4ccda139e18875) to the `wikiextractor` tool to fix this [issue](https://github.com/attardi/wikiextractor/issues/336) before extracting pages.

```bash
$ wikiextractor enwiki-20250501-pages-articles-multistream.xml --json --no-templates -o /path/to/wikipedia_en
```

Extract pages with links if you need.

```bash
$ wikiextractor enwiki-20250501-pages-articles-multistream.xml --json --no-templates--links -o /path/to/wikipedia_en
```

### 3. Load the Dataset to Database

This step will analyze all the pages extracted from the raw dataset, upload them to the object storage, and save the metadata to the database.

```bash
$ python . -w load -d wikipedia_en -p ./meta/wikipedia_en
```

### 4. Split Pages into Chunks, and Embed the Chunks

This step will split the pages into chunks of a certain size, save the chunks to the chunking database, and embed the chunks.

We use [Snowflake/snowflake-arctic-embed-m-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0) as default text embedding model.

```bash
$ python . -w embed_text -d wikipedia_en
```

You can run the above command multiple times parallelly to speed up the embedding process in a single machine or in a distributed environment. `Chipmunk` will automatically divide the dataset into multiple parts and process them in parallel. And also, a worker can be up and down dynamically, `Chipmunk` will automatically manage the workers and the dataset parts, you don't need to care about it.

## Query

```bash
$ python . -q [TOPIC]
```

## Docker

### Build

```bash
$ docker build -t chipmunk .
```

### Run

```bash
$ docker run --rm --gpus all -v ./.env:/app/.env chipmunk
```
