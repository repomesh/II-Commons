# II-Commons

II-Commons is a platform for collaboratively developing large, shared knowledge bases. It offers tools for distributed data handling, embedding computation, index creation, and information retrieval. Organizations and individuals can use it to create private or public knowledge resources.

For more details about our project, please visit our [blog post](https://www.ii.inc/web/blog/post/).


## Features

This repository II-Commons contains tools for managing text and image datasets, including loading, fetching, and embedding large datasets.

The dataset processed by these tools are suitable for model training, fine-tuning, RAG, MCP, and other applications.


## Requirements

- [PostgreSQL](https://www.postgresql.org/) for metadata and vector storage ([PostgreSQL License](https://opensource.org/license/postgresql))
- [VectorChord](https://github.com/tensorchord/vectorchord) for vector indexing ([ELv2](https://github.com/tensorchord/VectorChord/blob/main/licenses/LICENSE.ELv2), [AGPLv3](https://github.com/tensorchord/VectorChord/blob/main/licenses/LICENSE.AGPLv3))
- [pg_search](https://github.com/paradedb/paradedb/tree/dev/pg_search#overview) for [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) indexing ([AGPLv3](https://github.com/paradedb/paradedb?tab=AGPL-3.0-1-ov-file))

## Quick start

You can build your own dataset from scratch or quickly begin experimenting with our pre-prepared datasets.

This session shows how to recovery from our pre-computed database backup to run a vector similarity search instance.

Download a database backup from huggingface: [Wikipedia English](https://huggingface.co/datasets/Intelligent-Internet/wikipedia_en) or [PD12M](https://huggingface.co/datasets/Intelligent-Internet/pd12m)

Use our [Docker image](https://github.com/Intelligent-Internet/II-Commons/tree/main/examples/db) to run a postgresql node. for example, the `Wikipedia English` download dir is `/data/wikipedia_en`.

> [!NOTE]
> the default postgres password is `postgres.1234`, please change the password!


```
sudo docker run --rm -it \
  --name postgres-localvector \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres.1234 \
  -e POSTGRES_DB=localvector \
  -e PGDATA=/var/lib/postgresql/data/pgdata \
  -v /data/wikipedia_en:/var/lib/postgresql/data \
  -p 5432:5432 \
  postgres-17-parade-vchord
```

Use `psql` command to connect the postgresql node, and connect to database `localvector`:
```
postgres=# \c localvector
```
Run `\dx` command to make sure extensions `pg_search` and `vchord` are available.

Setup `probes` for vectorchord query, you can try a higher value to blance [query and performance](https://docs.vectorchord.ai/vectorchord/usage/performance-tuning.html#query-performance).
```
ALTER SYSTEM SET vchordrq.probes = 100;
```
then restart postgresql. Congratulation, the database is ready to use.

Next step, try to run [benchmark](https://github.com/Intelligent-Internet/II-Commons/tree/main/examples/benchmark), or [api server](https://github.com/Intelligent-Internet/II-Commons/tree/main/examples).

> [!NOTE]
> warm the index to improve performance:
> ```sql
> SELECT vchordrq_prewarm('ts_wikipedia_en_embed_vector_index');
> ```

## Installation

```bash
$ git clone https://github.com/Intelligent-Internet/ii-commons.git
$ cd ii-commons
$ pip install -r requirements.txt
```

## Configuration

Create a `.env` file from [sample.env](./sample.env) and configure the necessary parameters.

Be sure to configure the [PostgreSQL](https://www.postgresql.org/) and [S3](https://aws.amazon.com/s3/) related environment variables. Most of the features are dependent on them. The easiest way is to run it use our [Docker image](https://github.com/Intelligent-Internet/ii-commons/blob/main/examples/db/Dockerfile), or build your own


## Prebuilt datasets

We provide prebuilt datasets for your use. You can import, index and use them out of the box.

- 🤗 [Wikipedia English](https://huggingface.co/datasets/Intelligent-Internet/wikipedia_en)
- 🤗 [PD12M](https://huggingface.co/datasets/Intelligent-Internet/pd12m)

Skip the preparation steps and go to the [Query](#query) section if you want to use these prebuilt versions.

More prebuilt datasets are under construction and will be released soon.

## Evaluation

Evaluation NDCG@10 on [TREC-DL 2019](https://microsoft.github.io/TREC-2019-Deep-Learning/), with MS Marco v1.1 Dataset. Retrieval 30 results/query, (Hybrid search includes 30 embedding results and 30 BM25 results) with similarly sorting and reranker model.

| Approach       | Similarly| Ms-marco-MiniLM-L12-v2 [^1] | Bge-reranker-v2-m3 |
| ------------- |:-------------: | :---: |:---: |
| BM25 (pg_search)   | 0.302 | 0.418  | 0.415  |
| embedding (VectorChord)   | 0.661 |  0.712  | 0.700  |
| emb * 0.8 + bm25 * 0.2 | 0.598 | 0.723 | **0.726** |
| emb * 1.0 + bm25 * 0 [^2] | **0.661** | **0.733** | 0.723 |

[^1]: Ms-marco-MiniLM-L12-v2 trained on the MS Marco Passage Ranking tasks

[^2]: emb*1.0 / bm25*0: BM25 results included in search, scores set to 0

## Benchmark and Cost

Run random 500 queries on:

Google Cloud, e2-standard-2 (2 vCPU, 1 core, 8 GB memory)

* database dir on 100 GB ssd: Average 0.13s/query  (cost ~US$67/month)
* database dir on 100 GB balanced persistent  : Average 0.32s/query (cost ~US$60/month)

## Prepare a Image Dataset

`ii-Commons` supports multiple image datasets, for example [PD12M](https://huggingface.co/datasets/Spawning/PD12M), [CC12M](https://github.com/google-research-datasets/conceptual-12m), [
cc12m-cleaned](https://huggingface.co/datasets/opendiffusionai/cc12m-cleaned), and so on. It also supports custom datasets in parquet, jsonl, or csv format. In this demonstration, we will use a [sample mini dataset](https://github.com/Intelligent-Internet/ii-commons/tree/main/meta/PD12M) which is the [first 100,000 entries from PD12M](https://huggingface.co/datasets/Spawning/PD12M/tree/main/metadata?show_file_info=metadata%2Fpd12m.000.parquet) for the sake of speed.

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

You can run the above command multiple times parallelly to speed up the embedding process in a single machine or in a distributed environment. `II-commons` will automatically divide the dataset into multiple parts and embed them in parallel. And also, a worker can be up and down dynamically, `II-commons` will automatically manage the workers and the dataset parts, you don't need to care about it.


## Prepare a Text Dataset

`II-commons` is designed to support text based datasets like wikipedia, arXiv and so on. We will use the [Wikipedia English](https://dumps.wikimedia.org/) dataset for demonstration. Full support for arXiv is coming soon.

### 1. Get the Raw Dataset

Navigate to the [wikipedia dump](https://dumps.wikimedia.org/enwiki/20250501/) directory. Download the dump file `pages-articles-multistream` in `xml.bz2` format, like [enwiki-20250501-pages-articles-multistream.xml.bz2](https://dumps.wikimedia.org/enwiki/20250501/enwiki-20250501-pages-articles-multistream.xml.bz2). Extract the `xml` file from the `bz2` archive.

You can use the [sample mini dataset](https://github.com/Intelligent-Internet/ii-commons/tree/main/meta/wikipedia_en) for testing, jump to the [Load the Dataset to Database](#load-the-dataset-to-database) section.

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

You can run the above command multiple times parallelly to speed up the embedding process in a single machine or in a distributed environment. `II-commons` will automatically divide the dataset into multiple parts and process them in parallel. And also, a worker can be up and down dynamically, `II-commons` will automatically manage the workers and the dataset parts, you don't need to care about it.


## Query

```bash
$ python . -q [TOPIC]
```


## Docker

### Build

```bash
$ docker build -t ii-commons .
```

### Run

```bash
$ docker run --rm --gpus all -v ./.env:/app/.env ii-commons
```

## Try the demo API/MCP services

Checkout the [documentation](examples/) for API/MCP services and more details.


## FAQ


## What's Next: Our Roadmap

- [ ] Simplify installation and operation.
- [ ] Offer more pre-computed indexes for modalities like PDFs, video and audio.
- [ ] Create more AI-assisted generated knowledge bases for public good.
- [ ] Provide API services for datasets.
- [ ] Establish a knowledge base hub for easier sharing and downloading.
- [ ] Develop a desktop version for personal everyday data retrieval.
