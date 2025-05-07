# 🐿️ Chipmunk

<img src="https://github.com/user-attachments/assets/eb5ea0c1-17e4-4b2a-bccb-48dcb20b4344" alt="Chipmunk" width="400">

This repository `Chipmunk` contains tools for managing text and image datasets, including loading, fetching, and embedding large datasets.

The dataset processed by these tools are suitable for model training, fine-tuning, RAG, and other applications.

## Installation

```bash
$ git clone https://github.com/Intelligent-Internet/dataset-tools.git
$ cd dataset-tools
$ pip install -r requirements.txt
```

## Configuration

Create a `.env` file from [sample.env](./sample.env) and configure the necessary parameters.

Be sure to configure the `POSTGRES` and `S3` related environment variables. Most of the features are dependent on them.

## Usage for Text Dataset

`Chipmunk` supports multiple image datasets, for example [PD12M](https://huggingface.co/datasets/Spawning/PD12M), CC12M, and so on. It also supports custom datasets in parquet, jsonl, or csv format. In this demonstration, we will use the [first 100,000 entries from PD12M](https://huggingface.co/datasets/Spawning/PD12M/tree/main/metadata?show_file_info=metadata%2Fpd12m.000.parquet) for the sake of speed.

### Load Metadata to Database

First the dataset meta must be loaded into the database.

```bash
$ python . -w load -d pd12m -p ./meta/PD12M/metadata
```

### Fetch Data from Source

Then we need to fetch raw data items and save them to object storage. It supports [S3](https://aws.amazon.com/s3/) and S3-compatible object storage services. For local deployments, [SeaweedFS](https://github.com/seaweedfs/seaweedfs) is recommended.

```bash
$ python . -w fetch -d pd12m
```

### Embed Text in a Dataset

`Chipmunk` is designed to support text based datasets like wikipedia, arXiv and so on. We will use the [wikipedia_en](https://huggingface.co/datasets/Spawning/wikipedia_en) dataset for demonstration. Full support for arXiv is coming soon.

## Get the Raw Dataset

Navigate to the [wikipedia dump](https://dumps.wikimedia.org/enwiki/20250501/) directory. Download the dump file `pages-articles-multistream` in `xml.bz2` format, like [enwiki-20250501-pages-articles-multistream.xml.bz2](https://dumps.wikimedia.org/enwiki/20250501/enwiki-20250501-pages-articles-multistream.xml.bz2). Extract the `xml` file from the `bz2` archive.

You can use the sample mini dataset for testing, jump to the [Load the Dataset to Database](#load-the-dataset-to-database) section.

## Extract Pages from the Raw Dataset

The best way to extract pages from the raw dataset is to use the [wikiextractor](https://github.com/attardi/wikiextractor) tool.

```bash
$ wikiextractor enwiki-20250501-pages-articles-multistream.xml --json --no-templates -o /path/to/wikipedia_en
```

Extract pages with links if you need.

```bash
$ wikiextractor enwiki-20250501-pages-articles-multistream.xml --json --no-templates--links -o /path/to/wikipedia_en
```

## Load the Dataset to Database

This step will analyze all the pages extracted from the raw dataset, upload them to the object storage, and save the metadata to the database.

```bash
$ python . -w load -d wikipedia_en -p ./meta/wikipedia_en
```
