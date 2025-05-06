# Chipmunk

![image](https://github.com/user-attachments/assets/eb5ea0c1-17e4-4b2a-bccb-48dcb20b4344)

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

## Usage

### Load Metadata to Database

First the dataset meta must be loaded into the database. For this demonstration, we will use the [first 100,000 entries from PD12M](https://huggingface.co/datasets/Spawning/PD12M/tree/main/metadata?show_file_info=metadata%2Fpd12m.000.parquet) for the sake of speed.

```bash
$ python . -w load -d pd12m -p ./meta/PD12M/metadata
```

### Fetch Data from Source

Then we need to fetch raw data items and save them to object storage. It supports [S3](https://aws.amazon.com/s3/) and S3-compatible object storage services. For local deployments, [SeaweedFS](https://github.com/seaweedfs/seaweedfs) is recommended.

```bash
$ python . -w fetch -d pd12m
```

### Embed Image in a Dataset

After the data items are fetched, we can embed the images.

```bash
$ python . -w embed_image -d pd12m
```
