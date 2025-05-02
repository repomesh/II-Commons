#!/usr/bin/env python3

from lib.psql import query, batch_insert
from lib.s3 import s3, get_address_by_key
import os
import json
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("wikipedia_import.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

S3_BUCKET = "testing-01"
DATA_DIR = "/Volumes/Betty/Datasets/wiki_ext"
MAX_WORKERS = 8
BATCH_SIZE = 1000


def get_all_jsonl_files(root_dir):
    """Recursively find all JSONL files in the given directory structure"""
    jsonl_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.startswith("wiki_") and os.path.isfile(os.path.join(root, file)):
                jsonl_files.append(os.path.join(root, file))
    return jsonl_files


def process_jsonl_file(file_path):
    """Process a single JSONL file"""
    try:
        total_processed = 0
        records_to_insert = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    # Parse JSON record
                    record = json.loads(line.strip())

                    # Generate a unique S3 key
                    s3_key = os.path.join(
                        'enwiki-20250201-pa-ms', f"{record['id']}.json"
                    )

                    # Upload record to S3
                    s3.put_object(
                        Bucket=S3_BUCKET,
                        Key=s3_key,
                        Body=line,
                        ContentType='application/json'
                    )

                    # Add record to batch for database insertion
                    records_to_insert.append((
                        record['id'],
                        int(record['revid']),
                        record['url'],
                        record['title'],
                        get_address_by_key(s3_key)
                    ))

                    # Insert in batches to improve performance
                    if len(records_to_insert) >= BATCH_SIZE:
                        insert_records_batch(records_to_insert)
                        total_processed += len(records_to_insert)
                        records_to_insert = []

                except json.JSONDecodeError as je:
                    logger.error(
                        f"JSON decode error in {file_path}, line {line_num}: {je}")
                    continue
                except Exception as e:
                    logger.error(
                        f"Error processing line {line_num} in {file_path}: {e}")
                    continue

        # Insert any remaining records
        if records_to_insert:
            insert_records_batch(records_to_insert)
            total_processed += len(records_to_insert)

        return total_processed

    except Exception as e:
        logger.error(f"Failed to process file {file_path}: {e}")
        return 0


def insert_records_batch(records):
    # check if already exists {
    ids = [r[0] for r in records]
    res = query(
        f"SELECT id FROM ts_wikipedia_en WHERE id in ({','.join(ids)})"
    )
    if len(ids) == len(res):
        logger.info(
            f"Skipping {len(ids)} items records that already exist in the database")
        return
    # }
    """Insert a batch of records into the database"""
    insert_query = """
    INSERT INTO ts_wikipedia_en (id, revid, url, title, origin_storage_id)
    VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING
    """
    batch_insert(insert_query, records)


def main():
    """Main function to orchestrate the processing"""
    start_time = datetime.now()
    logger.info(f"Starting Wikipedia data processing at {start_time}")

    # Find all JSONL files
    jsonl_files = get_all_jsonl_files(DATA_DIR)
    logger.info(f"Found {len(jsonl_files)} JSONL files to process")

    # Create folder structure in S3 if needed
    try:
        s3.head_bucket(Bucket=S3_BUCKET)
    except:
        logger.error(
            f"S3 bucket {S3_BUCKET} not accessible. Please check your configuration.")
        return

    total_processed = 0

    # Process files using thread pool for better performance
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []

        # Submit tasks to thread pool
        for file_path in jsonl_files:
            # Create a new DB connection for each worker thread
            futures.append(executor.submit(process_jsonl_file, file_path))

        # Process results with progress bar
        for future in tqdm(futures, desc="Processing files"):
            try:
                file_processed = future.result()
                total_processed += file_processed
            except Exception as e:
                logger.error(f"Error in worker thread: {e}")

    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Processing completed at {end_time}")
    logger.info(f"Total duration: {duration}")
    logger.info(f"Total records processed: {total_processed}")


if __name__ == "__main__":
    main()
