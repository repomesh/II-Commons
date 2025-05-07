from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from lib import logger
from lib.psql import query, batch_insert
from lib.s3 import put
from tqdm import tqdm
import json
import os

ds = None
BATCH_SIZE = 1000


def get_all_jsonl_files(root_dir):
    jsonl_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.startswith("wiki_") and os.path.isfile(os.path.join(root, file)):
                jsonl_files.append(os.path.join(root, file))
    return jsonl_files


def process_jsonl_file(file_path):
    try:
        total_processed = 0
        records_to_insert = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())
                    s3_key = ds.get_s3_key(record)
                    origin_storage_id = put(line, s3_key)
                    records_to_insert.append((
                        record['id'],
                        int(record['revid']),
                        record['url'],
                        record['title'],
                        origin_storage_id
                    ))
                    if len(records_to_insert) >= BATCH_SIZE:
                        insert_records_batch(records_to_insert)
                        total_processed += len(records_to_insert)
                        records_to_insert = []
                except json.JSONDecodeError as je:
                    logger.error(
                        f"⚠️ JSON decode error in {file_path}, line {line_num}: {je}")
                    continue
                except Exception as e:
                    logger.error(
                        f"⚠️ Error processing line {line_num} in {file_path}: {e}")
                    continue
        if records_to_insert:
            insert_records_batch(records_to_insert)
            total_processed += len(records_to_insert)
        return total_processed
    except Exception as e:
        logger.error(f"⚠️ Failed to process file {file_path}: {e}")
        return 0


def insert_records_batch(records):
    ids = [r[0] for r in records]
    res = query(
        f"SELECT id FROM {ds.get_table_name()} WHERE id in ({','.join(ids)})"
    )
    if len(ids) == len(res):
        logger.info(
            f"Skipping {len(ids)} items records that already exist in the database")
        return
    insert_query = f"""
    INSERT INTO {ds.get_table_name()} (id, revid, url, title, origin_storage_id)
    VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING
    """
    return batch_insert(insert_query, records)


def load(_ds, meta_path):
    global ds
    ds = _ds
    start_time = datetime.now()
    logger.info(f"🚀 Starting Wikipedia data processing at {start_time}")
    jsonl_files = get_all_jsonl_files(meta_path)
    logger.info(f"🔍 Found {len(jsonl_files)} JSONL files to process")
    total_processed = 0
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = []
        for file_path in jsonl_files:
            futures.append(executor.submit(process_jsonl_file, file_path))
        for future in tqdm(futures, desc="Processing files"):
            try:
                file_processed = future.result()
                total_processed += file_processed
            except Exception as e:
                logger.error(f"⚠️ Error in worker thread: {e}")
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"🎉 Processing completed at {end_time}")
    logger.info(f"⏰ Total duration: {duration}")
    logger.info(f"🔢 Total records processed: {total_processed}")


__all__ = [
    'load'
]
