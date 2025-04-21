from lib.config import GlobalConfig
from lib.coordination import heartbeat
from lib.dataset import init
from lib.psql import batch_insert
from lib.s3 import download_file
from lib.text import process
from lib.utilitas import json_dumps, sha256, read_json
from lib.utilitas import sha256
import os
import sys
import tempfile
import time

BATCH_SIZE = 30
last_item, limit, buffer = 0, 0, []
src_ds, dst_ds = None, None
source_db, default_ds_name = None, 'text_0000002_en'


def get_unprocessed(name):
    # worker_count, worker_order = heartbeat(name)
    worker_count, worker_order = 1, 0
    # start_time = time.time()
    resp = src_ds.query(
        f"SELECT id, origin_storage_id FROM {src_ds.get_table_name()} t"
        + f' WHERE NOT EXISTS (SELECT 1 from {dst_ds.get_table_name()} s'
        + ' WHERE s.source_id = t.id) AND id %% %s = %s AND id > %s'
        + ' AND t.ignored = FALSE'
        + ' ORDER BY id ASC LIMIT %s',
        (worker_count, worker_order, last_item, BATCH_SIZE)
    )
    # print(
    #     f'Fetching {BATCH_SIZE} rows took {time.time() - start_time:.2f} seconds.'
    # )
    return resp


def trigger(force=False):
    global buffer
    # start_time = time.time()
    if force or len(buffer) >= BATCH_SIZE:
        if GlobalConfig.DRYRUN:
            print(f"Dryrun: {buffer}")
        else:
            embedding({'meta_items': buffer})
        buffer = []
    # print(
    #     f'Submitting {len(buffer)} rows took {time.time() - start_time:.2f} seconds.'
    # )


def embedding(args) -> dict:
    meta_items = args['meta_items'] if type(args['meta_items']) == list \
        else [args['meta_items']]
    task_hash = sha256(json_dumps(meta_items))
    temp_path = tempfile.TemporaryDirectory(suffix=f'-{task_hash}')
    texts = []
    for meta in meta_items:
        snapshot = src_ds.snapshot(meta)
        print(f'✨ Processing item: {snapshot}')
        s3_address = meta['origin_storage_id']
        filename = os.path.join(temp_path.name, f"{meta['hash']}.json")
        try:
            download_file(s3_address, filename)
            print(f'Downloaded {s3_address} to: {filename}')
            json = read_json(filename)
            if len(json['text']) == 0:
                src_ds.update_by_id(meta['id'], {'ignored': True})
                continue
            texts.append({
                'id': meta['id'], 'text': json['text'], 'meta': json,
                'origin_storage_id': meta['origin_storage_id'],
            })
        except Exception as e:
            print(f'❌ ({snapshot}) {e}')
            continue
    meta_items = []
    for txt in texts:
        try:
            print('Embedding Documents...')
            snapshot = json_dumps(txt['id'])
            end_res = process(txt['text'])
        except Exception as e:
            print(f'❌ ({snapshot}) Error embedding: {e}')
            continue
        if end_res is not None:
            snapshot = txt['meta']['url']
            items = []
            for j in range(len(end_res)):
                chk = end_res[j]
                items.append({
                    'title': txt['meta']['title'],
                    'url': txt['meta']['url'],
                    'snapshot': txt['origin_storage_id'],
                    'chunk_index': j,
                    'chunk_text': chk['chunk'],
                    'source_db': source_db,
                    'source_id': txt['id'],
                    'vector': chk['embedding'],
                })
            try:
                db_insert_time = time.time()
                insert_query = f"""INSERT INTO {dst_ds.get_table_name()}
                (title, url, snapshot, chunk_index,
                 chunk_text, source_db, source_id, vector)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (source_db, source_id, chunk_index) DO NOTHING"""
                records = []
                for record in items:
                    records.append((
                        record['title'],
                        record['url'],
                        record['snapshot'],
                        record['chunk_index'],
                        record['chunk_text'],
                        record['source_db'],
                        record['source_id'],
                        record['vector'],
                    ))
                batch_insert(insert_query, records)
                db_insert_time = time.time() - db_insert_time
                print(
                    f'🔥 ({snapshot}) Updated meta: {len(items)} items in {db_insert_time:.2f} seconds'
                )
                del txt['meta']
                del txt['text']
                meta_items.append(txt)
            except Exception as e:
                print(f'❌ ({snapshot}) Error updating meta: {e}')
                print(txt['meta'])
        else:
            print('❌ No embedding result.')
    print('👌 Done!')
    return {'meta_items': meta_items}


def run(name):
    global buffer, last_item, src_ds, dst_ds, source_db
    source_db = name
    if name == 'wikipedia_en':
        dataset_name = default_ds_name
    else:
        dataset_name = name
    i = 0
    try:
        src_ds = init(name)
    except Exception as e:
        print(f"❌ Unable to init src-dataset: {name}. Error: {e}")
        sys.exit(1)
    try:
        dst_ds = init(dataset_name)
    except Exception as e:
        print(f"❌ Unable to init dst-dataset: {dataset_name}. Error: {e}")
        sys.exit(1)
    meta_items = get_unprocessed(name)
    while len(meta_items) > 0:
        should_break = False
        for meta in meta_items:
            i += 1
            last_item = meta['id']
            meta_snapshot = src_ds.snapshot(meta)
            print(f"Processing row {i} - {last_item}: {meta_snapshot}")
            # print(meta)
            buffer.append({
                'id': meta['id'],
                'hash': sha256(meta['origin_storage_id']),
                'origin_storage_id': meta['origin_storage_id'],
            })
            trigger()
            if i >= limit > 0:
                should_break = True
                break
        if should_break:
            break
        meta_items = get_unprocessed(name)
    trigger(force=True)
    print('All Done!')


__all__ = [
    'run'
]
