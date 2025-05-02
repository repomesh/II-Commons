from lib.config import GlobalConfig
from lib.dataset import init
from lib.meta import parse_jsonl, parse_dict_parquet, parse_wiki_featured, parse_tube_parquet
import os
import time

BATCH_SIZE = 1000
last_item, limit, i = 0, 0, 0
dataset_name = None
ds = None
buffer = []


def trigger(force=False):
    global buffer
    if force or len(buffer) >= BATCH_SIZE:
        if GlobalConfig.DRYRUN:
            print(f"Dryrun: {buffer}")
        else:
            insert_data({'meta_items': buffer})
        buffer = []


def sleep(seconds=1):
    print(f'⏰ Sleeping for {seconds} second(s) to avoid rate limit...')
    time.sleep(seconds)


def insert_data(args) -> dict:
    meta_items = args['meta_items'] if type(args['meta_items']) == list \
        else [args['meta_items']]
    for meta in meta_items:
        snapshot = ds.snapshot(meta)
        print(f'✨ Processing item: {snapshot}')
        try:
            ds.insert(meta) # TODO: batch insert
            print(f'🔥 Inserted meta: {snapshot}')
        except Exception as e:
            print(f'❌ Error handling {snapshot}: {e}')
    print('👌 Done!')
    return {'meta_items': meta_items}


def run(name, meta_path):
    global buffer, ds, dataset_name
    dataset_name = name
    ds = init(dataset_name)
    meta_files = []
    if os.path.isdir(meta_path):
        for meta_file in sorted(os.listdir(meta_path)):
            meta_files.append(os.path.join(meta_path, meta_file))
    else:
        meta_files.append(meta_path)
    for meta_file in meta_files:
        if dataset_name == 'wikipedia_featured' and os.path.isdir(meta_file):
            meta_items = parse_wiki_featured(meta_file)
        elif dataset_name == 'megalith_10m':
            meta_items = parse_tube_parquet(meta_file)
        elif meta_file.endswith('.parquet'):
            meta_items = parse_dict_parquet(meta_file)
        elif meta_file.endswith('.jsonl'):
            meta_items = parse_jsonl(meta_file)
        elif meta_file.endswith('.json'):
            meta_items = parse_jsonl(meta_file)
        elif meta_file.startswith('.'):
            continue
        else:
            raise ValueError(f'Unsupported file format: {meta_file}')
        for meta in meta_items:
            i += 1
            if last_item and i <= last_item:
                print(f'Catching up: {i}')
                continue
            try:
                meta = ds.map_meta(meta)
            except Exception as e:
                print(f"Error mapping meta: {str(e)}")
                continue
            meta_snapshot = ds.snapshot(meta)
            if ds.exists(meta):
                print(f"Skipping existing {i}: {meta_snapshot}")
                continue
            print(f"Processing row {i}: {meta_snapshot}")
            buffer.append(meta)
            trigger()
            if limit > 0 and i >= limit:
                break
    trigger(force=True)
    print('Done!')


__all__ = [
    'run'
]
