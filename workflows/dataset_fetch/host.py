from lib.dataset import init as init_dataset
from lib.config import GlobalConfig
from lib.hatchet import push_dataset_event
from lib.meta import parse_jsonl, parse_dict_parquet, parse_wiki_featured, parse_tube_parquet
import os

BATCH_SIZE = 1

buffer = []
ds = None
dataset_name = None
last_item = 0
limit = 0


def stop():
    import sys
    sys.exit(0)


def trigger(force=False):
    global buffer
    if force or len(buffer) >= BATCH_SIZE:
        if GlobalConfig.DRYRUN:
            print(f"Dryrun: {buffer}")
        else:
            push_dataset_event('fetch', dataset_name, buffer)
        buffer = []


def run(name, meta_path):
    global buffer, ds, dataset_name
    dataset_name = name
    ds = init_dataset(dataset_name)
    i = 0
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
        elif meta_file.endswith('.DS_Store'):
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
