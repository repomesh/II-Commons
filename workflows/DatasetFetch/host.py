from lib.dataset import init
from lib.hatchet import push_dataset_event
from lib.meta import parse_jsonl, parse_parquet, parse_wiki_featured, parse_megalith_parquet
from lib.utilitas import get_file_type
import os
import tempfile
from lib.s3 import download_file

DATASET_BASE = '/Volumes/Betty/Datasets/meta'
DATASETS = {
    'cc12m': {'meta_path': 'meta.tsv'},  # done
    'cc12m_cleaned': {'meta_path': 'meta.jsonl'},  # done
    'cc12m_woman': {'meta_path': 'meta.jsonl'},  # done
    'vintage_450k': {'meta_path': 'meta.parquet'},  # done
    'PD12M': {'meta_path': 'meta'},  # done
    'wikipedia_featured': {'meta_path': 'meta'},  # done
    'megalith_10m': {'meta_path': 'meta'},  # Stopped, flickr limitation
    'arxiv': {'meta_path': 'arxiv-metadata-hash-abstracts-v0.2.0-2019-03-01.json'},
}

DATASET = 'arxiv'
META_PATH = os.path.join(DATASET_BASE, DATASET, DATASETS[DATASET]['meta_path'])
BATCH_SIZE = 1

buffer = []
ds = init(DATASET)
last_item = 0
limit = 0


def stop():
    import sys
    sys.exit(0)


def trigger(force=False):
    global buffer
    if force or len(buffer) >= BATCH_SIZE:
        push_dataset_event('fetch', DATASET, buffer)
        buffer = []


def run_host():
    global buffer
    i = 0
    meta_files = []
    if os.path.isdir(META_PATH):
        for meta_file in sorted(os.listdir(META_PATH)):
            meta_files.append(os.path.join(META_PATH, meta_file))
    else:
        meta_files.append(META_PATH)
    for meta_file in meta_files:
        if DATASET == 'wikipedia_featured' and os.path.isdir(meta_file):
            meta_items = parse_wiki_featured(meta_file)
        elif DATASET == 'megalith_10m':
            meta_items = parse_megalith_parquet(meta_file)
        elif meta_file.endswith('.parquet'):
            meta_items = parse_parquet(meta_file)
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
                print(meta)
                continue
            meta_snapshot = ds.snapshot(meta)
            # if DATASET == 'arxiv':
            #     temp_path = tempfile.TemporaryDirectory(
            #         suffix=f'-{meta["hash"]}'
            #     )
            #     filename = os.path.join(temp_path.name, f"{meta['hash']}.pdf")
            #     s3_key = ds.get_s3_key(meta)
            #     if ds.exists(meta):
            #         try:
            #             download_file(s3_key, filename)
            #             type = get_file_type(filename)
            #             if type == 'PDF':
            #                 print(f"File is OK {i}: {s3_key}")
            #                 continue
            #             elif type == 'TEXT':
            #                 print(f"File is TEXT {i}: {s3_key}")
            #         except Exception as e:
            #             print(f"Not ready {i}: {s3_key}, refetch...")
            #         print(f'Delete old meta {i}: {meta["paper_id"]}')
            #         ds.query(
            #             f"DELETE FROM {ds.get_table_name()} WHERE paper_id = %s",
            #             (meta['paper_id'],)
            #         )
            # el
            if ds.exists(meta):
                print(f"Skipping existing: {meta_snapshot}")
                continue
            print(f"Processing row {i}: {meta_snapshot}")
            # print(meta)
            buffer.append(meta)
            trigger()
            if limit > 0 and i >= limit:
                break
    trigger(force=True)
    print('Done!')


__all__ = [
    'run_host'
]
