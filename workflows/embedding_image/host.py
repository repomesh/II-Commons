from lib.config import GlobalConfig
from lib.dataset import init
from lib.embedding import BATCH_SIZE
from lib.hatchet import push_dataset_event
from lib.utilitas import sha256

dataset_name = None
ds = None
buffer = []
last_item = 0
limit = 0


def trigger(force=False):
    global buffer, dataset_name, ds
    if force or len(buffer) >= BATCH_SIZE:
        if GlobalConfig.dryrun:
            print(f"Dryrun: {buffer}")
        else:
            push_dataset_event('embedding_image', dataset_name, buffer)
        buffer = []


def get_unprocessed():
    return ds.get_unprocessed(limit=BATCH_SIZE, offset=last_item)


def run_host(name):
    global buffer, last_item, dataset_name, ds
    dataset_name = name
    ds = init(name)
    i = 0
    meta_items = get_unprocessed()
    while len(meta_items) > 0:
        should_break = False
        for meta in meta_items:
            i += 1
            last_item = meta['id']
            meta_snapshot = ds.snapshot(meta)
            print(f"Processing row {i} - {last_item}: {meta_snapshot}")
            # print(meta)
            buffer.append({
                'id': meta['id'],
                'hash': sha256(meta['processed_storage_id']) if dataset_name == 'alpha' else meta['hash'],
                'origin_storage_id': None if dataset_name == 'alpha' else meta['origin_storage_id'],
                'processed_storage_id': meta['processed_storage_id'] if dataset_name == 'alpha' else None,
            })
            trigger()
            if i >= limit > 0:
                should_break = True
                break
        if should_break:
            break
        meta_items = get_unprocessed()
    trigger(force=True)
    print('Done!')


__all__ = [
    'run_host'
]
