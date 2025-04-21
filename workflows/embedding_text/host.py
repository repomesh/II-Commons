from lib.config import GlobalConfig
from lib.dataset import init
from lib.hatchet import push_dataset_event
from lib.utilitas import sha256
import time
BATCH_SIZE = 30

dataset_name = None
buffer = []
ds = None
last_item = 349860
limit = 0
default_ds_name = 'text_0000002_en'


def trigger(force=False):
    global buffer
    # start_time = time.time()
    if force or len(buffer) >= BATCH_SIZE:
        if GlobalConfig.DRYRUN:
            print(f"Dryrun: {buffer}")
        else:
            if dataset_name == 'wikipedia_en':
                ds_name = default_ds_name
            else:
                ds_name = dataset_name
            push_dataset_event('embedding_text', ds_name, buffer)
        buffer = []
    # print(
    #     f'Submitting {len(buffer)} rows took {time.time() - start_time:.2f} seconds.'
    # )


def get_unprocessed():
    match dataset_name:
        case 'wikipedia_en':
            # start_time = time.time()
            resp = ds.query(
                f"SELECT id, origin_storage_id FROM {ds.get_table_name()} t"
                + f' WHERE NOT EXISTS (SELECT 1 from ts_{default_ds_name} s'
                + ' WHERE s.source_id = t.id) AND id > %s'
                + ' ORDER BY id ASC LIMIT %s', (last_item, BATCH_SIZE)
            )
            # print(
            #     f'Fetching {BATCH_SIZE} rows took {time.time() - start_time:.2f} seconds.'
            # )
            return resp
        case _:
            raise ValueError(f'Unsupported dataset: {dataset_name}')


def run(name):
    global buffer, last_item, ds, dataset_name
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
                'hash': sha256(meta['origin_storage_id']),
                'origin_storage_id': meta['origin_storage_id'],
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
    'run'
]
