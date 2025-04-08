from lib.config import GlobalConfig
from lib.dataset import init
from lib.hatchet import push_dataset_event
from lib.utilitas import sha256

BATCH_SIZE = 1

dataset_name = None
buffer = []
ds = None
last_item = 0
limit = 0


def trigger(force=False):
    global buffer
    if force or len(buffer) >= BATCH_SIZE:
        if GlobalConfig.dryrun:
            print(f"Dryrun: {buffer}")
        else:
            if dataset_name == 'wikipedia_en':
                ds_name = 'text_0000001_en'
            else:
                ds_name = dataset_name
            push_dataset_event('embedding_text', ds_name, buffer)
        buffer = []


def get_unprocessed():
    match dataset_name:
        case 'wikipedia_en':
            return ds.query(
                f"SELECT id, origin_storage_id FROM {ds.get_table_name()}"
                + ' WHERE id NOT IN (select source_id from ts_text_0000001_en)'
                + ' AND namespace = %s AND redirecturl = %s'
                + ' AND id > %s ORDER BY id ASC LIMIT %s',
                ('Page', '', last_item, BATCH_SIZE)
            )
        case _:
            raise ValueError(f'Unsupported dataset: {dataset_name}')


def run(name):
    global buffer, last_item, ds
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
