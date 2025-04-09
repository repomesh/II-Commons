from lib.dataset import init
from lib.caption import BATCH_SIZE
from lib.hatchet import push_dataset_event
from lib.config import GlobalConfig

dataset_name = None
buffer = []
ds = None
last_item = 0
limit = 8


def trigger(force=False):
    global buffer
    if force or len(buffer) >= BATCH_SIZE:
        if GlobalConfig.DRYRUN:
            print(f"Dryrun: {buffer}")
        else:
            push_dataset_event('caption', dataset_name, buffer)
        buffer = []


def get_unprocessed():
    return ds.query(
        f'SELECT id, processed_storage_id FROM {ds.get_table_name()}'
        + " WHERE id > %s AND caption_qw25vl != ''"
        + ' ORDER BY id ASC LIMIT %s', (last_item, BATCH_SIZE)
    )


def run(name):
    global buffer, ds, last_item, dataset_name
    dataset_name = name
    ds = init(name)
    i = 0
    meta_items = get_unprocessed()
    while len(meta_items) > 0:
        should_break = False
        for meta in meta_items:
            i += 1
            last_item = meta['id']
            print(f"✨ Processing {last_item}: {meta['processed_storage_id']}")
            # print(meta)
            buffer.append({
                'id': meta['id'],
                'processed_storage_id': meta['processed_storage_id'],
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
