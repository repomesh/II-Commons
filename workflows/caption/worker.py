from lib.caption import BATCH_SIZE, caption_image
from lib.config import GlobalConfig
from lib.dataset import init
from lib.s3 import get_url_by_key

last_item, limit = 0, 0
dataset_name = None
ds = None
buffer = []


def get_unprocessed():
    return ds.query(
        f'SELECT id, processed_storage_id FROM {ds.get_table_name()}'
        + " WHERE id > %s AND caption_qw25vl != ''"
        + ' ORDER BY id ASC LIMIT %s', (last_item, BATCH_SIZE)
    )


def trigger(force=False):
    global buffer
    if force or len(buffer) >= BATCH_SIZE:
        if GlobalConfig.DRYRUN:
            print(f"Dryrun: {buffer}")
        else:
            caption({'meta_items': buffer})
        buffer = []


def caption(args) -> dict:
    meta_items = args['meta_items'] if type(args['meta_items']) == list \
        else [args['meta_items']]
    urls = {}
    for _, meta in enumerate(meta_items):
        urls[meta['id']] = get_url_by_key(meta['processed_storage_id'])
    print('Caption images...')
    try:
        cap_res = caption_image(meta_items)
        for i in cap_res:
            snapshot = f'[{i}] {urls[i]}'
            try:
                ds.update_by_id(i, {
                    'caption_qw25vl': cap_res[i]['caption'],
                    'caption_long_qw25vl': cap_res[i]['caption_long']
                })
                print(f'🔥 ({snapshot}) Updated caption: ' +
                    cap_res[i]['caption'])
            except Exception as e:
                print(f'❌ ({snapshot}) Error updating caption: {e}')
                print(cap_res)
    except Exception as e:
        print(f'❌ Error captioning: {e}')
    print('👌 Done!')
    return {'meta_items': meta_items}


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
