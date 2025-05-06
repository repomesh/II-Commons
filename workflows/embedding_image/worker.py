from lib.config import GlobalConfig
from lib.coordination import heartbeat
from lib.dataset import init
from lib.embedding import BATCH_SIZE, encode_image
from lib.preprocess import process
from lib.s3 import download_file, upload_file
from lib.utilitas import json_dumps, sha256, write_image
import os
import sys
import tempfile
import time

last_item, limit = 0, 0
dataset_name = None
ds = None
buffer = []


def get_unprocessed(name):
    global last_item
    worker_count, worker_order, reset = heartbeat(name)
    if reset:
        last_item = 0
    # worker_count, worker_order = 1, 0
    where_conditions = ['(processed_storage_id = %s OR vector IS NULL)']
    params = ['']
    if worker_count > 1:
        where_conditions.append('id %% %s = %s')
        params.extend([worker_count, worker_order])
    resp = ds.query(f'SELECT * FROM {ds.get_table_name()}'
                 + ' WHERE ' + ' AND '.join(where_conditions)
                 + ' AND id > %s ORDER BY id ASC LIMIT %s',
                 tuple(params + [last_item, BATCH_SIZE]))
    res = []
    for item in resp:
        nItem = item.copy()
        for field in item.keys():
            if field.startswith('vector'):
                nItem.pop(field, None)
        res.append(nItem)
    return res


def trigger(force=False):
    global buffer
    if force or len(buffer) >= BATCH_SIZE:
        if GlobalConfig.DRYRUN:
            print(f"Dryrun: {buffer}")
        else:
            embedding({'meta_items': buffer})
        buffer = []


def embedding(args) -> dict:
    meta_items = args['meta_items'] if type(args['meta_items']) == list \
        else [args['meta_items']]
    task_hash = sha256(json_dumps(meta_items))
    temp_path = tempfile.TemporaryDirectory(suffix=f'-{task_hash}')
    images = []
    for meta in meta_items:
        snapshot = ds.snapshot(meta)
        print(f'✨ Processing item: {snapshot}')
        match dataset_name:
            case 'alpha':
                s3_address = meta['processed_storage_id']
            case _:
                s3_address = meta['origin_storage_id']
        match dataset_name:
            case 'arxiv':
                filename = os.path.join(temp_path.name, f"{meta['hash']}.pdf")
            case _:
                filename = os.path.join(temp_path.name, f"{meta['hash']}.jpg")
        try:
            download_file(s3_address, filename)
            print(f'Downloaded {s3_address} to: {filename}')
            ps_result = process(filename)
            id = meta['id']
            if dataset_name == 'alpha':
                meta['processed_storage_id'] = s3_address
            # alpha does not need processed image
            elif dataset_name != 'alpha' and ps_result['processed']:
                subfix = '.processed.jpg'
                sub_name = f"{meta['hash']}{subfix}"
                filename = os.path.join(temp_path.name, sub_name)
                write_image(ps_result['processed_image'], filename)
                s3_key = f'{s3_address}{subfix}'
                s3_address = upload_file(filename, s3_key)
                print(f"Uploaded processed image to S3: {s3_address}")
                meta = ps_result['meta']
                meta['processed_storage_id'] = s3_address
            else:
                meta = {'processed_storage_id': meta['origin_storage_id']}
            if not os.path.exists(filename) and not s3_address:
                raise Exception(f'Download / upload failed.')
            images.append({
                'image': ps_result['processed_image'],
                'id': id, 'meta': meta,
            })
        except Exception as e:
            print(f'❌ ({snapshot}) {e}')
            continue
    meta_items, end_res = [], None
    try:
        print('Embedding images...')
        snapshot = json_dumps([img['id'] for img in images])
        end_res = encode_image([img['image'] for img in images])
    except Exception as e:
        print(f'❌ ({snapshot}) Error embedding: {e}')
    if end_res is not None:
        for i, img in enumerate(images):
            snapshot = img['meta']['processed_storage_id']
            img['meta']['vector'] = end_res[i].tolist()
            try:
                if dataset_name == 'alpha':
                    del img['meta']['hash']
                    del img['meta']['origin_storage_id']
                    del img['meta']['processed_storage_id']
                # print(img['id'], img['meta'])
                ds.update_by_id(img['id'], img['meta'])
                print(f'🔥 ({snapshot}) Updated meta.')
                del img['meta']['vector']
                meta_items.append({'id': img['id'], **img['meta']})
            except Exception as e:
                print(f'❌ ({snapshot}) Error updating meta: {e}')
                print(img['meta'])
    else:
        print('❌ No embedding result.')
    print('👌 Done!')
    return {'meta_items': meta_items}


def run(name):
    global buffer, last_item, dataset_name, ds
    i = 0
    dataset_name = name
    try:
        ds = init(dataset_name)
    except Exception as e:
        print(f"❌ Unable to init src-dataset: {dataset_name}. Error: {e}")
        sys.exit(1)
    meta_items = get_unprocessed(dataset_name)
    while len(meta_items) > 0:
        should_break = False
        for meta in meta_items:
            i += 1
            last_item = meta['id']
            meta_snapshot = ds.snapshot(meta)
            print(f"Processing row {i} - {last_item}: {meta_snapshot}")
            # print(meta)
            meta['hash'] = meta['hash'] if meta.get('hash') else sha256(meta['url'])
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
        meta_items = get_unprocessed(dataset_name)
    trigger(force=True)
    print('All Done!')


__all__ = [
    'run'
]
