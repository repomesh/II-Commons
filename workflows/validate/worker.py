from lib.config import GlobalConfig
from lib.coordination import heartbeat
from lib.dataset import init
from lib.gcs import download_file as download_gcs_file
from lib.s3 import exists, upload_file, download_file as download_s3_file
from lib.utilitas import download, json_dumps, sha256, get_file_type
import os
import sys
import tempfile
import time

BATCH_SIZE = 100
dataset_name = None
ds = None
buffer = []


def get_unprocessed():
    worker_count, worker_order, _ = heartbeat(dataset_name)
    # worker_count, worker_order = 1, 0
    where_conditions = ['validated = FALSE']
    params = []
    if worker_count > 1:
        where_conditions.append('id %% %s = %s')
        params.extend([worker_count, worker_order])
    resp = ds.query(f'SELECT * FROM {ds.get_table_name()}'
                    + ' WHERE ' + ' AND '.join(where_conditions)
                    + ' ORDER BY id ASC LIMIT %s',
                    tuple(params + [BATCH_SIZE]))
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
            download_data({'meta_items': buffer})
        buffer = []


def sleep(seconds=1):
    print(f'⏰ Sleeping for {seconds} second(s) to avoid rate limit...')
    time.sleep(seconds)


def download_data(args) -> dict:
    meta_items = args['meta_items'] if type(args['meta_items']) == list \
        else [args['meta_items']]
    task_hash = sha256(json_dumps(meta_items))
    temp_path = tempfile.TemporaryDirectory(suffix=f'-{task_hash}')
    results = []
    for meta in meta_items:
        snapshot = ds.snapshot(meta)
        print(f'✨ Processing item: {snapshot}')
        current_s3_key = meta.get('origin_storage_id')
        s3_key = meta['origin_storage_id'] if (
            current_s3_key is not None and current_s3_key != ''
        ) else ds.get_s3_key(meta)
        match dataset_name:
            case 'arxiv':
                subfix = 'pdf'
            case _:
                subfix = 'jpg'
        filename = os.path.join(
            temp_path.name, f"{sha256(meta['url'])}.{subfix}")
        from_s3 = False
        try:
            if exists(s3_key):
                download_s3_file(s3_key, filename)
                print(f"Downloaded from S3: {s3_key} to: {filename}")
                from_s3 = True
            else:
                match dataset_name:
                    case 'arxiv':
                        arr_pid = meta['paper_id'].split('/')
                        if len(arr_pid) == 1:
                            gcs_url = f"gs://arxiv-dataset/arxiv/arxiv/pdf/{meta['paper_id'].split('.')[0]}/{meta['paper_id']}{meta['versions'][-1]}.pdf"
                        elif len(arr_pid) == 2:
                            gcs_url = f"gs://arxiv-dataset/arxiv/{arr_pid[0]}/pdf/{arr_pid[1][:4]}/{arr_pid[1]}{meta['versions'][-1]}.pdf"
                        else:
                            raise ValueError('Invalid paper_id.')
                        try:
                            # hack:
                            os.environ['GCS_BUCKET'] = "arxiv-dataset"
                            download_gcs_file(gcs_url, filename)
                            print(
                                f"Downloaded from GCS: {gcs_url} to: {filename}")
                        except Exception as e:
                            print(
                                f"Fownload failed from GCS: {gcs_url}, try direct download..."
                            )
                            download(meta['url'], filename)
                            print(
                                f"Downloaded from source: {meta['url']} to: {filename}")
                    case _:
                        download(meta['url'], filename)
                        print(
                            f"Downloaded from source: {meta['url']} to: {filename}")
                        meta['origin_storage_id'] = upload_file(
                            filename, s3_key)
                        print(f"Uploaded to S3: {meta['origin_storage_id']}")
            to_update = {'validated': True}
            match dataset_name:
                case 'arxiv':
                    if get_file_type(filename) == 'PDF':
                        if from_s3:
                            print(
                                f"✅ File already exists in S3: {s3_key} and is valid.")
                        else:
                            to_update['origin_storage_id'] = upload_file(
                                filename, s3_key)
                            print(
                                f"✅ Uploaded to S3: {meta['origin_storage_id']}")
                    else:
                        to_update['origin_storage_id'] = ''
            ds.update_by_id(meta['id'], to_update)
            print(f'🔥 Updated meta: {snapshot}')
        except Exception as e:
            print(f'❌ Error handling {snapshot}: {e}')
            continue
        results.append(meta)
    print('👌 Done!')
    return {'meta_items': results}


def run(name):
    global buffer,  dataset_name, ds
    dataset_name = name
    try:
        ds = init(dataset_name)
    except Exception as e:
        print(f"❌ Unable to init src-dataset: {dataset_name}. Error: {e}")
        sys.exit(1)
    meta_items = get_unprocessed()
    while len(meta_items) > 0:
        should_break = False
        for meta in meta_items:
            meta_snapshot = ds.snapshot(meta)
            print(f"Processing row {meta['id']}: {meta_snapshot}")
            # print(meta)
            buffer.append(meta)
            trigger()
        if should_break:
            break
        meta_items = get_unprocessed()
    trigger(force=True)
    print('All Done!')


__all__ = [
    'run'
]
