from lib.config import GlobalConfig
from lib.dataset import init
from lib.gcs import download_file
from lib.meta import parse_jsonl, parse_dict_parquet, parse_wiki_featured, parse_tube_parquet
from lib.s3 import exists, get_address_by_key, upload_file
from lib.utilitas import download, json_dumps, sha256, get_file_type
import os
import tempfile
import time

BATCH_SIZE = 1
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
        s3_key = ds.get_s3_key(meta)
        match dataset_name:
            case 'arxiv':
                subfix = 'pdf'
            case _:
                subfix = 'jpg'
        filename = os.path.join(temp_path.name, f"{meta['hash']}.{subfix}")
        try:
            match dataset_name:
                case 'arxiv':
                    if exists(s3_key):
                        meta['origin_storage_id'] = get_address_by_key(s3_key)
                        print(f"Skipping download: {meta['origin_storage_id']}")
                    else:
                        arr_pid = meta['paper_id'].split('/')
                        if len(arr_pid) == 1:
                            gcs_url = f"gs://arxiv-dataset/arxiv/arxiv/pdf/{meta['paper_id'].split('.')[0]}/{meta['paper_id']}{meta['versions'][-1]}.pdf"
                        elif len(arr_pid) == 2:
                            gcs_url = f"gs://arxiv-dataset/arxiv/{arr_pid[0]}/pdf/{arr_pid[1][:4]}/{arr_pid[1]}{meta['versions'][-1]}.pdf"
                        else:
                            print(meta)
                            raise ValueError('Invalid paper_id.')
                        try:
                            # hack:
                            os.environ['GCS_BUCKET'] = "arxiv-dataset"
                            download_file(gcs_url, filename)
                            print(f"Downloaded {gcs_url} to: {filename}")
                        except Exception as e:
                            print(
                                f"Fownload failed from GCS: {gcs_url}, try direct download..."
                            )
                            download(meta['url'], filename)
                            print(f"Downloaded {meta['url']} to: {filename}")
                        if get_file_type(filename) != 'PDF':
                            raise ValueError('Unexpected file type.')
                        meta['origin_storage_id'] = upload_file(filename, s3_key)
                        print(f"Uploaded to S3: {meta['origin_storage_id']}")
                case _:
                    if meta['url'].endswith('.jpg'):
                        print(f"Skipping download: {meta['url']}")
            ds.insert(meta)
            print(f'🔥 Inserted meta: {snapshot}')
        except Exception as e:
            print(f'❌ Error handling {snapshot}: {e}')
            continue
        results.append(meta)
    print('👌 Done!')
    return {'meta_items': results}


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
