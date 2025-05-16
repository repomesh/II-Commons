from .wikipedia import load as load_wikipedia
from lib.config import GlobalConfig
from lib.dataset import init
from lib.meta import parse_jsonl, parse_dict_parquet, parse_wiki_featured, parse_tube_parquet
from lib.psql import batch_insert, enrich_data
import os
import time

BATCH_SIZE = 1000
dataset_name = None
ds = None
buffer = []


def trigger(force=False):
    global buffer
    if len(buffer) >= (1 if force else BATCH_SIZE):
        if GlobalConfig.DRYRUN:
            print(f"Dryrun: {buffer}")
        else:
            insert_data({'meta_items': buffer})
        buffer = []


def sleep(seconds=1):
    print(f'⏰ Sleeping for {seconds} second(s) to avoid rate limit...')
    time.sleep(seconds)


def insert_data(args) -> dict:
    # dataset patch:
    # sql = f"""UPDATE {ds.get_table_name()} SET
    #     origin_source = %s, license = %s WHERE url = %s"""
    # meta_items = args['meta_items'] if type(args['meta_items']) == list \
    #     else [args['meta_items']]
    # to_update = []
    # for meta in meta_items:
    #     to_update.append((
    #         meta['origin_source'],
    #         meta['license'],
    #         meta['url']
    #     ))
    # batch_insert(sql, to_update)
    # print(f'🔥 Upserted meta: {len(meta_items)} items.')
    # return {'meta_items': meta_items}
    sql = f"""INSERT INTO {ds.get_table_name()} (url, hash, caption,
        caption_long, origin_source, origin_hash, origin_width, origin_height,
        exif, meta, source, license) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s) ON CONFLICT (url) DO NOTHING"""
    meta_items = args['meta_items'] if type(args['meta_items']) == list \
        else [args['meta_items']]
    urls = []
    for meta in meta_items:
        urls.append(meta['url'])
    chk_res = ds.query(
        f"SELECT count(*) FROM {ds.get_table_name()} WHERE url IN ({', '.join(['%s'] * len(urls))})",
        urls
    )
    if chk_res[0]['count'] != len(urls):
        to_insert = []
        for meta in meta_items:
            meta = enrich_data({**meta, 'source': [meta['source']]})
            to_insert.append((
                meta['url'],
                meta['hash'],
                meta['caption'],
                meta['caption_long'],
                meta['origin_source'],
                meta['origin_hash'],
                meta['origin_width'],
                meta['origin_height'],
                meta['exif'],
                meta['meta'],
                meta['source'],
                meta['license'],
            ))
        batch_insert(sql, to_insert)
        print(f'🔥 Upserted meta: {len(to_insert)} items.')
    else:
        print(f'👌 Skipping existing {len(urls)} items.')
    return {'meta_items': meta_items}


def run(name, meta_path):
    global buffer, ds, dataset_name
    i, last_item, limit = 0, 0, 0
    dataset_name = name
    ds = init(dataset_name)
    match name:
        case 'wikipedia_en':
            return load_wikipedia(ds, meta_path)
    meta_files = []
    if os.path.isdir(meta_path):
        for meta_file in sorted(os.listdir(meta_path)):
            meta_files.append(os.path.join(meta_path, meta_file))
    else:
        meta_files.append(meta_path)
    for meta_file in meta_files:
        if dataset_name == 'wikipedia_featured' and os.path.isdir(meta_file):
            meta_items = parse_wiki_featured(meta_file)
        elif dataset_name in ['megalith_10m', 'pd12m']:
            meta_items = parse_tube_parquet(meta_file)
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
            print(f"Processing row {i}: {meta_snapshot}")
            buffer.append(meta)
            trigger()
            if limit > 0 and i >= limit:
                break
    trigger(force=True)
    print('👌 Done!')


__all__ = [
    'run'
]
