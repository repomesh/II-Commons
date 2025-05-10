from lib.config import GlobalConfig
from lib.dataset import init
from pathlib import Path
from tqdm import tqdm
import os
import shutil
import time

BATCH_SIZE = 100
ENCODE = 'utf-8'
last_item, limit, slice_size = 0, 0, 1000 * 1000 * 1000  # 1GB
buffer, file_index = '', 0


def get_unprocessed(ds):
    start_time = time.time()
    resp = ds.query(
        f'SELECT * FROM {ds.get_table_name()}'
        + ' WHERE id > %s ORDER BY id ASC LIMIT %s',
        (last_item, BATCH_SIZE)
    )
    for row in resp:
        row.pop('origin_storage_id')
        row.pop('processed_storage_id')
    print(f'👌 Done! Time taken: {time.time() - start_time:.2f} seconds')
    return resp


def get_header(item):
    return ','.join(item.keys()) + '\n'


def write_file(ds, header, output_dir):
    global buffer, file_index
    if len(buffer) > 0:
        with open(
            Path(output_dir) / f'{ds.get_table_name()}_{file_index:07}.csv', \
                'w', encoding=ENCODE) as f:
            f.write(header + buffer)
        buffer = ''
        file_index += 1


def dump_data(ds_name, path, force=False):
    global buffer, last_item, limit
    last_item, limit = -1, 0
    ds = init(ds_name)
    assert path is not None, "Path is required."
    output_dir = Path(path) / ds_name
    if force:
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)
    total_lines = ds.query(f'SELECT COUNT(*) FROM {ds.get_table_name()}')[0]['count']
    data = get_unprocessed(ds)
    header = get_header(data[0]) if len(data) > 0 else ''
    i = 0
    with tqdm(total=total_lines, desc="Processing") as pbar:
        should_break = False
        while len(data) > 0:
            for item in data:
                i += 1
                pbar.update(1)
                last_item = item['id']
                buffer += f'{item}\n'
                if limit > 0 and i >= limit:
                    should_break = True
                    break
                if len((header + buffer).encode(ENCODE)) > slice_size:
                    write_file(ds, header, output_dir)
            if should_break:
                break
            data = get_unprocessed(ds)
    write_file(ds, header, output_dir)

def run(name, path, force=False):
    print(f'⏬ Dumping {name}...')
    start_time = time.time()
    match name:
        case 'pd12m':
            dump_data('pd12m', path, force)
        case 'wikipedia_en':
            dump_data('wikipedia_en', path, force)
            dump_data('wikipedia_en_embed', path, force)
        case _:
            raise ValueError(f'Unsupported dataset: {name}')
    print(f'👌 Done! Time taken: {time.time() - start_time:.2f} seconds')


__all__ = [
    'run'
]
