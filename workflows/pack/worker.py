from pathlib import Path
from tqdm import tqdm
import hashlib
import json
import os
import shutil
import tarfile
import time

ENCODING = 'utf-8'
MAX_BYTES = 1000 * 1000 * 1000  # 1GB

def get_file_name(path: str, index: int):
    base_path = Path(path) / f'{index:07}'
    return base_path.with_suffix('.csv'), \
        base_path.with_suffix('.tar'), \
        base_path.with_suffix('.json')


def compress_to_tar(tar_path, csv_path):
    with tarfile.open(tar_path, 'w') as tar:
        tar.add(csv_path, arcname=csv_path.name)


def finalize_shard(csv_path, tar_path, json_path):
    with open(csv_path, 'r', encoding=ENCODING) as cf:
        lines = cf.readlines()
        num_rows = len(lines) - 1
        header_row = lines[0].strip().split(',')
        first_row = dict(zip(header_row, lines[1].strip().split(','))) if num_rows > 0 else {}
        last_row = dict(zip(header_row, lines[-1].strip().split(','))) if num_rows > 0 else {}
    md5 = hashlib.md5()
    with open(csv_path, 'rb') as cf:
        for chunk in iter(lambda: cf.read(8192), b''):
            md5.update(chunk)
    index = {
        "filename": tar_path.name,
        "csv_inside_tar": csv_path.name,
        "num_rows": num_rows,
        "md5": md5.hexdigest(),
        "header": header_row,
        "first_row": first_row,
        "last_row": last_row
    }
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(index, jf, ensure_ascii=False, indent=2)
    compress_to_tar(tar_path, csv_path)
    os.remove(csv_path)


def pack_data(input, output, force=False):
    input = (input or '').strip()
    output = (output or '').strip()
    assert len(input) > 2, 'Input is required.'
    assert len(output) > 2 , 'Output is required.'
    if force:
        shutil.rmtree(output)
    os.makedirs(output, exist_ok=False)
    total_bytes = os.path.getsize(input)
    file_index = 0
    with open(input, "r", encoding=ENCODING) as f:
        header = f.readline()
        csv_path, tar_path, json_path = get_file_name(output, file_index)
        current_file = open(csv_path, 'w', encoding=ENCODING)
        current_file.write(header)
        current_size = current_file.tell()
        pbar = tqdm(total=total_bytes, desc='Processing (by bytes)')
        bytes_read = len(header.encode(ENCODING))
        pbar.update(bytes_read)
        for line in f:
            line_size = len(line.encode(ENCODING))
            if current_size + line_size > MAX_BYTES:
                current_file.close()
                finalize_shard(csv_path, tar_path, json_path)
                file_index += 1
                csv_path, tar_path, json_path = get_file_name(output, file_index)
                current_file = open(csv_path, 'w', encoding=ENCODING)
                current_file.write(header)
                current_size = current_file.tell()
            current_file.write(line)
            current_size += line_size
            bytes_read += line_size
            pbar.update(line_size)
        current_file.close()
        finalize_shard(csv_path, tar_path, json_path)
        pbar.close()


def run(input, output, force=False):
    print(f'📦 Packing {input}...')
    start_time = time.time()
    pack_data(input, output, force)
    print(f'👌 Done! Time taken: {time.time() - start_time:.2f} seconds')


__all__ = [
    'run'
]
