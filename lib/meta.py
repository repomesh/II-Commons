from lib.utilitas import read_file, read_json
import json
import os
import pandas


def parse_dict_parquet(filename):
    return [line.to_dict() for line in pandas.read_parquet(
        filename, engine='pyarrow'
    ).iterrows()]


def parse_jsonl(filename):
    return [json.loads(line) for line in read_file(filename)]


def parse_wiki_featured(folder):
    text_meta_path = os.path.join(folder, 'text.json')
    img_meta_path = os.path.join(folder, 'img/meta.json')
    text_meta = json.loads(read_json(text_meta_path))
    img_meta_s = json.loads(read_json(img_meta_path))['img_meta']
    return list(map(lambda im: {'text': text_meta, 'image': im}, img_meta_s))


def parse_tube_parquet(filename):
    return [{'id': id, **line.to_dict()} for id, line in pandas.read_parquet(
        filename, engine='pyarrow'
    ).iterrows()]


__all__ = [
    'parse_dict_parquet',
    'parse_jsonl',
    'parse_tube_parquet',
    'parse_wiki_featured',
]
