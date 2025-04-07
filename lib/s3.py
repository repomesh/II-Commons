import os
import boto3
import json


S3_BUCKET = os.getenv('S3_BUCKET')
S3_ENDPOINT = os.getenv('S3_ENDPOINT')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

s3 = boto3.client(
    's3', endpoint_url=S3_ENDPOINT, aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)


def get_key_by_address(s3_key):
    return s3_key[5:][s3_key[5:].index('/') + 1:]


def get_address_by_key(s3_key):
    return f's3://{S3_BUCKET}/{s3_key}'


def ensure_key(s3_key):
    return get_key_by_address(s3_key) if s3_key.startswith('s3://') else s3_key


def get_url_by_key(s3_key):
    return f'https://s3.jhuo.ca/{S3_BUCKET}/{ensure_key(s3_key)}'


def download_file(s3_key, filename=None, log=False, parse_json=False):
    import time
    key = ensure_key(s3_key)
    start = time.time()
    if filename:
        s3.download_file(S3_BUCKET, key, filename)
    else:
        content = s3.get_object(Bucket=S3_BUCKET, Key=key)['Body'].read()
        if parse_json:
            content = json.loads(content)
    end = time.time()
    if log:
        print(f'Downloaded {s3_key} in {end - start} seconds')
    return filename if filename else content


def upload_file(filename, s3_key):
    s3_key = ensure_key(s3_key)
    s3.upload_file(filename, S3_BUCKET, s3_key)
    return get_address_by_key(s3_key)


def head(s3_key):
    return s3.head_object(Bucket=S3_BUCKET, Key=s3_key)


def exists(s3_key):
    try:
        head(ensure_key(s3_key))
        return True
    except:
        return False


__all__ = [
    'download_file',
    'ensure_key',
    'exists',
    'get_address_by_key',
    'get_key_by_address',
    'get_url_by_key',
    'head',
    's3',
    'upload_file',
]
