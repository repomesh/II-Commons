from google.cloud import storage
import os

GCS_BUCKET = os.getenv('GCS_BUCKET')

gcs = storage.Client()


def get_key_by_address(gs_key):
    return gs_key[5:][gs_key[5:].index('/') + 1:]


def get_address_by_key(gs_key):
    return f'gs://{GCS_BUCKET}/{gs_key}'


def ensure_key(gs_key):
    return get_key_by_address(gs_key) if gs_key.startswith('gs://') else gs_key


def download_file(gcs_key, filename):
    bucket = gcs.bucket(GCS_BUCKET)
    blob = bucket.blob(ensure_key(gcs_key))
    blob.download_to_filename(filename)
    return filename


__all__ = [
    'download_file',
    'ensure_key',
    'get_address_by_key',
    'get_key_by_address',
    'gcs',
]
