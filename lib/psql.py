from lib.embedding import DIMENSION
from lib.utilitas import Empty
from lib.s3 import get_url_by_key
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool
from psycopg.types.json import Jsonb
from psycopg.errors import UniqueViolation
import datetime
import os
import random
import time

POSTGRES_HOST_1 = os.getenv('POSTGRES_HOST_1')
POSTGRES_PORT_1 = os.getenv('POSTGRES_PORT_1')
POSTGRES_USER_1 = os.getenv('POSTGRES_USER_1')
POSTGRES_PASSWORD_1 = os.getenv('POSTGRES_PASSWORD_1')
POSTGRES_DB_1 = os.getenv('POSTGRES_DB_1')

POSTGRES_HOST_2 = os.getenv('POSTGRES_HOST_2')
POSTGRES_PORT_2 = os.getenv('POSTGRES_PORT_2')
POSTGRES_USER_2 = os.getenv('POSTGRES_USER_2')
POSTGRES_PASSWORD_2 = os.getenv('POSTGRES_PASSWORD_2')
POSTGRES_DB_2 = os.getenv('POSTGRES_DB_2')

DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
VACUUM_CHANCE = 100000
EMPTY_OBJECT = '{}'


def configure(conn):
    register_vector(conn)


pool_1 = ConnectionPool(
    conninfo=f'postgresql://{POSTGRES_USER_1}:{POSTGRES_PASSWORD_1}@{POSTGRES_HOST_1}:{POSTGRES_PORT_1}/{POSTGRES_DB_1}',
    open=True, configure=configure, min_size=3, max_size=100000
)

pool_2 = ConnectionPool(
    conninfo=f'postgresql://{POSTGRES_USER_2}:{POSTGRES_PASSWORD_2}@{POSTGRES_HOST_2}:{POSTGRES_PORT_2}/{POSTGRES_DB_2}',
    open=True, configure=configure, min_size=3, max_size=100000
)


def execute(pool_id, sql, values=None, log=False, autocommit=True, batch=False):
    match pool_id:
        case 'pool_1':
            pool = pool_1
        case 'pool_2':
            pool = pool_2
        case _:
            raise ValueError('Invalid pool ID')
    with pool.connection() as conn:
        if log or DEBUG:
            render_value = f' w/ {values}' if values else ''
            print(f'Executing: {sql}{render_value}')
        str_time = time.time()
        if batch:
            conn.autocommit = False
            cursor = None
            for value in values:
                conn.execute(sql, value)
        else:
            conn.autocommit = autocommit
            cursor = conn.execute(sql, values)
        if not conn.autocommit:
            conn.commit()
        end_time = time.time()
        if log or DEBUG:
            print(f'>>> Execution time: {end_time - str_time:.2f} seconds.')
        return cursor


def ensure_vector_extension(pool_id):
    sql = 'CREATE EXTENSION IF NOT EXISTS vector'
    res = execute(pool_id, sql)
    # print(f'Setup: {sql} => {res.statusmessage}')


def check_dataset(dataset):
    if not dataset:
        raise ValueError('`dataset` is required.')


def vacuum_table(pool_id, table_name, force=False):
    if force or random.randint(1, VACUUM_CHANCE) == 1:
        return execute(pool_id, f'VACUUM {table_name}', log=True)


def get_table_name(dataset, materialized=False):
    check_dataset(dataset)
    head, tail = 'sc' if materialized else 'ds', ''
    # @todo: temp fix for arxiv_oai
    if dataset == 'arxiv_oai':
        dataset = 'arxiv'
    if dataset == 'ms_marco':
        return 'ms_marco'
    match dataset:
        case 'alpha':
            head = 'ii'
        case 'xxx_yyy':
            tail = 'v3'
        case 'text_0000001_en' | 'wikipedia_en' | 'arxiv':
            head = 'ts'
    table_name = f'{head}_{dataset}' + (f'_{tail}' if tail else '')
    # todo: disabled for now by @Leask for speed up SSCD
    # vacuum_table(table_name)
    return table_name


def truncate(pool_id, dataset, force=False, materialized=False):
    assert_materialized(materialized)
    assert force, 'Make sure you know what you are doing!'
    table_name = get_table_name(dataset)
    return execute(pool_id, f'TRUNCATE {table_name}', log=True)


def generate_empty_vector(dim=DIMENSION):
    return f"'{','.join(['0'] * dim)}'::vector"


def init(pool_id, dataset, materialized=False):
    table_name = get_table_name(dataset, materialized=materialized)
    if dataset == 'wikipedia_en':
        list_sql = []
    elif dataset == 'text_0000001_en':
        list_sql = []
    elif dataset == 'arxiv':
        list_sql = []
    elif dataset == 'arxiv_oai':
        list_sql = []
    elif dataset == 'ms_marco':
        list_sql = []
    elif dataset == 'alpha':
        list_sql = [
            f'CREATE UNIQUE INDEX IF NOT EXISTS {table_name}_url_index ON {table_name} (url)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_origin_width_index ON {table_name} (origin_width)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_origin_height_index ON {table_name} (origin_height)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_origin_storage_id_index ON {table_name} (origin_storage_id)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_processed_storage_id_index ON {table_name} (processed_storage_id)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_processed_width_index ON {table_name} (processed_width)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_processed_height_index ON {table_name} (processed_height)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_aspect_ratio_index ON {table_name} (aspect_ratio)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_exif_index ON {table_name} USING gin(exif)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_meta_index ON {table_name} USING gin(meta)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_source_index ON {table_name} USING gin(source)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_created_at_index ON {table_name} (created_at)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_updated_at_index ON {table_name} (updated_at)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_vector_index ON {table_name} USING hnsw(vector vector_cosine_ops)',
        ]
    elif materialized:
        list_sql = [
            f'CREATE INDEX IF NOT EXISTS {table_name}_id_index ON {table_name} (id)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_url_index ON {table_name} (url)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_origin_width_index ON {table_name} (origin_width)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_origin_height_index ON {table_name} (origin_height)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_origin_storage_id_index ON {table_name} (origin_storage_id)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_processed_storage_id_index ON {table_name} (processed_storage_id)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_processed_width_index ON {table_name} (processed_width)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_processed_height_index ON {table_name} (processed_height)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_aspect_ratio_index ON {table_name} (aspect_ratio)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_exif_index ON {table_name} USING gin(exif)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_meta_index ON {table_name} USING gin(meta)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_vector_index ON {table_name} USING hnsw(vector vector_cosine_ops)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_created_at_index ON {table_name} (created_at)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_updated_at_index ON {table_name} (updated_at)',
        ]
    else:
        list_sql = [
            f"""CREATE TABLE IF NOT EXISTS {table_name} (
                id BIGSERIAL PRIMARY KEY,
                url VARCHAR NOT NULL,
                hash VARCHAR(1024) NOT NULL,
                caption VARCHAR NOT NULL DEFAULT '',
                caption_long VARCHAR NOT NULL DEFAULT '',
                origin_hash VARCHAR(1024) NOT NULL DEFAULT '',
                origin_width BIGINT NOT NULL DEFAULT 0,
                origin_height BIGINT NOT NULL DEFAULT 0,
                origin_storage_id VARCHAR(1024) NOT NULL DEFAULT '',
                processed_storage_id VARCHAR(1024) NOT NULL DEFAULT '',
                processed_width BIGINT NOT NULL DEFAULT 0,
                processed_height BIGINT NOT NULL DEFAULT 0,
                aspect_ratio DOUBLE PRECISION NOT NULL DEFAULT 0,
                exif JSONB NOT NULL DEFAULT '{EMPTY_OBJECT}',
                meta JSONB NOT NULL DEFAULT '{EMPTY_OBJECT}',
                vector VECTOR(768) DEFAULT NULL,
                similarity FLOAT NOT NULL DEFAULT 0,
                similar_to INT NOT NULL DEFAULT 0,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""",
            f'CREATE UNIQUE INDEX IF NOT EXISTS {table_name}_url_index ON {table_name} (url)',
            f'CREATE UNIQUE INDEX IF NOT EXISTS {table_name}_hash_index ON {table_name} (hash)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_origin_hash_index ON {table_name} (origin_hash)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_origin_width_index ON {table_name} (origin_width)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_origin_height_index ON {table_name} (origin_height)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_origin_storage_id_index ON {table_name} (origin_storage_id)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_processed_storage_id_index ON {table_name} (processed_storage_id)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_processed_width_index ON {table_name} (processed_width)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_processed_height_index ON {table_name} (processed_height)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_aspect_ratio_index ON {table_name} (aspect_ratio)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_exif_index ON {table_name} USING gin(exif)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_meta_index ON {table_name} USING gin(meta)',
            # todo // drop vector index temporarily
            # f'CREATE INDEX IF NOT EXISTS {table_name}_vector_index ON {table_name} USING hnsw(vector vector_cosine_ops)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_similarity_index ON {table_name} (similarity)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_similar_to_index ON {table_name} (similar_to)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_created_at_index ON {table_name} (created_at)',
            f'CREATE INDEX IF NOT EXISTS {table_name}_updated_at_index ON {table_name} (updated_at)',
        ]
    result = []
    for sql in list_sql:
        res = execute(pool_id, sql)
        result.append(res)
        # print(f'Init: {sql} => {res.statusmessage}')
    return result


def enrich_data(data):
    data = data.copy()  # prevent mutation
    if data.get('exif') is not None and data['exif'] != '':
        data['exif'] = Jsonb(data['exif'])
    if data.get('meta') is not None and data['meta'] != '':
        data['meta'] = Jsonb(data['meta'])
    if data.get('source') is not None and data['source'] != '':
        data['source'] = Jsonb(data['source'])
    if data.get('categories') is not None and data['categories'] != '':
        data['categories'] = Jsonb(data['categories'])
    if data.get('versions') is not None and data['versions'] != '':
        data['versions'] = Jsonb(data['versions'])
    if data.get('authors') is not None and data['authors'] != '':
        data['authors'] = Jsonb(data['authors'])
    if data.get('submitter') is not None and data['submitter'] != '':
        data['submitter'] = Jsonb(data['submitter'])
    if data.get('updated_at') is None:
        data['updated_at'] = datetime.datetime.now()
    return data


def insert(pool_id, dataset, data, deplicate_ignore=[], tail='', materialized=False):
    assert_materialized(materialized)
    table_name = get_table_name(dataset)
    if not data:
        raise ValueError('`data` is required.')
    if type(data) is not list:
        data = [data]
    list_sql = []
    if deplicate_ignore:
        tail = f" ON CONFLICT ({', '.join(deplicate_ignore)}) DO NOTHING {tail}"
    for d in data:
        items = list(enrich_data(d).items())
        columns = ', '.join(key for key, _ in items)
        placeholders = ', '.join(['%s'] * len(items))
        values = tuple(value for _, value in items)
        list_sql.append([
            f'INSERT INTO {table_name} ({columns}) VALUES ({placeholders}) {tail}',
            values
        ])
    result = []
    for sql in list_sql:
        result.append(query(pool_id, sql[0], sql[1]))
    return result


def hash_exists(pool_id, dataset, hash, materialized=False):
    table_name = get_table_name(dataset, materialized=materialized)
    result = query(
        pool_id, f'SELECT hash FROM {table_name} WHERE hash = %s', (hash,)
    )
    return len(result) > 0


def url_exists(pool_id, dataset, url, materialized=False):
    table_name = get_table_name(dataset, materialized=materialized)
    result = query(
        pool_id, f'SELECT url FROM {table_name} WHERE url = %s', (url,)
    )
    return len(result) > 0


def snapshot(meta):
    if meta.get('processed_storage_id'):
        return get_url_by_key(meta.get('processed_storage_id'))
    elif meta.get('origin_storage_id'):
        return get_url_by_key(meta.get('origin_storage_id'))
    elif meta.get('url'):
        return meta.get('url')
    elif meta.get('id'):
        return meta.get('id')
    else:
        raise ValueError('No valid identifier found.')


def assert_materialized(materialized=False):
    assert not materialized, 'Materialized dataset is read-only.'


def update_by_id(pool_id, dataset, id, data, deplicate_ignore=[], tail='', materialized=False):
    assert_materialized(materialized)
    table_name = get_table_name(dataset)
    if not id:
        raise ValueError('`id` is required.')
    if not data:
        raise ValueError('`data` is required.')
    if type(id) is list and type(data) is list:
        if len(id) != len(data):
            raise ValueError('`id` and `data` must have the same length.')
    elif type(id) is not list and type(data) is not list:
        id = [id]
        data = [data]
    else:
        raise ValueError('`id` and `data` must have the same length.')
    list_sql = []
    result = []
    for i, d in zip(id, data):
        items = list(enrich_data(d).items())
        columns = ', '.join(key for key, _ in items)
        placeholders = ', '.join(['%s'] * len(items))
        values = tuple(value for _, value in items)
        list_sql.append([
            f'UPDATE {table_name} SET ({columns}) = ({placeholders}) WHERE id = %s {tail}',
            values + (i,)
        ])
    for sql in list_sql:
        resp, err = None, None
        try:
            resp = query(pool_id, sql[0], sql[1])
        except UniqueViolation as e:
            err = e
            if deplicate_ignore:
                print(e)
            else:
                raise e
        except Exception as e:
            raise e
        result.append(resp or err)
    return result


def get_unprocessed(pool_id, dataset, limit=10, offset=0):
    table_name = get_table_name(dataset)
    resp = query(pool_id,
                 f'SELECT * FROM {table_name}'
                 + ' WHERE (processed_storage_id = %s OR vector_siglip IS NULL)'
                 + ' AND id > %s ORDER BY id ASC LIMIT %s',
                 ('', offset, limit)
                 )
    res = []
    for item in resp:
        for field in item.keys():
            if field.startswith('vector'):
                nItem = item.copy()
                nItem.pop(field, None)
                res.append(nItem)
    return res


def batch_insert(pool_id, sql=None, values=None, **kwargs):
    return execute(pool_id, sql, values, batch=True, **kwargs)


def query(pool_id, sql=None, values=None, **kwargs):
    cursor = execute(pool_id, sql, values, **kwargs)
    try:
        columns = [desc[0] for desc in cursor.description]
        results = cursor.fetchall()
        return [dict(zip(columns, row)) for row in results]
    except Exception as e:
        pass
    return cursor


def get_dataset(pool_id, dataset, materialized=False):
    check_dataset(dataset)
    ds = Empty()

    def init_instant():
        return init(pool_id, dataset, materialized=ds.materialized)

    def get_table_name_instant():
        return get_table_name(dataset, materialized=ds.materialized)

    def url_exists_instant(url):
        return url_exists(pool_id, dataset, url, materialized=ds.materialized)

    def insert_instant(data, deplicate_ignore=[], tail=''):
        return insert(
            pool_id, dataset, data, deplicate_ignore=deplicate_ignore,
            tail=tail, materialized=ds.materialized
        )

    def update_by_id_instant(id, data, deplicate_ignore=[], tail=''):
        return update_by_id(
            pool_id, dataset, id, data, deplicate_ignore=deplicate_ignore,
            tail=tail, materialized=ds.materialized
        )

    def execute_instant(sql, values=None, **kwargs):
        return execute(pool_id, sql, values, **kwargs)

    def query_instant(sql, values=None, **kwargs):
        return query(pool_id, sql, values, **kwargs)

    def truncate_instant(force=False):
        return truncate(pool_id, dataset, force=force, materialized=ds.materialized)

    def get_unprocessed_instant(**kwargs):
        assert not ds.materialized, 'No unprocessed data in materialized view.'
        return get_unprocessed(pool_id, dataset, **kwargs)

    def exists(meta):
        return url_exists_instant(meta['url'])

    ds.pool_id = pool_id
    ds.materialized = materialized
    ds.execute = execute_instant
    ds.init = init_instant
    ds.exists = exists
    ds.get_table_name = get_table_name_instant
    ds.get_unprocessed = get_unprocessed_instant
    ds.insert = insert_instant
    ds.truncate = truncate_instant
    ds.query = query_instant
    ds.snapshot = snapshot
    ds.update_by_id = update_by_id_instant
    ds.url_exists = url_exists_instant
    return ds


ensure_vector_extension('pool_1')
ensure_vector_extension('pool_2')

__all__ = [
    'conn',
    'enrich_data',
    'execute',
    'generate_empty_vector',
    'get_dataset',
    'get_table_name',
    'get_unprocessed',
    'batch_insert',
    'hash_exists',
    'init',
    'insert',
    'query',
    'snapshot',
    'truncate',
    'update_by_id',
    'url_exists',
]
