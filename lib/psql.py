from lib.config import GlobalConfig
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


POSTGRES_HOST = os.getenv('POSTGRES_HOST')
POSTGRES_PORT = os.getenv('POSTGRES_PORT')
POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_DB = os.getenv('POSTGRES_DB')

DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
VACUUM_CHANCE = 100000
EMPTY_OBJECT = '{}'


def configure(conn):
    register_vector(conn)


pool = ConnectionPool(
    conninfo=f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}',
    open=True, configure=configure, min_size=3, max_size=100000
)


def execute(sql, values=None, log=False, autocommit=True, batch=False):
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


def ensure_vector_extension():
    # https://docs.vectorchord.ai/vectorchord/getting-started/overview.html
    sqls = [
        'CREATE EXTENSION IF NOT EXISTS vchord CASCADE',
        # 'SET vchordrq.probes = 100'
        # "ALTER SYSTEM SET vchordrq.prewarm_dim = '64,128,256,384,512,768,1024,1152,1536'"
    ]
    for sql in sqls:
        res = execute(sql)
        if GlobalConfig.DEBUG:
            print(f'Init: {sql} => {res.statusmessage}')
    return res


def ensure_bm25_extension():
    sql = 'CREATE EXTENSION IF NOT EXISTS pg_search'
    res = execute(sql)
    if GlobalConfig.DEBUG:
        print(f'Init: {sql} => {res.statusmessage}')
    return res


def check_dataset(dataset):
    if not dataset:
        raise ValueError('`dataset` is required.')


def vacuum_table(table_name, force=False):
    if force or random.randint(1, VACUUM_CHANCE) == 1:
        return execute(f'VACUUM {table_name}', log=True)


def get_table_name(dataset):
    check_dataset(dataset)
    head, tail = '', ''
    match dataset:
        case 'alpha':
            head = 'ii'
        case 'pd12m':
            head = 'is'
        case 'wikipedia_en' | 'wikipedia_en_embed' | 'arxiv' | 'ms_marco' | 'ms_marco_embed':
            head = 'ts'
        case 'workers':
            head = 'sc'
        case _:
            raise ValueError(f'Unsupported dataset: {dataset}')
    table_name = f'{head}_{dataset}' + (f'_{tail}' if tail else '')
    # todo: disabled for now by @Leask for speed up
    # vacuum_table(table_name)
    return table_name


def truncate(dataset, force=False):
    assert force, 'Make sure you know what you are doing!'
    table_name = get_table_name(dataset)
    return execute(f'TRUNCATE {table_name}', log=True)


def generate_empty_vector(dim=DIMENSION):
    return f"'{','.join(['0'] * dim)}'::vector"


def init(dataset):
    table_name = get_table_name(dataset)
    result, list_sql = [], []
    match dataset:
        # wikiextractor enwiki-20250201-pages-articles-multistream.xml - -json - o wiki_ext - -no-templates
        case 'wikipedia_en':
            list_sql = [
                f"""CREATE TABLE IF NOT EXISTS {table_name} (
                    id BIGSERIAL PRIMARY KEY,
                    revid BIGINT NOT NULL,
                    url VARCHAR NOT NULL,
                    title VARCHAR NOT NULL DEFAULT '',
                    origin_storage_id VARCHAR(1024) NOT NULL DEFAULT '',
                    ignored BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )""",
                f'CREATE INDEX IF NOT EXISTS {table_name}_ignored_index ON {table_name} (ignored)',
            ]
            init('wikipedia_en_embed')
        case 'wikipedia_en_embed' | 'ms_marco_embed':
            list_sql.extend([
                f"""CREATE TABLE IF NOT EXISTS {table_name} (
                    id BIGSERIAL PRIMARY KEY,
                    title VARCHAR NOT NULL,
                    url VARCHAR NOT NULL,
                    snapshot VARCHAR NOT NULL,
                    source_id BIGINT NOT NULL,
                    chunk_index BIGINT NOT NULL,
                    chunk_text VARCHAR NOT NULL,
                    vector halfvec(768) DEFAULT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )""",
                f'CREATE INDEX IF NOT EXISTS {table_name}_source_id_index ON {table_name} (source_id)',
                f'CREATE INDEX IF NOT EXISTS {table_name}_chunk_index_index ON {table_name} (chunk_index)',
                f"CREATE INDEX IF NOT EXISTS {table_name}_chunk_text_index ON {table_name} USING bm25 (id, title, chunk_text) WITH (key_field='id')",
                f'CREATE UNIQUE INDEX IF NOT EXISTS {table_name}_source_index ON {table_name} (source_id, chunk_index)',
                f"""CREATE INDEX IF NOT EXISTS {table_name}_vector_index ON {table_name} USING vchordrq (vector halfvec_cosine_ops) WITH (options = $$
                    [build.internal]
                    lists = [20000]
                    build_threads = 6
                    spherical_centroids = true
                $$)""",
                f'CREATE INDEX IF NOT EXISTS {table_name}_vector_null_index ON {table_name} (vector) WHERE vector IS NULL',
                f"SELECT vchordrq_prewarm('{table_name}_vector_index')"
            ])
        case 'arxiv':
            list_sql = [
                # todo: disabled temporary by @Leask
                # f"""CREATE TABLE IF NOT EXISTS {table_name} (
                #     id BIGSERIAL PRIMARY KEY,
                #     paper_id VARCHAR NOT NULL, -- id
                #     submitter JSONB NOT NULL DEFAULT '[]',
                #     authors JSONB NOT NULL DEFAULT '[]',
                #     title VARCHAR NOT NULL,
                #     comments VARCHAR NOT NULL DEFAULT '',
                #     journal_ref VARCHAR NOT NULL DEFAULT '', -- journal-ref
                #     doi VARCHAR NOT NULL DEFAULT '',
                #     report_no VARCHAR NOT NULL DEFAULT '', -- report-no
                #     categories JSONB NOT NULL DEFAULT '[]',
                #     versions JSONB NOT NULL DEFAULT '[]',
                #     hash VARCHAR NOT NULL, --abstract_md5
                #     license VARCHAR NOT NULL DEFAULT '',
                #     abstract VARCHAR NOT NULL DEFAULT '',
                #     url VARCHAR NOT NULL,
                #     origin_storage_id VARCHAR(1024) NOT NULL DEFAULT '',
                #     created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                #     updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                # )""",
                # f'CREATE INDEX IF NOT EXISTS {table_name}_abstract_index ON {table_name} USING gin(abstract)',
                # f'CREATE INDEX IF NOT EXISTS {table_name}_authors_index ON {table_name} USING gin(authors)',
                # f'CREATE INDEX IF NOT EXISTS {table_name}_categories_index ON {table_name} USING gin(categories)',
                # f'CREATE INDEX IF NOT EXISTS {table_name}_comments_index ON {table_name} USING gin(comments)',
                # f'CREATE INDEX IF NOT EXISTS {table_name}_created_at_index ON {table_name} (created_at)',
                # f'CREATE INDEX IF NOT EXISTS {table_name}_doi_index ON {table_name} (doi)',
                # f'CREATE INDEX IF NOT EXISTS {table_name}_journal_ref_index ON {table_name} (journal_ref)',
                # f'CREATE INDEX IF NOT EXISTS {table_name}_license_index ON {table_name} (license)',
                # f'CREATE INDEX IF NOT EXISTS {table_name}_origin_storage_id_index ON {table_name} (origin_storage_id)',
                # f'CREATE INDEX IF NOT EXISTS {table_name}_report_no_index ON {table_name} (report_no)',
                # f'CREATE INDEX IF NOT EXISTS {table_name}_submitter_index ON {table_name} USING gin(submitter)',
                # f'CREATE INDEX IF NOT EXISTS {table_name}_title_index ON {table_name} USING gin(title)',
                # f'CREATE INDEX IF NOT EXISTS {table_name}_updated_at_index ON {table_name} (updated_at)',
                # f'CREATE INDEX IF NOT EXISTS {table_name}_versions_index ON {table_name} USING gin(versions)',
                # f'CREATE UNIQUE INDEX IF NOT EXISTS {table_name}_hash_index ON {table_name} (hash)',
                # f'CREATE UNIQUE INDEX IF NOT EXISTS {table_name}_paper_id_index ON {table_name} (paper_id)',
                # f'CREATE UNIQUE INDEX IF NOT EXISTS {table_name}_url_index ON {table_name} (url)',
                # patch for arxiv dataset
                # f'ALTER TABLE ts_arxiv ADD COLUMN validated BOOLEAN NOT NULL DEFAULT FALSE;',
                # f'CREATE INDEX IF NOT EXISTS ts_arxiv_validated_index ON ts_arxiv (validated);',
            ]
        case 'alpha' | 'pd12m':
            list_sql = [
                f"""CREATE TABLE IF NOT EXISTS {table_name} (
                    id BIGSERIAL PRIMARY KEY,
                    url VARCHAR NOT NULL,
                    caption VARCHAR NOT NULL DEFAULT '',
                    caption_long VARCHAR NOT NULL DEFAULT '',
                    origin_width BIGINT NOT NULL DEFAULT 0,
                    origin_height BIGINT NOT NULL DEFAULT 0,
                    origin_storage_id VARCHAR(1024) NOT NULL DEFAULT '',
                    processed_storage_id VARCHAR(1024) NOT NULL DEFAULT '',
                    processed_width BIGINT NOT NULL DEFAULT 0,
                    processed_height BIGINT NOT NULL DEFAULT 0,
                    aspect_ratio DOUBLE PRECISION NOT NULL DEFAULT 0,
                    exif JSONB NOT NULL DEFAULT '{EMPTY_OBJECT}',
                    meta JSONB NOT NULL DEFAULT '{EMPTY_OBJECT}',
                    source JSONB NOT NULL DEFAULT '[]',
                    vector halfvec(1152) DEFAULT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )""",
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
                f"""CREATE INDEX IF NOT EXISTS {table_name}_vector_index ON {table_name} USING vchordrq (vector halfvec_l2_ops)  WITH (options = $$
                    residual_quantization = true
                    [build.internal]
                    lists = [20000]
                    build_threads = 6
                    spherical_centroids = false
                $$)""",
                f"CREATE INDEX IF NOT EXISTS {table_name}_caption_index ON {table_name} (caption) WHERE caption = ''",
                f"CREATE INDEX IF NOT EXISTS {table_name}_caption_long_index ON {table_name} (caption_long) WHERE caption_long = ''",
                f'CREATE INDEX IF NOT EXISTS {table_name}_vector_null_index ON {table_name} (vector) WHERE vector IS NULL',
                f"SELECT vchordrq_prewarm('{table_name}_vector_index')"
            ]
        case 'workers':
            list_sql = [
                f"""CREATE TABLE IF NOT EXISTS {table_name} (
                    id BIGSERIAL PRIMARY KEY,
                    uuid VARCHAR NOT NULL,
                    workflow VARCHAR NOT NULL,
                    last_heartbeat TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )""",
                f'CREATE UNIQUE INDEX IF NOT EXISTS {table_name}_uuid_index ON {table_name} (uuid)',
                f'CREATE INDEX IF NOT EXISTS {table_name}_last_heartbeat_index ON {table_name} (last_heartbeat)',
            ]
        case _:
            raise ValueError(f'Unsupported dataset: {dataset}')
            # KEEP this for SDSS algorithm
            # list_sql = [
            #     f"""CREATE TABLE IF NOT EXISTS {table_name} (
            #         id BIGSERIAL PRIMARY KEY,
            #         url VARCHAR NOT NULL,
            #         hash VARCHAR(1024) NOT NULL,
            #         caption VARCHAR NOT NULL DEFAULT '',
            #         caption_long VARCHAR NOT NULL DEFAULT '',
            #         origin_hash VARCHAR(1024) NOT NULL DEFAULT '',
            #         origin_width BIGINT NOT NULL DEFAULT 0,
            #         origin_height BIGINT NOT NULL DEFAULT 0,
            #         origin_storage_id VARCHAR(1024) NOT NULL DEFAULT '',
            #         processed_storage_id VARCHAR(1024) NOT NULL DEFAULT '',
            #         processed_width BIGINT NOT NULL DEFAULT 0,
            #         processed_height BIGINT NOT NULL DEFAULT 0,
            #         aspect_ratio DOUBLE PRECISION NOT NULL DEFAULT 0,
            #         exif JSONB NOT NULL DEFAULT '{EMPTY_OBJECT}',
            #         meta JSONB NOT NULL DEFAULT '{EMPTY_OBJECT}',
            #         vector VECTOR(1152) DEFAULT NULL,
            #         similarity FLOAT NOT NULL DEFAULT 0,
            #         similar_to INT NOT NULL DEFAULT 0,
            #         created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            #         updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            #     )""",
            #     f'CREATE UNIQUE INDEX IF NOT EXISTS {table_name}_url_index ON {table_name} (url)',
            #     f'CREATE UNIQUE INDEX IF NOT EXISTS {table_name}_hash_index ON {table_name} (hash)',
            #     f'CREATE INDEX IF NOT EXISTS {table_name}_origin_hash_index ON {table_name} (origin_hash)',
            #     f'CREATE INDEX IF NOT EXISTS {table_name}_origin_width_index ON {table_name} (origin_width)',
            #     f'CREATE INDEX IF NOT EXISTS {table_name}_origin_height_index ON {table_name} (origin_height)',
            #     f'CREATE INDEX IF NOT EXISTS {table_name}_origin_storage_id_index ON {table_name} (origin_storage_id)',
            #     f'CREATE INDEX IF NOT EXISTS {table_name}_processed_storage_id_index ON {table_name} (processed_storage_id)',
            #     f'CREATE INDEX IF NOT EXISTS {table_name}_processed_width_index ON {table_name} (processed_width)',
            #     f'CREATE INDEX IF NOT EXISTS {table_name}_processed_height_index ON {table_name} (processed_height)',
            #     f'CREATE INDEX IF NOT EXISTS {table_name}_aspect_ratio_index ON {table_name} (aspect_ratio)',
            #     f'CREATE INDEX IF NOT EXISTS {table_name}_exif_index ON {table_name} USING gin(exif)',
            #     f'CREATE INDEX IF NOT EXISTS {table_name}_meta_index ON {table_name} USING gin(meta)',
            #     f'CREATE INDEX IF NOT EXISTS {table_name}_vector_index ON {table_name} USING hnsw(vector vector_cosine_ops)',
            #     f'CREATE INDEX IF NOT EXISTS {table_name}_similarity_index ON {table_name} (similarity)',
            #     f'CREATE INDEX IF NOT EXISTS {table_name}_similar_to_index ON {table_name} (similar_to)',
            #     f'CREATE INDEX IF NOT EXISTS {table_name}_created_at_index ON {table_name} (created_at)',
            #     f'CREATE INDEX IF NOT EXISTS {table_name}_updated_at_index ON {table_name} (updated_at)',
            #     f'CREATE INDEX IF NOT EXISTS {table_name}_vector_null_index ON {table_name} (vector) WHERE vector IS NULL',
            # ]
    for sql in list_sql:
        res = execute(sql)
        result.append(res)
        if GlobalConfig.DEBUG:
            print(f'Init: {sql} => {res.statusmessage}')
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


def insert(dataset, data, deplicate_ignore=[], tail=''):
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
        result.append(query(sql[0], sql[1]))
    return result


def hash_exists(dataset, hash):
    table_name = get_table_name(dataset)
    result = query(
        f'SELECT hash FROM {table_name} WHERE hash = %s', (hash,)
    )
    return len(result) > 0


def url_exists(dataset, url):
    table_name = get_table_name(dataset)
    result = query(
        f'SELECT url FROM {table_name} WHERE url = %s', (url,)
    )
    return len(result) > 0


def snapshot(meta):
    if meta.get('processed_storage_id'):
        return get_url_by_key(meta.get('processed_storage_id'))
    elif meta.get('origin_storage_id'):
        return get_url_by_key(meta.get('origin_storage_id'))
    elif meta.get('url'):
        return meta.get('url')
    elif meta.get('id') is not None:
        return meta.get('id')
    else:
        raise ValueError('No valid identifier found.')


def update_by_id(dataset, id, data, deplicate_ignore=[], tail=''):
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
            resp = query(sql[0], sql[1])
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


def batch_insert(sql=None, values=None, **kwargs):
    return execute(sql, values, batch=True, **kwargs)


def query(sql=None, values=None, **kwargs):
    cursor = execute(sql, values, **kwargs)
    try:
        columns = [desc[0] for desc in cursor.description]
        results = cursor.fetchall()
        return [dict(zip(columns, row)) for row in results]
    except Exception as e:
        pass
    return cursor


def get_dataset(dataset):
    check_dataset(dataset)
    ds = Empty()

    def init_instant():
        return init(dataset)

    def get_table_name_instant():
        return get_table_name(dataset)

    def url_exists_instant(url):
        return url_exists(dataset, url)

    def insert_instant(data, deplicate_ignore=[], tail=''):
        return insert(
            dataset, data, deplicate_ignore=deplicate_ignore, tail=tail
        )

    def update_by_id_instant(id, data, deplicate_ignore=[], tail=''):
        return update_by_id(
            dataset, id, data, deplicate_ignore=deplicate_ignore, tail=tail
        )

    def execute_instant(sql, values=None, **kwargs):
        return execute(sql, values, **kwargs)

    def query_instant(sql, values=None, **kwargs):
        return query(sql, values, **kwargs)

    def truncate_instant(force=False):
        return truncate(dataset, force=force)

    def exists(meta):
        return url_exists_instant(meta['url'])

    ds.execute = execute_instant
    ds.init = init_instant
    ds.exists = exists
    ds.get_table_name = get_table_name_instant
    ds.insert = insert_instant
    ds.truncate = truncate_instant
    ds.query = query_instant
    ds.snapshot = snapshot
    ds.update_by_id = update_by_id_instant
    ds.url_exists = url_exists_instant
    return ds


ensure_vector_extension()
ensure_bm25_extension()

__all__ = [
    'conn',
    'enrich_data',
    'execute',
    'generate_empty_vector',
    'get_dataset',
    'get_table_name',
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
