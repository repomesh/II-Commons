from time import time
from lib.psql import query
import uuid

TABLE_NAME = 'workers'
LIVE_TIME = 20  # 60 * 10
_uuid = None


def init_table():
    list_sql = [
        f"""CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id BIGSERIAL PRIMARY KEY,
            uuid VARCHAR NOT NULL,
            last_heartbeat TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )""",
        f'CREATE UNIQUE INDEX IF NOT EXISTS {TABLE_NAME}_uuid_index ON {TABLE_NAME} (uuid)',
        f'CREATE INDEX IF NOT EXISTS {TABLE_NAME}_last_heartbeat_index ON {TABLE_NAME} (last_heartbeat)',
    ]
    for sql in list_sql:
        query(sql)


def heartbeat():
    global _uuid
    if _uuid is None:
        init_table()
        _uuid = str(uuid.uuid4())
    now = time()
    # delete expired workers
    query(
        f"DELETE FROM {TABLE_NAME} WHERE last_heartbeat < to_timestamp(%s)",
        (now - LIVE_TIME,)
    )
    # register new worker
    query(
        f"INSERT INTO {TABLE_NAME} (uuid, last_heartbeat) VALUES (%s, to_timestamp(%s)) ON CONFLICT (uuid) DO UPDATE SET last_heartbeat = to_timestamp(%s)",
        (_uuid, now, now)
    )
    # get all workers
    workers = query(f"SELECT * FROM {TABLE_NAME} ORDER BY id ASC")
    # get current worker
    order = next((i for i, w in enumerate(
        workers) if w['uuid'] == _uuid), None)
    if order is None:
        raise Exception("Worker not found")
    # return
    count = len(workers)
    print(f'❤️ [{_uuid}] Heartbeat: {order} / {count}')
    return count, order


if __name__ == '__main__':
    heartbeat()

__all__ = [
    'init_table',
    'heartbeat',
]
