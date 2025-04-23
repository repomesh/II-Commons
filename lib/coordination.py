from time import time
from lib.psql import init
import uuid

LIVE_TIME = 60 * 3
dataset, ds = 'workers', None
_uuid, _worker_count = None, 0


def heartbeat(workflow):
    global ds, _uuid, _worker_count
    if ds is None:
        ds = init(dataset)
        _uuid = str(uuid.uuid4())
    if workflow is None:
        raise Exception("Workflow is required")
    workflow = str(workflow).upper()
    now = time()
    # delete expired workers
    ds.query(
        f"DELETE FROM {ds.get_table_name()} WHERE last_heartbeat < to_timestamp(%s)",
        (now - LIVE_TIME,)
    )
    # register new worker
    ds.query(
        f"INSERT INTO {ds.get_table_name()} (uuid, workflow, last_heartbeat) VALUES (%s, %s, to_timestamp(%s)) ON CONFLICT (uuid) DO UPDATE SET last_heartbeat = to_timestamp(%s)",
        (_uuid, workflow, now, now)
    )
    # get all workers
    workers = ds.query(
        f"SELECT * FROM {ds.get_table_name()} WHERE workflow = %s ORDER BY id ASC",
        (workflow,)
    )
    # get current worker
    order = next((i for i, w in enumerate(
        workers) if w['uuid'] == _uuid), None)
    if order is None:
        raise Exception("Worker not found")
    # return
    count = len(workers)
    reset = True if count != _worker_count else False
    _worker_count = count
    print(
        f'❤️ [{workflow}:{_uuid}] Heartbeat: '
        + f'{order} / {count} {"(reset)" if reset else ""}'
    )
    return count, order, reset


if __name__ == '__main__':
    heartbeat('testing')

__all__ = [
    'init_table',
    'heartbeat',
]
