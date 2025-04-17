from lib.dataset import init
from lib.hatchet import Context, SCHEDULE_TIMEOUT, STEP_RETRIES, STEP_TIMEOUT, concurrency, hatchet, logs, set_signal_handler, WORKFLOW_LIMIT, WorkflowInput
from lib.psql import batch_insert
from lib.s3 import download_file
from lib.text import process
from lib.utilitas import json_dumps, sha256, read_json
import os
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

WORKFLOW = 'Embedding_Text'
WORKER = 'Embedding_Text'
SLOTS = int(os.environ.get('WORKER_SLOTS_EMBEDDING_TEXT', '10'))

# Create a thread pool for database operations
db_executor = ThreadPoolExecutor(max_workers=10)

def insert_records_batch(ds, items):
    """Insert a batch of records into the database"""
    insert_query = f"""
    INSERT INTO {ds.get_table_name()}
    (title, url, snapshot, chunk_index, chunk_text, source_db, source_id, vector)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (source_db, source_id, chunk_index) DO NOTHING
    """
    records = []
    for record in items:
        records.append((
            record['title'],
            record['url'],
            record['snapshot'],
            record['chunk_index'],
            record['chunk_text'],
            record['source_db'],
            record['source_id'],
            record['vector'],
        ))
    try:
        # Submit the database operation to the thread pool and continue without waiting
        db_executor.submit(batch_insert, insert_query, records)
    except Exception as e:
        print(f"Failed to submit database operation: {e}")


EmbeddingWorkflow = hatchet.workflow(
    name=WORKFLOW, on_events=['dataset:embedding_text'],
    concurrency=concurrency(WORKFLOW_LIMIT), input_validator=WorkflowInput
)

@EmbeddingWorkflow.task(
    schedule_timeout=SCHEDULE_TIMEOUT,
    execution_timeout=STEP_TIMEOUT,
    retries=STEP_RETRIES
)
def embedding(args: WorkflowInput, context: Context) -> dict:

    task_id = uuid.uuid4()

    def log(msg, task_id=task_id):
        return logs(context, msg, task_id)

    try:
        ds = init(args.dataset)
    except Exception as e:
        log(f"❌ Unable to init dataset: {args.dataset}. Error: {e}")
        return {'dataset': args.dataset, 'meta_items': []}
    meta_items = args.meta_items if type(args.meta_items) == list \
        else [args.meta_items]
    task_hash = sha256(json_dumps(meta_items))
    temp_path = tempfile.TemporaryDirectory(suffix=f'-{task_hash}')
    texts = []
    for meta in meta_items:
        if context.done():
            log(f"❌ Job canceled: {args.dataset}.")
            return {'dataset': args.dataset, 'meta_items': []}
        snapshot = ds.snapshot(meta)
        log(f'✨ Processing item: {snapshot}')
        s3_address = meta['origin_storage_id']
        filename = os.path.join(temp_path.name, f"{meta['hash']}.json")
        try:
            download_file(s3_address, filename)
            log(f'Downloaded {s3_address} to: {filename}')
            json = read_json(filename)
            if len(json['text']) == 0:
                continue
            texts.append({
                'id': meta['id'], 'text': json['text'], 'meta': json,
                'origin_storage_id': meta['origin_storage_id'],
            })
        except Exception as e:
            log(f'❌ ({snapshot}) {e}')
            continue
    meta_items = []
    for i, txt in enumerate(texts):
        try:
            log('Embedding Documents...')
            snapshot = json_dumps(txt['id'])
            end_res = process(txt['text'])
        except Exception as e:
            log(f'❌ ({snapshot}) Error embedding: {e}')
            continue
        if end_res is not None:
            if context.done():
                log(f"❌ Job canceled: {args.dataset}.")
                return {'dataset': args.dataset, 'meta_items': []}
            snapshot = txt['meta']['url']
            items = []
            for j in range(len(end_res)):
                chk = end_res[j]
                items.append({
                    'title': txt['meta']['title'],
                    'url': txt['meta']['url'],
                    'snapshot': txt['origin_storage_id'],
                    'chunk_index': j,
                    'chunk_text': chk['chunk'],
                    'source_db': args.dataset,
                    'source_id': txt['id'],
                    'vector': chk['embedding'],
                })
            try:
                db_insert_time = time.time()
                insert_records_batch(ds, items)
                db_insert_time = time.time() - db_insert_time
                log(f'🔥 ({snapshot}) Updated meta: {len(items)} items in {db_insert_time:.2f} seconds')
                del txt['meta']
                del txt['text']
                meta_items.append(txt)
            except Exception as e:
                log(f'❌ ({snapshot}) Error updating meta: {e}')
                print(txt['meta'])
        else:
            log('❌ No embedding result.')
    log('👌 Done!')
    return {
        'dataset': args.dataset,
        'meta_items': meta_items,
    }


def run():
    worker = hatchet.worker(
        WORKER, slots=SLOTS,
        workflows=[EmbeddingWorkflow]
    )
    set_signal_handler()
    worker.start()


__all__ = [
    'run'
]
