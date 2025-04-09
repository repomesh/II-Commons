from lib.dataset import init
from lib.hatchet import SCHEDULE_TIMEOUT, STEP_RETRIES, STEP_TIMEOUT, concurrency, hatchet, logs
from lib.psql import batch_insert
from lib.s3 import download_file
from lib.late import chunking
from lib.utilitas import json_dumps, sha256, read_json
import os
import tempfile
import uuid

WORKFLOW = 'Embedding_Text'
WORKER = 'Embedding_Text'
WORKFLOW_LIMIT = int(os.environ.get('WORKFLOW_LIMIT_EMBEDDING', '500'))
WORKER_LIMIT = int(os.environ.get('WORKER_LIMIT_EMBEDDING', '1'))


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
        # print(records)
    try:
        batch_insert(insert_query, records)
        # print(f"Inserted {len(records)} records")
    except Exception as e:
        print(f"Failed to insert records: {e}")


@hatchet.workflow(
    name=WORKFLOW, schedule_timeout=SCHEDULE_TIMEOUT,
    on_events=['dataset:embedding_text'], concurrency=concurrency(WORKFLOW_LIMIT)
)
class EmbeddingWorkflow:
    @hatchet.step(timeout=STEP_TIMEOUT, retries=STEP_RETRIES)
    def embedding(self, context):

        task_id = uuid.uuid4()

        def log(msg, task_id=task_id):
            return logs(context, msg, task_id)

        args = context.workflow_input()
        try:
            ds = init(args['dataset'])
        except Exception as e:
            log(f"❌ Unable to init dataset: {args['dataset']}. Error: {e}")
            return {'dataset': args['dataset'], 'meta_items': []}
        meta_items = args['meta_items'] if type(args['meta_items']) == list \
            else [args['meta_items']]
        task_hash = sha256(json_dumps(meta_items))
        temp_path = tempfile.TemporaryDirectory(suffix=f'-{task_hash}')
        texts = []
        for meta in meta_items:
            if context.done():
                log(f"❌ Job canceled: {args['dataset']}.")
                return {'dataset': args['dataset'], 'meta_items': []}
            snapshot = ds.snapshot(meta)
            log(f'✨ Processing item: {snapshot}')
            s3_address = meta['origin_storage_id']
            filename = os.path.join(temp_path.name, f"{meta['hash']}.json")
            try:
                download_file(s3_address, filename)
                log(f'Downloaded {s3_address} to: {filename}')
                json = read_json(filename)
                if len(json['title']) == 0 or len(json['text']) == 0:
                    continue
                text = f"---\nTitle: {json['title']}" \
                    + f"\nUrl: {json['url']}"
                if (json.get('contributor')
                    and json['contributor'].get('username')
                        and len(json['contributor']['username']) > 0):
                    text += f"\nContributor: {json['contributor']['username']}"
                if (len(json['timestamp']) > 0):
                    text += f"\nTimestamp: {json['timestamp']}"
                text += f"\n---\n\n{json['text']}"
                texts.append({
                    'id': meta['id'], 'text': text, 'meta': json,
                    'origin_storage_id': meta['origin_storage_id'],
                })
            except Exception as e:
                log(f'❌ ({snapshot}) {e}')
                continue
        meta_items = []
        for i, txt in enumerate(texts):
            chunks, embeddings = None, None
            try:
                log('Embedding Documents...')
                snapshot = json_dumps(txt['id'])
                chunks, _, embeddings = chunking(txt['text'])
                print(chunks, embeddings, len(chunks),
                      len(embeddings), len(embeddings[0]))
            except Exception as e:
                log(f'❌ ({snapshot}) Error embedding: {e}')
            if chunks is not None and embeddings is not None:
                if context.done():
                    log(f"❌ Job canceled: {args['dataset']}.")
                    return {'dataset': args['dataset'], 'meta_items': []}
                snapshot = txt['meta']['url']
                items = []
                for j in range(len(chunks)):
                    items.append({
                        'title': txt['meta']['title'],
                        'url': txt['meta']['url'],
                        'snapshot': txt['origin_storage_id'],
                        'chunk_index': j,
                        'chunk_text': chunks[j],
                        'source_db': args['dataset'],
                        'source_id': txt['id'],
                        'vector': embeddings[j],
                    })
                try:
                    insert_records_batch(ds, items)
                    log(f'🔥 ({snapshot}) Updated meta.')
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
            'dataset': args['dataset'],
            'meta_items': meta_items,
        }


def run():
    worker = hatchet.worker(WORKER, max_runs=WORKER_LIMIT)
    worker.register_workflow(EmbeddingWorkflow())
    worker.start()


__all__ = [
    'run'
]
