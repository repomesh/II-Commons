from lib.dataset import init
from lib.embedding import encode_image
from lib.hatchet import Context, SCHEDULE_TIMEOUT, STEP_RETRIES, STEP_TIMEOUT, concurrency, hatchet, logs, set_signal_handler, WORKFLOW_LIMIT, WorkflowInput
from lib.preprocess import process
from lib.s3 import download_file, upload_file
from lib.utilitas import json_dumps, sha256
from lib.utilitas import write_image
import os
import tempfile
import uuid

WORKFLOW = 'Embedding_Image'
WORKER = 'Embedding_Image'
SLOTS = int(os.environ.get('WORKER_SLOTS_EMBEDDING_IMAGE', '1'))


EmbeddingWorkflow = hatchet.workflow(
    name=WORKFLOW,
    on_events=['dataset:embedding_image'],
    concurrency=concurrency(WORKFLOW_LIMIT),
    input_validator=WorkflowInput
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
    images = []
    for meta in meta_items:
        if context.done():
            log(f"❌ Job canceled: {args.dataset}.")
            return {'dataset': args.dataset, 'meta_items': []}
        snapshot = ds.snapshot(meta)
        log(f'✨ Processing item: {snapshot}')
        s3_address = meta['processed_storage_id'] if args.dataset == 'alpha' else meta['origin_storage_id']
        filename = os.path.join(temp_path.name, f"{meta['hash']}.jpg")
        try:
            download_file(s3_address, filename)
            log(f'Downloaded {s3_address} to: {filename}')
            ps_result = process(filename)
            id = meta['id']
            if args.dataset == 'alpha':
                meta['processed_storage_id'] = s3_address
            # alpha does not need processed image
            elif args.dataset != 'alpha' and ps_result['processed']:
                subfix = '.processed.jpg'
                sub_name = f"{meta['hash']}{subfix}"
                filename = os.path.join(temp_path.name, sub_name)
                write_image(ps_result['processed_image'], filename)
                s3_key = f'{s3_address}{subfix}'
                s3_address = upload_file(filename, s3_key)
                log(f"Uploaded processed image to S3: {s3_address}")
                meta = ps_result['meta']
                meta['processed_storage_id'] = s3_address
            else:
                meta = {}
            if not os.path.exists(filename) and not s3_address:
                raise Exception(f'Download / upload failed.')
            images.append({
                'image': ps_result['processed_image'],
                'id': id, 'meta': meta,
            })
        except Exception as e:
            log(f'❌ ({snapshot}) {e}')
            continue
    meta_items, end_res = [], None
    try:
        log('Embedding images...')
        snapshot = json_dumps([img['id'] for img in images])
        end_res = encode_image([img['image'] for img in images])
    except Exception as e:
        log(f'❌ ({snapshot}) Error embedding: {e}')
    if end_res is not None:
        for i, img in enumerate(images):
            if context.done():
                log(f"❌ Job canceled: {args.dataset}.")
                return {'dataset': args.dataset, 'meta_items': []}
            snapshot = img['meta']['processed_storage_id']
            img['meta']['vector'] = end_res[i].tolist()
            try:
                if args.dataset == 'alpha':
                    del img['meta']['hash']
                    del img['meta']['origin_storage_id']
                    del img['meta']['processed_storage_id']
                # print(img['id'], img['meta'])
                ds.update_by_id(img['id'], img['meta'])
                log(f'🔥 ({snapshot}) Updated meta.')
                del img['meta']['vector']
                meta_items.append({'id': img['id'], **img['meta']})
            except Exception as e:
                log(f'❌ ({snapshot}) Error updating meta: {e}')
                print(img['meta'])
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
