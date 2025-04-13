from lib.caption import caption_image
from lib.dataset import init
from lib.hatchet import SCHEDULE_TIMEOUT, STEP_RETRIES, STEP_TIMEOUT, concurrency, hatchet, logs, set_signal_handler
from lib.s3 import get_url_by_key
import os
import uuid

WORKFLOW = 'Caption'
WORKER = 'Caption'
WORKFLOW_LIMIT = int(os.environ.get('WORKFLOW_LIMIT_CAPTION', '30'))
WORKER_LIMIT = int(os.environ.get('WORKER_LIMIT_CAPTION', '1'))


@hatchet.workflow(
    name=WORKFLOW, schedule_timeout=SCHEDULE_TIMEOUT,
    on_events=['dataset:caption'], concurrency=concurrency(WORKFLOW_LIMIT)
)
class CaptionWorkflow:
    @hatchet.step(timeout=STEP_TIMEOUT, retries=STEP_RETRIES)
    def caption(self, context):

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
        urls = {}
        for _, meta in enumerate(meta_items):
            urls[meta['id']] = get_url_by_key(meta['processed_storage_id'])
        log('Caption images...')
        try:
            cap_res = caption_image(meta_items)
            for i in cap_res:
                if context.done():
                    log(f"❌ Job canceled: {args['dataset']}.")
                    return {'dataset': args['dataset'], 'captions': {}}
                snapshot = f'[{i}] {urls[i]}'
                try:
                    ds.update_by_id(i, {
                        'caption_qw25vl': cap_res[i]['caption'],
                        'caption_long_qw25vl': cap_res[i]['caption_long']
                    })
                    log(f'🔥 ({snapshot}) Updated caption: ' +
                        cap_res[i]['caption'])
                except Exception as e:
                    log(f'❌ ({snapshot}) Error updating caption: {e}')
                    print(cap_res)
        except Exception as e:
            log(f'❌ Error captioning: {e}')
            return {'dataset': args['dataset'], 'captions': {}}
        log('👌 Done!')
        return {'dataset': args['dataset'], 'captions': cap_res}


def run():
    worker = hatchet.worker(WORKER, max_runs=WORKER_LIMIT)
    worker.register_workflow(CaptionWorkflow())
    set_signal_handler()
    worker.start()


__all__ = [
    'run'
]
