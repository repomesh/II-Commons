from lib.dataset import init
from lib.hatchet import SCHEDULE_TIMEOUT, STEP_RETRIES, STEP_TIMEOUT, concurrency, hatchet, logs, set_signal_handler
from lib.s3 import exists, get_address_by_key, upload_file
from lib.gcs import download_file
from lib.utilitas import download, json_dumps, sha256, get_file_type
import os
import tempfile
import time
import uuid

WORKFLOW = 'Fetch'
WORKER = 'Fetch'
WORKFLOW_LIMIT = int(os.environ.get('WORKFLOW_LIMIT_DATASETFETCH', '500'))
WORKER_LIMIT = int(os.environ.get('WORKER_LIMIT_DATASETFETCH', '30'))
i = 0


@hatchet.workflow(
    name=WORKFLOW, schedule_timeout=SCHEDULE_TIMEOUT,
    on_events=['dataset:fetch'], concurrency=concurrency(WORKFLOW_LIMIT)
)
class DatasetFetchWorkflow:
    @hatchet.step(timeout=STEP_TIMEOUT, retries=STEP_RETRIES)
    def download_data(self, context):
        global i
        task_id = uuid.uuid4()

        def log(msg, task_id=task_id):
            return logs(context, msg, task_id)

        def sleep(seconds=1):
            log(f'⏰ Sleeping for {seconds} second(s) to avoid rate limit...')
            time.sleep(seconds)

        # https://docs.hatchet.run/home/features/timeouts
        # context.refresh_timeout("15s")
        args = context.workflow_input()
        if args['dataset'] == 'megalith_10m':
            return {}
        try:
            ds = init(args['dataset'])
        except Exception as e:
            log(f"❌ Unable to init dataset: {args['dataset']}. Error: {e}")
            return {'dataset': args['dataset'], 'meta_items': []}
        meta_items = args['meta_items'] if type(args['meta_items']) == list \
            else [args['meta_items']]
        task_hash = sha256(json_dumps(meta_items))
        temp_path = tempfile.TemporaryDirectory(suffix=f'-{task_hash}')
        results = []
        for meta in meta_items:
            # i += 1
            # if i %  == 0:
            # sleep(5)
            snapshot = ds.snapshot(meta)
            log(f'✨ Processing item: {snapshot}')
            s3_key = ds.get_s3_key(meta)
            subfix = 'jpg'
            match args['dataset']:
                case 'arxiv':
                    subfix = 'pdf'
            filename = os.path.join(temp_path.name, f"{meta['hash']}.{subfix}")
            # hack:
            if meta['url'].endswith('.jpg'):
                log(f"Skipping download: {meta['url']}")
                continue
            try:
                s3_exs = exists(s3_key)
                if s3_exs:
                    meta['origin_storage_id'] = get_address_by_key(s3_key)
                    log(f"Skipping download: {meta['origin_storage_id']}")
                else:
                    arr_pid = meta['paper_id'].split('/')
                    if len(arr_pid) == 1:
                        gcs_url = f"gs://arxiv-dataset/arxiv/arxiv/pdf/{meta['paper_id'].split('.')[0]}/{meta['paper_id']}{meta['versions'][-1]}.pdf"
                    elif len(arr_pid) == 2:
                        gcs_url = f"gs://arxiv-dataset/arxiv/{arr_pid[0]}/pdf/{arr_pid[1][:4]}/{arr_pid[1]}{meta['versions'][-1]}.pdf"
                    else:
                        print(meta)
                        raise ValueError('Invalid paper_id.')
                    try:
                        # hack:
                        os.environ['GCS_BUCKET'] = "arxiv-dataset"
                        download_file(gcs_url, filename)
                        log(f"Downloaded {gcs_url} to: {filename}")
                    except Exception as e:
                        log(
                            f"Fownload failed from GCS: {gcs_url}, try direct download..."
                        )
                        download(meta['url'], filename)
                        log(f"Downloaded {meta['url']} to: {filename}")
                    if get_file_type(filename) != 'PDF':
                        raise ValueError('Unexpected file type.')
                    meta['origin_storage_id'] = upload_file(filename, s3_key)
                    log(f"Uploaded to S3: {meta['origin_storage_id']}")
                ds.insert(meta)
                log(f'🔥 Inserted meta: {snapshot}')
            except Exception as e:
                log(f'❌ Error handling {snapshot}: {e}')
                continue
            results.append(meta)
        log('👌 Done!')
        return {
            'dataset': args['dataset'],
            'meta_items': results,
        }


def run():
    worker = hatchet.worker(WORKER, max_runs=WORKER_LIMIT)
    worker.register_workflow(DatasetFetchWorkflow())
    set_signal_handler()
    worker.start()


__all__ = [
    'run'
]
