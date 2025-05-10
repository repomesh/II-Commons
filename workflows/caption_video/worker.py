from lib.caption import MODEL_NAME, Caption_video_api_segment
from lib.dataset import init
from lib.hatchet import (
    SCHEDULE_TIMEOUT,
    STEP_RETRIES,
    STEP_TIMEOUT,
    concurrency,
    hatchet,
    logs,
)
import os
import uuid

WORKFLOW = "Caption_Video"
WORKER = "Caption_Video"
WORKFLOW_LIMIT = int(os.environ.get("WORKFLOW_LIMIT_CAPTION_VIDEO", "10"))
WORKER_LIMIT = int(os.environ.get("WORKER_LIMIT_CAPTION_VIDEO", "1"))
API_KEY = os.environ.get("API_KEY")
VIDEO_CAPTION_MODEL = os.environ.get("VIDEO_CAPTION_MODEL")
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "10"))
RETRY_WAIT_SECONDS = int(os.environ.get("RETRY_WAIT_SECONDS", "1"))


@hatchet.workflow(
    name=WORKFLOW,
    schedule_timeout=SCHEDULE_TIMEOUT,
    on_events=["dataset:caption_video"],
    concurrency=concurrency(WORKFLOW_LIMIT),
)
class CaptionWorkflow:
    @hatchet.step(timeout=STEP_TIMEOUT, retries=STEP_RETRIES)
    def caption(self, context):

        task_id = uuid.uuid4()

        def log(msg, task_id=task_id):
            return logs(context, msg, task_id)

        args = context.workflow_input()
        try:
            ds = init(args["dataset"])
        except Exception as e:
            log(f"❌ Unable to init dataset: {args['dataset']}. Error: {e}")
            return {"dataset": args["dataset"], "meta_items": []}
        meta_items = (
            args["meta_items"]
            if type(args["meta_items"]) == list
            else [args["meta_items"]]
        )
        log("Caption video segments...")
        try:
            cap_res = Caption_video_api_segment(
                meta_items,
                api_key=API_KEY,
                model_name=VIDEO_CAPTION_MODEL,
                max_retries=MAX_RETRIES,
                retry_wait_seconds=RETRY_WAIT_SECONDS,
            )

            for i, cap_res in enumerate(cap_res):
                if context.done():
                    log(f"❌ Job canceled: {args['dataset']}.")
                    return {"dataset": args["dataset"], "captions": {}}
                snapshot = f'[{i}] {meta_items[i]["video_uri"]}'
                try:
                    ds.update_by_id(
                        i,
                        {
                            "caption": cap_res[i]["caption"],
                            "status": "finished",
                        },
                    )
                    log(f"🔥 ({snapshot}) Updated caption: " +
                        cap_res[i]["caption"])
                except Exception as e:
                    log(f"❌ ({snapshot}) Error updating caption: {e}")
                    print(cap_res)
        except Exception as e:
            log(f"❌ Error captioning: {e}")
            return {"dataset": args["dataset"], "captions": {}}
        log("👌 Done!")
        return {"dataset": args["dataset"], "captions": cap_res}


def run_worker():
    worker = hatchet.worker(WORKER, max_runs=WORKER_LIMIT)
    worker.register_workflow(CaptionWorkflow())
    worker.start()
