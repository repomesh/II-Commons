import os
import signal
from hatchet_sdk import ConcurrencyExpression, ConcurrencyLimitStrategy, Hatchet
from lib.config import GlobalConfig

# https://docs.hatchet.run/home/features/timeouts
SCHEDULE_TIMEOUT = f'{3 * 24}h'
STEP_TIMEOUT = '30m'
STEP_RETRIES = 3

# https://docs.hatchet.run/home/features/concurrency/round-robin
LIMIT_EXPRESSION = 'input.dataset'
LIMIT_STRATEGY = ConcurrencyLimitStrategy.GROUP_ROUND_ROBIN


def logs(ctx, msg, task_id=None):
    if task_id:
        msg = f'[{task_id}]: {msg}'
    print(msg)
    ctx.log(msg)


def push(event, data):
    return hatchet.event.push(event, data)


def push_dataset_event(type, dataset, meta_items):
    # print(meta_items)
    count, event = len(meta_items), f'dataset:{type}'
    if count > 0:
        if count > 1:
            print(f'Event `{event}` submitted with {count} items.')
        return push(event, {'dataset': dataset, 'meta_items': meta_items})


def concurrency(max_runs):
    return ConcurrencyExpression(
        expression=LIMIT_EXPRESSION, limit_strategy=LIMIT_STRATEGY,
        max_runs=max_runs,
    )


def signal_handler(signum, frame):
    pgid = os.getpgid(0)
    os.killpg(pgid, signal.SIGKILL)
    os._exit(0)


def set_signal_handler():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGQUIT, signal_handler)


hatchet = Hatchet(debug=GlobalConfig.DEBUG)

__all__ = [
    'LIMIT_EXPRESSION',
    'LIMIT_STRATEGY',
    'SCHEDULE_TIMEOUT',
    'STEP_RETRIES',
    'STEP_TIMEOUT',
    'concurrency',
    'ConcurrencyExpression',
    'ConcurrencyLimitStrategy',
    'hatchet',
    'logs',
    'push_dataset_event',
    'push',
    'set_signal_handler',
]
