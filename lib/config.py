import logging
import sys
import os

os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'
os.environ['GRPC_POLL_STRATEGY'] = 'poll'


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    if 'grpc' in str(exc_value).lower():
        return

    sys.__excepthook__(exc_type, exc_value, exc_traceback)


sys.excepthook = handle_exception


class GlobalConfig:
    LOG_LEVEL = logging.INFO

    VERSION = '1.0.0'

    DEBUG = False

    DRYRUN = False

    @classmethod
    def set_log_level(cls, level: int):
        cls.LOG_LEVEL = level
        logging.getLogger().setLevel(level)
        if level == logging.DEBUG:
            cls.DEBUG = True

    @classmethod
    def set_dryrun(cls, dryrun: bool):
        cls.DRYRUN = dryrun
