#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import sys
from typing import Optional
from workflows.DatasetFetch import run_host as run_dataset_fetch_host, run_worker as run_dataset_fetch_worker

VERSION = '1.0.0'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='II Dataset Tools',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Demo:
    python -m dataset_tools --load dataset_file_path
    python -m dataset_tools --worker worker_name
    python -m dataset_tools --version
        """
    )

    parser.add_argument(
        '-l', '--load',
        help='load dataset',
        type=str,
        required=False
    )
    parser.add_argument(
        '-w', '--worker',
        help='run worker: fetch, embedding',
        type=str,
        required=False
    )
    parser.add_argument(
        '-v', '--verbose',
        help='show verbose logs',
        action='store_true'
    )
    parser.add_argument(
        '--version',
        help='show version information',
        action='version',
        version=f'dataset-tools {VERSION}'
    )

    return parser.parse_args()


def run_workflow_host(workflow_name: str):
    if workflow_name == 'dataset_fetch':
        run_dataset_fetch_host()
    else:
        logger.error(f"Invalid workflow name: {workflow_name}")
        return 1


# def run_workflow_worker():
#     run_dataset_fetch_worker()


def main() -> Optional[int]:
    try:
        args = parse_args()

        if args.verbose:
            logger.setLevel(logging.DEBUG)

        if args.load:
            logger.info(f"Load dataset: {args.load}")
            # TODO: 實現文件處理邏輯
        elif args.worker:
            logger.info(f"Run worker: {args.worker}")
            # TODO: 實現輸出邏輯
        else:
            logger.info("No action specified")
            return 1

        return 0

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
