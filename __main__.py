#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from bin.analyze import run as run_analyze
from bin.fusion import query as query
from bin.fusion_lite import query as query_lite
from lib.config import GlobalConfig
from typing import Optional
# from workflows.caption import run_host as run_caption_host, run_worker as run_caption_worker
from workflows.embedding_image import run_worker as run_embedding_image_worker
# from workflows.embedding_text import run_host as run_embedding_text_host, run_worker as run_embedding_text_worker
# from workflows.fetch import run_host as run_dataset_fetch_host, run_worker as run_dataset_fetch_worker
import argparse
import atexit
import logging
import sys

logging.basicConfig(
    level=GlobalConfig.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cleanup():
    try:
        pass
    except Exception as e:
        pass


atexit.register(cleanup)


def run_worker(worker_name: str, dataset_name: str):
    # if worker_name == 'dataset_fetch':
    #     run_dataset_fetch_worker()
    # elif worker_name == 'embedding_text':
    #     run_embedding_text_worker()
    if worker_name == 'embedding_image':
        run_embedding_image_worker(dataset_name)
    # elif worker_name == 'caption':
    #     run_caption_worker()
    else:
        logger.error(f"Invalid worker name: {worker_name}")
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='II Dataset Tools',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Demo:
    python -m dataset_tools --dataset dataset_path --name dataset_name
    python -m dataset_tools --worker worker_name
    python -m dataset_tools --version
        """
    )

    parser.add_argument(
        '-d', '--dataset',
        help='dataset path: metadata file or directory',
        type=str,
        required=False
    )

    parser.add_argument(
        '-n', '--name',
        help='dataset name: cc12m, cc12m_cleaned, cc12m_woman, vintage_450k, pd12m, wikipedia_featured, megalith_10m, arxiv',
        type=str,
        required=False
    )

    parser.add_argument(
        '-r', '--dryrun',
        help='dryrun: dryrun',
        action='store_true',
        required=False
    )

    parser.add_argument(
        '-w', '--worker',
        help='run worker: fetch, embedding',
        type=str,
        required=False
    )

    parser.add_argument(
        '-a', '--analyze',
        help='analyze: analyze',
        action='store_true',
        required=False
    )

    parser.add_argument(
        '-q', '--query',
        help='query: topic',
        type=str,
        required=False
    )

    parser.add_argument(
        '-lq', '--lite-query',
        help='lite query: topic',
        type=str,
        required=False
    )

    parser.add_argument(
        '-e', '--verbose',
        help='show verbose logs',
        action='store_true'
    )

    parser.add_argument(
        '-v', '--version',
        help='show version information',
        action='version',
        version=f'dataset-tools {GlobalConfig.VERSION}'
    )

    return parser.parse_args()


# def run_workflow_worker():
#     run_dataset_fetch_worker()

# DATASET_BASE = '/Volumes/Betty/Datasets/meta'
# DATASETS = {
#     'cc12m': {'meta_path': 'meta.tsv'},  # done
#     'cc12m_cleaned': {'meta_path': 'meta.jsonl'},  # done
#     'cc12m_woman': {'meta_path': 'meta.jsonl'},  # done
#     'vintage_450k': {'meta_path': 'meta.parquet'},  # done
#     'PD12M': {'meta_path': 'meta'},  # done
#     'wikipedia_featured': {'meta_path': 'meta'},  # done
#     'megalith_10m': {'meta_path': 'meta'},  # Stopped, flickr limitation
#     'arxiv': {'meta_path': 'arxiv-metadata-hash-abstracts-v0.2.0-2019-03-01.json'},
#     'arxiv_oai': {'meta_path': 'arxiv-metadata-oai-snapshot.json'},
# }
# META_PATH = os.path.join(DATASET_BASE, DATASET, DATASETS[DATASET]['meta_path'])


def main() -> Optional[int]:
    try:
        args = parse_args()

        if args.verbose:
            GlobalConfig.set_log_level(logging.DEBUG)

        if args.dryrun:
            GlobalConfig.set_dryrun(True)

        if args.worker:
            logger.info(f"Run worker: {args.worker}...")
            run_worker(args.worker, args.name)
        elif args.analyze:
            logger.info(f"Run analyze...")
            run_analyze()
        elif args.query:
            logger.info(f"Run query: {args.query}...")
            query(args.query)
        elif args.lite_query:
            logger.info(f"Run lite query: {args.lite_query}...")
            query_lite(args.lite_query)
        else:
            logger.info("No action specified")
            return 1

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
