# This code is taken and modified from - https://github.com/oneal2000/DRAGIN/tree/main
# This source code is licensed under the license found here - https://github.com/oneal2000/DRAGIN/blob/main/LICENSE

from typing import List, Tuple, Union, Dict
import argparse
import glob
import time
import csv
import json
import logging
from tqdm import tqdm
from beir.datasets.data_loader import GenericDataLoader
from elasticsearch import exceptions

def build_elasticsearch(
    beir_corpus_file_pattern: str,
    index_name: str,
):
    beir_corpus_files = glob.glob(beir_corpus_file_pattern)
    print(f'#files {len(beir_corpus_files)}')
    from beir.retrieval.search.lexical.elastic_search import ElasticSearch
    config = {
        'hostname': 'localhost',
        'index_name': index_name,
        'keys': {'title': 'title', 'body': 'txt'},
        'timeout': 100,
        'retry_on_timeout': True,
        'maxsize': 24,
        'number_of_shards': 'default',
        'language': 'english',
    }
    es = ElasticSearch(config)

    # create index
    print(f'create index {index_name}')
    try:
        es.delete_index()
    except exceptions.NotFoundError:
        pass
    time.sleep(5)
    es.create_index()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None, help='input file')
    parser.add_argument("--index_name", type=str, default=None, help="index name")
    args = parser.parse_args()
    build_elasticsearch(args.data_path, index_name=args.index_name)
