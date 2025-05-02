from lib.meta import parse_tube_parquet
from lib.dataset import init
from lib.text import encode

ds = init('ms_marco')


def load():
    file_path = '/Volumes/Ann/Tmp/ms_marco/v1.1/test-00000-of-00001.parquet'
    data = parse_tube_parquet(file_path)
    limit = 1000
    cur = 0
    res = []
    ds.query(f'TRUNCATE TABLE {ds.get_table_name()}')
    while cur < limit:
        print('Loading:', data[cur]['id'], data[cur]['query'])
        print('Embedding...')
        chunks = encode(data[cur]['passages']['passage_text'])
        for i in range(len(data[cur]['passages']['passage_text'])):
            item = {
                'item_id': data[cur]['id'],
                'answers': [a.strip() for a in data[cur]['answers']],  # array
                'query': data[cur]['query'],  # string
                'passage_id': i,
                'passage_text': data[cur]['passages']['passage_text'][i],
                'is_selected': int(data[cur]['passages']['is_selected'][i]),
                'url': data[cur]['passages']['url'][i],
                'query_id': data[cur]['query_id'],
                'query_type': data[cur]['query_type'],
                'wellFormedAnswers': [a.strip() for a in data[cur]['wellFormedAnswers']],
                'vector': chunks[i],
            }
            res.append(item)
            ds.query(f'INSERT INTO {ds.get_table_name()} (item_id, answers, query, passage_text, is_selected, url, query_id, query_type, well_formed_answers, vector, passage_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)', (
                item['item_id'],
                item['answers'],
                item['query'],
                item['passage_text'],
                item['is_selected'],
                item['url'],
                item['query_id'],
                item['query_type'],
                item['wellFormedAnswers'],
                item['vector'],
                item['passage_id'],
            ))
        cur += 1
    return res


# data = pd.read_parquet(os.path.join(file_path, fn))
#     for i in tqdm(range(len(data)), colour="green", desc="Tokenizing:" + fn):
#         if docs_count >= max_docs:
#             break
#         query = data.iloc[i]['query']
#         query_id = data.iloc[i]['query_id']
#         passages_idx = 0
#         for rel, text in zip(data.iloc[i]['passages']['is_selected'], data.iloc[i]['passages']['passage_text']):


if __name__ == '__main__':
    load()
