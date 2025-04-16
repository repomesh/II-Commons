from lib.text import encode as encode_text
from lib.dataset import init

MAX_DISTANCE = 2
MAX_SCORE = 50


def fusion_sort_key(result):
    distance = result['distance'] if result['distance'] is not None else MAX_DISTANCE
    score = result['score'] if result['score'] is not None else 0
    normalized_score = score / MAX_SCORE
    normalized_distance = 1 - (distance / MAX_DISTANCE)
    vector_weight = 0.6
    bm25_weight = 0.4
    fusion_score = vector_weight * normalized_distance + bm25_weight * normalized_score
    return fusion_score


def query(topic):
    ds = init('ms_marco')

    tp_resp = {"sentences": [topic], "keywords": [topic]}
    # Embedding phrases
    print("> Embedding phrases...")
    eb_resp = encode_text(tp_resp["sentences"], query=True)

    # Embedding phrases search
    print("> Embedding vector search...")
    e_res = []
    for e in eb_resp:
        e_res.append(ds.query(
            f"""SELECT id, item_id, answers, query, passage_text, is_selected, url, query_id, query_type, well_formed_answers, passage_id,
            (vector <=> %s::vector) as distance,
            ((2 - (vector <=> %s::vector)) / 2) as similarity
            FROM {ds.get_table_name()} order by (vector <=> %s::vector) ASC OFFSET %s LIMIT %s""",
            (e, e, e, 0, 100)
        ))

    # Unique embedding phrases search results
    unique_e_res = {}
    for result_set in e_res:
        for row in result_set:
            result_id = row['id']
            result_distance = row['distance']
            if result_id not in unique_e_res or result_distance < unique_e_res[result_id]['distance']:
                unique_e_res[result_id] = row
    e_res = sorted(list(unique_e_res.values()), key=lambda x: x['distance'])

    # BM25 search
    print("> BM25 search...")
    b_res = []
    for b in tp_resp['keywords']:
        b_res.append(ds.query(
            f"""SELECT id, item_id, answers, query, passage_text, is_selected, url, query_id, query_type, passage_id,
            well_formed_answers, paradedb.score(passage_text) as score
            FROM {ds.get_table_name()} WHERE passage_text @@@ %s
            ORDER BY score DESC OFFSET %s LIMIT %s""",
            (b, 0, 100)
        ))

    # Unique BM25 search results
    unique_b_res = {}
    for result_set in b_res:
        for row in result_set:
            result_id = row['id']
            result_score = row['score']
            if result_id not in unique_b_res or result_score > unique_b_res[result_id]['score']:
                unique_b_res[result_id] = row
    b_res = sorted(
        list(unique_b_res.values()),
        key=lambda x: x['score'],
        reverse=True
    )

    # Merge Vector and BM25 search results
    merged_results = {}
    for row in e_res:
        result_id = row['id']
        merged_results[result_id] = {
            **row,
            'score': None
        }
    for row in b_res:
        result_id = row['id']
        if result_id in merged_results:
            merged_results[result_id].update(row)
        else:
            merged_results[result_id] = {
                **row,
                'distance': None,
                'similarity': None
            }
    m_res = sorted(merged_results.values(
    ), key=lambda x: x['distance'] if x['distance'] is not None else MAX_DISTANCE)

    m_res = sorted(m_res, key=fusion_sort_key, reverse=True)
    m_res = m_res[:100]

    # Print fusion results
    print("Fusion Results:")
    print(f"{'Item ID':<10}    {'Passage ID':<10}    {'Is Selected':<5}    {'Distance':<15}    {'Score':<15}    {'URL':<50}    {'Passage Text':<30}")
    print("-" * 200)
    for row in m_res:
        distance = f"{row['distance']:.4f}" if isinstance(
            row['distance'], (int, float)) else "N/A"
        score = f"{row['score']:.4f}" if isinstance(
            row['score'], (int, float)) else "N/A"
        url = row.get('url', 'N/A')
        if len(url) >= 50:
            url = f"{url[:47]}..."
        passage_text = f"{row['passage_text']:<25}..."
        print(f"{row['item_id']:<10}    {row['passage_id']:<10}    {row['is_selected']:<5}    {distance:<15}    {score:<15}    {url:<50}    {passage_text:<25}")

    return m_res


if __name__ == "__main__":
    query("what cut is a bolar roast")

__all__ = [
    "query",
]
