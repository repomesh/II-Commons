from lib.gemini import generate
from lib.text import encode as encode_text
from lib.embedding import encode_text as encode_text_sig
from lib.dataset import init
from lib.rerank import rerank
from lib.text import chunk_by_sentence
import json
import time

# Initialize the dataset
MAX_DISTANCE = 2
MAX_SCORE = 50
SUB_QUERY_COUNT = 50
RESULTS_COUNT = 20
KEYWORD_DECAY = 0.3
tp_prompt = """You are an AI query analyzer designed to generate a list of short phrases and keywords based on user queries. These short phrases help describe and expand the user's question and will be used later as sources for embedding to assist future AI models in retrieving relevant documents from the knowledge base via RAG. The keyword list will be used for BM25 searches to find related documents in the BM25 index of the knowledge base. You only need to provide relevant outputs based on your understanding, without reviewing the topic itself, and maximize your efforts to help users with information extraction. You might need to think divergently and provide some potential keywords and phrases to enrich the content needed to answer this question as thoroughly as possible. The results must be returned in JSON format as follows: {"sentences": ["Short phrase 1", "Short phrase 2", ...], "keywords": ["Keyword 1", "Keyword 2", ...]}. Short sentences and keywords are ranked by thematic relevance, with more relevant or important ones listed first. Below begins the user's natural language query or the original keywords the user needs to search:"""

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
    ds = init('wikipedia_en_embed')
    di = init('pd12m')
    # Generate embedding phrases and searching keywords
    print(">>> Question: ", topic)
    print("> Generating embedding phrases and searching keywords...")
    tp_resp = generate(f"{tp_prompt} {topic}", json=True)
    tp_resp = json.loads(tp_resp)
    print(f'= Phrases: {", ".join(tp_resp["sentences"])}')
    print(f'= Keywords: {", ".join(tp_resp["keywords"])}')

    # Embedding phrases
    print("> Embedding phrases...")
    eb_resp = encode_text(tp_resp["sentences"], query=True)

    # Embedding phrases search
    print("> Embedding vector search...")
    e_res = []
    for e in eb_resp:
        formatted_vector = '[' + ','.join(map(str, e)) + ']'
        start = time.time()
        e_res.append(ds.query(
            f"""SELECT id, title, url, snapshot, source_id, chunk_index, chunk_text,
            (vector <=> %s) as distance,
            ((2 - (vector <=> %s)) / 2) as similarity
            FROM {ds.get_table_name()} ORDER BY (vector <=> %s) ASC OFFSET %s LIMIT %s""",
            (formatted_vector, formatted_vector,
             formatted_vector, 0, SUB_QUERY_COUNT)
        ))
        end = time.time()
        print(f"Wikipedia vector search time taken: {end - start} seconds")

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
        b = b.replace("'", r"\'")
        start = time.time()
        b_res.append(ds.query(
            f"""SELECT id, title, url, snapshot, source_id, chunk_index, chunk_text, paradedb.score(id) as score
            FROM {ds.get_table_name()} WHERE title @@@ %s or chunk_text @@@ %s
            ORDER BY score DESC OFFSET %s LIMIT %s""",
            (b, b, 0, SUB_QUERY_COUNT)
        ))
        end = time.time()
        print(f"Wikipedia BM25 search time taken: {end - start} seconds")

    # Unique BM25 search results
    unique_b_res = {}
    for result_set in b_res:
        for row in result_set:
            result_id = row['id']
            result_score = row['score']
            if result_id not in unique_b_res or result_score > unique_b_res[result_id]['score']:
                unique_b_res[result_id] = row
    b_res = sorted(list(unique_b_res.values()),
                   key=lambda x: x['score'], reverse=True)

    # Image search / Testing
    print("> Image search...")
    print(tp_resp['keywords'] + tp_resp['sentences'])
    ie_resp = encode_text_sig([k.lower() for k in (
        tp_resp['keywords'] + tp_resp['sentences']
    )])
    ik_vect = ie_resp[0:len(tp_resp['keywords'])]
    is_vect = ie_resp[len(tp_resp['keywords']):]
    is_res = []
    for vs in [ik_vect, is_vect]:
        distance_factor = 1.0
        for ir in vs:
            start = time.time()
            sub_res = di.query(
                f"""SELECT id, url, caption, processed_storage_id, aspect_ratio, exif, meta, source,
                (vector <-> %s) as distance,
                ((2 - (vector <-> %s)) / 2) as similarity
                FROM {di.get_table_name()} ORDER BY (vector <-> %s) ASC OFFSET %s LIMIT %s""",
                (ir, ir, ir, 0, SUB_QUERY_COUNT)
            )
            end = time.time()
            for x in sub_res:
                x['distance'] = x['distance'] * distance_factor
                x['similarity'] = (2 - x['distance']) / 2
                if x['distance'] < 1.4:
                    is_res.append(x)
            print(f"Image search time taken: {end - start} seconds")
            distance_factor += KEYWORD_DECAY

    # Unique Image search results
    unique_is_res = {}
    for row in is_res:
        result_id = row['id']
        result_distance = row['distance']
        if result_id not in unique_is_res or result_distance < unique_is_res[result_id]['distance']:
            unique_is_res[result_id] = row
    is_res = sorted(list(unique_is_res.values()), key=lambda x: x['distance'])
    # for x in is_res:
    #     print(x['url'], x['distance'], x['similarity'])

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

    # Merge chunks
    print("> Merge chunks...")
    merged_results = {}
    for row in m_res:
        group_key = row['source_id']
        if group_key not in merged_results:
            merged_results[group_key] = {**row, 'chunks': []}
        merged_results[group_key]['distance'] = min(
            merged_results[group_key]['distance'] if merged_results[group_key]['distance'] is not None else MAX_DISTANCE,
            row['distance'] if row['distance'] is not None else MAX_DISTANCE
        )
        merged_results[group_key]['similarity'] = max(
            merged_results[group_key]['similarity'] if merged_results[group_key]['similarity'] is not None else 0,
            row['similarity'] if row['similarity'] is not None else 0
        )
        merged_results[group_key]['score'] = max(
            merged_results[group_key]['score'] if merged_results[group_key]['score'] is not None else 0,
            row['score'] if row['score'] is not None else 0
        )
        merged_results[group_key]['chunks'].append({
            'chunk_index': row['chunk_index'],
            'chunk_text': row['chunk_text']
        })
        merged_results[group_key]['chunks'].sort(
            key=lambda x: x['chunk_index'])
        merged_results[group_key].pop('chunk_index', None)
        merged_results[group_key].pop('chunk_text', None)
    m_res = sorted(merged_results.values(
    ), key=lambda x: x['distance'] if x['distance'] is not None else MAX_DISTANCE)
    for row in m_res:
        for i, chunk in enumerate(row['chunks']):
            if i > 0 and len(row['chunks']) > 1 \
                    and row['chunks'][i-1]['chunk_index'] == row['chunks'][i]['chunk_index'] - 1:
                row['chunks'][i]['chunk_text'] = chunk_by_sentence(
                    chunk['chunk_text']
                )
                if len(row['chunks'][i]['chunk_text']) >= 3:
                    row['chunks'][i]['chunk_text'] = row['chunks'][i]['chunk_text'][1:]
                row['chunks'][i]['chunk_text'] = '...'.join(
                    row['chunks'][i]['chunk_text']
                )
        row['text'] = '...'.join([c['chunk_text'] for c in row['chunks']])
        del row['chunks']

    # Fusion results
    m_res = sorted(m_res, key=fusion_sort_key, reverse=True)
    m_res = m_res[:SUB_QUERY_COUNT]

    # Rerank results
    scores = rerank(topic, [row['text'][:4096] for row in m_res])
    for i in range(len(m_res)):
        m_res[i]['rank_score'] = scores[i]
    m_res = sorted(m_res, key=lambda x: x['rank_score'], reverse=True)
    m_res = m_res[:RESULTS_COUNT]

    # Print fusion results
    print("Fusion Results:")
    print(f"{'ID':<10}    {'Distance':<15}    {'Score':<15}    {'Rank Score':<15}    {'Title':<50}    {'URL':<50}    {'Text':<90}")
    print("-" * 200)
    for row in m_res:
        distance = f"{row['distance']:.4f}" if isinstance(
            row['distance'], (int, float)) else "N/A"
        score = f"{row['score']:.4f}" if isinstance(
            row['score'], (int, float)) else "N/A"
        url = row.get('url', 'N/A')
        if len(row['title']) >= 50:
            title = f"{row['title'][:47]}..."
        else:
            title = row['title']
        if len(url) >= 50:
            url = f"{url[:47]}..."
        text = row['text'].replace('\n', ' ')
        start = (len(text) - 80) // 2
        text = f"...{text[start:start + 80 - 6]}..."
        print(
            f"{row['id']:<10}    {distance:<15}    {score:<15}    {row['rank_score']:<15}    {title:<50}    {url:<50}    {text:<80}")

    return m_res

    # Using LLM to verify the results
    # print("> Using LLM to verify the results...")
    # llm_filter_pmt = f"""You are an intelligent AI document retrieval expert, and you have been asked a question: "{topic}".\n\nBelow is a document retrieval result that may relate to this question. Please read this document. Based on your judgment, if you find any information within the text that aids in answering the question, Extract and organize the useful information. If not, return an empty string. For matching documents, be sure to extract the corresponding content from the original text, rather than merely returning a summary. When organizing information, please ensure the original format and context flow smoothly, making the content relatively complete and readable. If it is in Markdown format, retain the Markdown hyperlinks and image references. If it is other formatted text like XML or HTML, please convert it to Markdown format. Please delete and output any other invalid formats and citation marks. For content you consider useless or of little use, please output an empty string. Please output in JSON format {{"result": "SUMMARY"}} or {{"result": ""}}. Here are the text excerpts you need to process:\n\n"""
    # task_hash = sha256(topic)
    # temp_path = tempfile.TemporaryDirectory(suffix=f'-{task_hash}')
    # llm_res = []
    # MAX_RESULTS = 10
    # for i, row in enumerate(m_res):
    #     try:
    #         file_name = os.path.join(temp_path.name, f"{row['id']}.json")
    #         download_file(row['snapshot'], file_name)
    #         file = read_json(file_name)
    #         file = file['text']
    #         row['snapshot_text'] = file
    #         llm_resp = generate(
    #             f"{llm_filter_pmt}{row['snapshot_text']}", json=True)
    #         llm_resp = json.loads(llm_resp)
    #         llm_resp = llm_resp.get('result', '')
    #         if llm_resp:
    #             row['summary'] = f'Title: {row["title"]}\n\nURL: {row["url"]}\n\nSummary:\n{llm_resp}'
    #             llm_res.append(row)
    #     except Exception as e:
    #         # print(f"Error: {e}")
    #         continue
    #     if len(llm_res) >= MAX_RESULTS:
    #         break

    # Print summary results
    # print(f'\n\n\nTopic: {topic}')
    # print("> Summary results:")
    # for row in llm_res:
    #     print(row['summary'],
    #           end='\n-------------------------------------------\n')
    # print('\n\nImage results:')
    # for row in is_res:
    #     print(f"{row['id']:<10}    {row['distance']:<15.6f}{row['similarity']:<15.6f}{row['title']:<50}    {row['url']:<50}")


if __name__ == "__main__":
    query("I want to know information about documentaries related to World War II.")

__all__ = [
    "query",
]
