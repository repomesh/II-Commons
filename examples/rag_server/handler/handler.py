from psycopg_pool import AsyncConnectionPool
import requests
import os
import asyncio
from pydantic import BaseModel
from . import db_helper
import nltk

TABLE_NAME = "ts_text_0000002_en"
IMAGE_TABLE_NAME = "ii_alpha"

# TABLE_NAME = "ts_ms_marco"
pool = None
MAX_DISTANCE = 2
MAX_SCORE = 50
TEXT_EMBEDDING_MODEL = 'Snowflake/snowflake-arctic-embed-m-v2.0'
RERANK_MODEL = 'jinaai/jina-reranker-m0'
SELF_HOST_MODEL_SERVER_URL_BASE = os.getenv("MODEL_SERVER_URL_BASE", "http://localhost:8001")
USE_JINA_RERANK_API = os.getenv("USE_JINA_RERANK_API", "false").lower() == "true"
if USE_JINA_RERANK_API:
    print("> Using Jina Rerank API...")
class QueryConfiguration(BaseModel):
    refine_query: bool = False
    rerank: bool = True

    class Config:
        json_schema_extra = {
            "example": {
                "refine_query": False,
                "rerank": False,
            }
        }

async def init_db():
    global pool
    if pool is not None:
        return
    POSTGRES_HOST = os.getenv('POSTGRES_HOST')
    POSTGRES_PORT = os.getenv('POSTGRES_PORT')
    POSTGRES_USER = os.getenv('POSTGRES_USER')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
    POSTGRES_DB = os.getenv('POSTGRES_DB')
    if not all([POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB]):
        raise EnvironmentError("One or more required environment variables are missing.")

    print(f"> Connecting to PostgreSQL at {POSTGRES_HOST}:{POSTGRES_PORT}...")
    print(f"> Using database: {POSTGRES_DB}")
    print(f"> Using table: {TABLE_NAME}")
    print(f"> Using text embedding model: {TEXT_EMBEDDING_MODEL}")
    print(f"> Using max distance: {MAX_DISTANCE}")
    print(f"> Using max score: {MAX_SCORE}")
    pool = AsyncConnectionPool(
        conninfo=f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}',
        open=False, min_size=3, max_size=100000
    )
    await pool.open()

async def clean_db():
    global pool
    if pool:
        await pool.close()
        pool = None

async def init():
    await init_db()
    nltk.download('punkt_tab')

async def clean():
    await clean_db()

def fusion_sort_key(result):
    distance = result['distance'] if result['distance'] is not None else MAX_DISTANCE
    score = result['score'] if result['score'] is not None else 0
    normalized_score = score / MAX_SCORE
    normalized_distance = 1 - (distance / MAX_DISTANCE)
    vector_weight = 0.6
    bm25_weight = 0.4
    fusion_score = vector_weight * normalized_distance + bm25_weight * normalized_score
    return fusion_score

def text_encode(input: list) -> list:
    headers = {"Content-Type": "application/json"}
    data = {
        "queries": input,
        "prompt_name": "query"
    }
    response = requests.post(f"{SELF_HOST_MODEL_SERVER_URL_BASE}/embedding", json=data, headers=headers)


    if response.status_code == 200:
        # print("Embeddings:", response.json())
        return response.json()
    else:
        print("Error:", response.status_code, response.text)
        return []
    
def text_encode_sig(input: list) -> list:
    headers = {"Content-Type": "application/json"}
    data = {
        "queries": input,
    }
    response = requests.post(f"{SELF_HOST_MODEL_SERVER_URL_BASE}/siglip2/encode_text", json=data, headers=headers)

    if response.status_code == 200:
        # print("Embeddings:", response.json())
        return response.json()
    else:
        print("Error:", response.status_code, response.text)
        return []
    


async def query_db(cur, sql=None, values=None, **kwargs):
    cursor = await cur.execute(sql, values, **kwargs)
    try:
        columns = [desc[0] for desc in cursor.description]
        results = await cursor.fetchall()
        return [dict(zip(columns, row)) for row in results]
    except Exception as e:
        pass
    return cursor

async def execute_query(sql, values=None):
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            return await query_db(cur, sql, values)
        
async def refine_query(topic):
    return {"sentences": [topic], "keywords": [topic]}

def rerank(query, documents, max_results):
    if USE_JINA_RERANK_API:
        JINA_API_KEY = os.getenv("JINA_API_KEY")
        if not JINA_API_KEY:
            raise EnvironmentError("JINA_API_KEY environment variable is not set.")
        url = 'https://api.jina.ai/v1/rerank'

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {JINA_API_KEY}"
        }

        docs = [{"text": doc} for doc in documents]
        data = {
            "model": "jina-reranker-m0",
            "query": "small language model data extraction",
            "top_n": f"{max_results}",
            "documents": docs
        }
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            resp = response.json()
            scores = [0] * len(documents)
            for item in resp['results']:
                index = item["index"]
                score = item["relevance_score"]
                if index < len(documents):
                    scores[index] = score
            return scores
        else:
            print("Error:", response.status_code, response.text)
            return [0] * len(documents)
    else:
        headers = {"Content-Type": "application/json"}
        data = {
            "query": query,
            "documents": documents
        }
        response = requests.post(f"{SELF_HOST_MODEL_SERVER_URL_BASE}/rerank", json=data, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            print("Error:", response.status_code, response.text)
            return [0] * len(documents)

def chunk_by_sentence(document):
    return nltk.sent_tokenize(document)

async def query(topic, max_results=100, config=QueryConfiguration()):
    # Dynamically construct the function name based on TABLE_NAME
    v_search_fn_name = f"tempalte_vector_search_{TABLE_NAME}"
    bm25_search_fn_name = f"tempalte_bm25_search_{TABLE_NAME}"
    img_search_fn_name = f""f"tempalte_vector_search_{IMAGE_TABLE_NAME}"

    if not hasattr(db_helper, v_search_fn_name):
        raise AttributeError(f"Function '{v_search_fn_name}' not found in db_helper.")
    if not hasattr(db_helper, bm25_search_fn_name):
        raise AttributeError(f"Function '{bm25_search_fn_name}' not found in db_helper.")
    if not hasattr(db_helper, img_search_fn_name):
        raise AttributeError(f"Function '{img_search_fn_name}' not found in db_helper.")
    v_search_tmpl_builder = getattr(db_helper, v_search_fn_name)
    bm25_search_tmpl_builder = getattr(db_helper, bm25_search_fn_name)
    img_search_tmpl_builder = getattr(db_helper, img_search_fn_name)
    await init()

    tp_resp = {"sentences": [topic], "keywords": [topic]}

    if refine_query:
        print("> Refining query...")
        tp_resp = await refine_query(topic)
        if not tp_resp:
            print("Refined query is empty, using original topic.")
            tp_resp = {"sentences": [topic], "keywords": [topic]}

    # Embedding phrases
    print("> Embedding phrases...")
    eb_resp = text_encode(tp_resp["sentences"])

    # Embedding phrases search
    print("> Embedding vector search...")
    tasks = []
    for e in eb_resp:
        sql, values = v_search_tmpl_builder(TABLE_NAME, e)
        tasks.append(
            asyncio.create_task(
                execute_query(
                    sql,
                    values
                )
            )
        )
    e_res = await asyncio.gather(*tasks)

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
    bm25_tasks = []
    for b in tp_resp['keywords']:
        sql, values = bm25_search_tmpl_builder(TABLE_NAME, b)
        bm25_tasks.append(
            asyncio.create_task(
                execute_query(
                    sql,
                    values
                )
            )
        )
    b_res = await asyncio.gather(*bm25_tasks)

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

 
    # Merge chunks
    print("> Merge chunks...")
    merged_results = {}
    for row in m_res:
        group_key = (row['source_id'], row['source_db'])
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
    m_res = sorted(m_res, key=fusion_sort_key, reverse=True)

    if config.rerank:
        print("> Reranking...")
        # Rerank the results based on the fusion score
        scores = rerank(topic, [row['text'][:4096] for row in m_res], max_results)
        for i in range(len(m_res)):
            m_res[i]['rank_score'] = scores[i]
    else:
        for i in range(len(m_res)):
            m_res[i]['rank_score'] = m_res[i].get('score', 0)
    m_res = sorted(m_res, key=lambda x: x['rank_score'], reverse=True)
    m_res = m_res[:max_results]

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


    # # Image search
    # print("> Image search...")
    # ie_resp = text_encode_sig(tp_resp['keywords'] + tp_resp['sentences'])
    # tasks = []
    # for ir in ie_resp:
    #     sql, values = img_search_tmpl_builder(IMAGE_TABLE_NAME, ir)
    #     tasks.append(
    #         asyncio.create_task(
    #             execute_query(
    #                 sql,
    #                 values
    #             )
    #         )
    #     )
    # is_res = await asyncio.gather(*tasks)


    # unique_is_res = {}
    # for result_set in is_res:
    #     for row in result_set:
    #         result_id = row['id']
    #         result_distance = row['distance']
    #         if result_id not in unique_is_res or result_distance < unique_is_res[result_id]['distance']:
    #             unique_is_res[result_id] = row
    # is_res = sorted(list(unique_is_res.values()), key=lambda x: x['distance'])
    # is_res = is_res[:max_results]

    # return [], is_res

    is_res = []
    return m_res, is_res


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(query("what cut is a bolar roast"))