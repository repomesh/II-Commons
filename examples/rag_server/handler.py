from psycopg_pool import AsyncConnectionPool
import requests
import os

# TABLE_NAME = "ts_text_0000002_en"
TABLE_NAME = "ts_ms_marco"
pool = None
MAX_DISTANCE = 2
MAX_SCORE = 50
TEXT_EMBEDDING_MODEL = 'Snowflake/snowflake-arctic-embed-m-v2.0'

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
    base_url = os.getenv("MODEL_SERVER_URL_BASE", "http://localhost:8001")

    headers = {"Content-Type": "application/json"}
    data = {
        "queries": input,
        "prompt_name": "query"
    }
    response = requests.post(f"{base_url}/embedding", json=data, headers=headers)


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

async def query(topic, max_results=100):
    await init()

    tp_resp = {"sentences": [topic], "keywords": [topic]}
    # Embedding phrases
    print("> Embedding phrases...")
    eb_resp = text_encode(tp_resp["sentences"])

    # Embedding phrases search
    print("> Embedding vector search...")
    tasks = []
    for e in eb_resp:
        tasks.append(
            asyncio.create_task(
                execute_query(
                    f"""SELECT id, item_id, answers, query, passage_text, is_selected, url, query_id, query_type, well_formed_answers, passage_id,
                    (vector <=> %s::vector) as distance,
                    ((2 - (vector <=> %s::vector)) / 2) as similarity
                    FROM {TABLE_NAME} order by (vector <=> %s::vector) ASC OFFSET %s LIMIT %s""",
                    (e, e, e, 0, 100)
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
        bm25_tasks.append(
            asyncio.create_task(
                execute_query(
                    f"""SELECT id, item_id, answers, query, passage_text, is_selected, url, query_id, query_type, passage_id,
                    well_formed_answers, paradedb.score(passage_text) as score
                    FROM {TABLE_NAME} WHERE passage_text @@@ %s
                    ORDER BY score DESC OFFSET %s LIMIT %s""",
                    (b, 0, 100)
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

    m_res = sorted(m_res, key=fusion_sort_key, reverse=True)
    m_res = m_res[:max_results]

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
    import asyncio
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(query("what cut is a bolar roast"))

__all__ = [
    "query",
]
