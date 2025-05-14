from psycopg_pool import AsyncConnectionPool
import aiohttp
import os
import asyncio
import time
import functools
from typing import Literal
from pydantic import BaseModel
from fastapi import UploadFile
from . import db_helper
import nltk

TABLE_NAME = "ts_wikipedia_en_embed"
IMAGE_TABLE_NAME = "is_pd12m"

pool = None
MAX_DISTANCE = 2
MAX_SCORE = 50
SELF_HOST_MODEL_SERVER_URL_BASE = os.getenv("MODEL_SERVER_URL_BASE", "http://localhost:8001")
MAX_RERANK_INPUT_LEN = 200
MAX_SUBQUERY_COUNT = 3

class QueryConfiguration(BaseModel):
    refine_query: bool = True
    rerank: bool = True
    vector_weight: float = 0.9
    bm25_weight: float = 0.1
    search_type: Literal["text", "image", "all"] = "text"

    class Config:
        json_schema_extra = {
            "example": {
                "refine_query": True,
                "rerank": True,
                "vector_weight": 0.9,
                "bm25_weight": 0.1,
                "search_type": "text"
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
    if os.environ.get('NLTK_PROXY'):
        nltk.set_proxy(os.environ['NLTK_PROXY'])
    nltk.download('punkt_tab')

async def clean():
    await clean_db()

def fusion_sort_key(result, vector_weight=0.9, bm25_weight=0.1):
    distance = result['distance'] if result['distance'] is not None else MAX_DISTANCE
    score = result['score'] if result['score'] is not None else 0
    normalized_score = score / MAX_SCORE
    normalized_distance = 1 - (distance / MAX_DISTANCE)
    fusion_score = vector_weight * normalized_distance + bm25_weight * normalized_score
    return fusion_score

async def refine_query(query: str) -> dict:
    headers = {"Content-Type": "application/json"}
    data = {
        "query": query
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{SELF_HOST_MODEL_SERVER_URL_BASE}/refine_query", json=data, headers=headers) as response:
            if response.status == 200:
                return await response.json()
            else:
                print("Error:", response.status, await response.text())
                return {} # Return empty dict on error

async def text_encode(input: list) -> list:
    headers = {"Content-Type": "application/json"}
    data = {
        "queries": input,
        "prompt_name": "query"
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{SELF_HOST_MODEL_SERVER_URL_BASE}/embedding", json=data, headers=headers) as response:
            if response.status == 200:
                return await response.json()
            else:
                print("Error:", response.status, await response.text())
                return []

async def text_encode_sig(input: list) -> list:
    headers = {"Content-Type": "application/json"}
    data = {
        "queries": input,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{SELF_HOST_MODEL_SERVER_URL_BASE}/siglip2/encode_text", json=data, headers=headers) as response:
            if response.status == 200:
                return await response.json()
            else:
                print("Error:", response.status, await response.text())
                return []

async def query_db(cur, sql=None, values=None, **kwargs):
    try:
        cursor = await cur.execute(sql, values, **kwargs)
        columns = [desc[0] for desc in cursor.description]
        results = await cursor.fetchall()
        return [dict(zip(columns, row)) for row in results]
    except Exception as e:
        print(f"Error executing or fetching from query: {e}")
        print(f"SQL: {sql}")
        print(f"Values: {values}")
        raise e

def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper():
                result = await func(*args, **kwargs)
                end_time = time.perf_counter()
                print(f"Function {func.__name__!r} executed in {end_time - start_time:.4f}s")
                return result
            return async_wrapper()
        else:
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            print(f"Function {func.__name__!r} executed in {end_time - start_time:.4f}s")
            return result
    return wrapper

@timeit
async def _perform_embedding_search(tp_resp, v_search_tmpl_builder, table_name):
    """Performs embedding search asynchronously."""
    print("> Embedding vector search...")
    eb_resp = await text_encode(tp_resp["sentences"])
    tasks = []
    if not eb_resp: # Handle empty response from text_encode
        print("Warning: Embedding response was empty.")
        return []
    for e in eb_resp:
        sql, values = v_search_tmpl_builder(table_name, e)
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
    return sorted(list(unique_e_res.values()), key=lambda x: x['distance'])

@timeit
async def _perform_bm25_search(tp_resp, bm25_search_tmpl_builder, table_name):
    """Performs BM25 search asynchronously."""
    print("> BM25 search...")
    bm25_tasks = []
    for b in tp_resp['keywords']:
        sql, values = bm25_search_tmpl_builder(table_name, b)
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
    return sorted(
        list(unique_b_res.values()),
        key=lambda x: x['score'],
        reverse=True
    )

@timeit
async def _perform_img_search(tp_resp, img_search_tmpl_builder):
    """Performs image search asynchronously."""
    print("> Image search...")
    ie_resp = await text_encode_sig(tp_resp['keywords'] + tp_resp['sentences']) # Use await
    tasks = []
    if not ie_resp: # Handle empty response
        print("Warning: Image embedding response was empty.")
    else:
        for ir in ie_resp:
            sql, values = img_search_tmpl_builder(IMAGE_TABLE_NAME, ir)
        tasks.append(
            asyncio.create_task(
                execute_query(
                    sql,
                    values
                )
            )
        )
    is_res = await asyncio.gather(*tasks)
    return is_res

async def execute_query(sql, values=None):
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            return await query_db(cur, sql, values)

@timeit
async def rerank(query, documents, max_results):
    headers = {"Content-Type": "application/json"}
    data = {
        "query": query,
        "documents": documents
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{SELF_HOST_MODEL_SERVER_URL_BASE}/rerank", json=data, headers=headers) as response:
            if response.status == 200:
                return await response.json()
            else:
                print("Error:", response.status, await response.text())
                # Return a list of zeros with the same length as documents on error
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

    # Initialize final results lists
    m_res_final = [] 
    is_res_final = []
    # Intermediate data storage for search results
    e_res_data, b_res_data, is_res_data = [], [], []

    tp_resp = {"sentences": [topic], "keywords": [topic]}

    if config.refine_query:
        print("> Refining query...")
        tp_resp = await refine_query(topic)
        if not tp_resp:
            print("Refined query is empty, using original topic.")
            tp_resp = {"sentences": [topic], "keywords": [topic]}
        tp_resp["sentences"] = tp_resp["sentences"][:MAX_SUBQUERY_COUNT]
        tp_resp["keywords"] = tp_resp["keywords"][:MAX_SUBQUERY_COUNT]

    # Conditionally perform searches based on search_type
    tasks_to_run = []
    embedding_search_task_idx, bm25_search_task_idx, img_search_task_idx = -1, -1, -1
    current_task_idx = 0

    if config.search_type in ["text", "all"]:
        tasks_to_run.append(asyncio.create_task(
            _perform_embedding_search(tp_resp, v_search_tmpl_builder, TABLE_NAME)
        ))
        embedding_search_task_idx = current_task_idx
        current_task_idx += 1

        tasks_to_run.append(asyncio.create_task(
            _perform_bm25_search(tp_resp, bm25_search_tmpl_builder, TABLE_NAME)
        ))
        bm25_search_task_idx = current_task_idx
        current_task_idx += 1
    
    if config.search_type in ["image", "all"]:
        tasks_to_run.append(asyncio.create_task(
            _perform_img_search(tp_resp, img_search_tmpl_builder)
        ))
        img_search_task_idx = current_task_idx

    if tasks_to_run:
        all_task_results = await asyncio.gather(*tasks_to_run)

        if embedding_search_task_idx != -1:
            e_res_data = all_task_results[embedding_search_task_idx]
        if bm25_search_task_idx != -1:
            b_res_data = all_task_results[bm25_search_task_idx]
        if img_search_task_idx != -1:
            is_res_data = all_task_results[img_search_task_idx] 

    # Process text results if requested
    if config.search_type in ["text", "all"]:
        print("> Fuse Score (Text)...")
        merged_text_results = {}
        for row in e_res_data:
            result_id = row['id']
            merged_text_results[result_id] = {**row, 'score': None}
        for row in b_res_data:
            result_id = row['id']
            if result_id in merged_text_results:
                merged_text_results[result_id].update(row)
            else:
                merged_text_results[result_id] = {**row, 'distance': None, 'similarity': None}
        
        if merged_text_results:
            m_res_intermediate = sorted(merged_text_results.values(), 
                                     key=lambda x: x['distance'] if x['distance'] is not None else MAX_DISTANCE)
            def sort_key_fn(result):
                return fusion_sort_key(result, config.vector_weight, config.bm25_weight)
            m_res_intermediate = sorted(m_res_intermediate, key=sort_key_fn, reverse=True)
            m_res_intermediate = m_res_intermediate[:max(MAX_RERANK_INPUT_LEN, max_results*10)]

            print("> Merge chunks (Text)...")
            merged_chunks_results = {}
            for row in m_res_intermediate:
                group_key = row['source_id']
                if group_key not in merged_chunks_results:
                    merged_chunks_results[group_key] = {**row, 'chunks': []}
                # Update aggregated scores/distances for the group
                merged_chunks_results[group_key]['distance'] = min(
                    merged_chunks_results[group_key]['distance'] if merged_chunks_results[group_key]['distance'] is not None else MAX_DISTANCE,
                    row['distance'] if row['distance'] is not None else MAX_DISTANCE)
                merged_chunks_results[group_key]['similarity'] = max(
                    merged_chunks_results[group_key]['similarity'] if merged_chunks_results[group_key]['similarity'] is not None else 0,
                    row['similarity'] if row['similarity'] is not None else 0)
                merged_chunks_results[group_key]['score'] = max(
                    merged_chunks_results[group_key]['score'] if merged_chunks_results[group_key]['score'] is not None else 0,
                    row['score'] if row['score'] is not None else 0)
                merged_chunks_results[group_key]['chunks'].append({'chunk_index': row['chunk_index'], 'chunk_text': row['chunk_text']})
                merged_chunks_results[group_key]['chunks'].sort(key=lambda x: x['chunk_index'])
                # Pop individual chunk data after aggregation if it's part of the row dict itself
                merged_chunks_results[group_key].pop('chunk_index', None)
                merged_chunks_results[group_key].pop('chunk_text', None)

            m_res_intermediate = sorted(merged_chunks_results.values(), 
                                     key=lambda x: x['distance'] if x['distance'] is not None else MAX_DISTANCE)
            for row in m_res_intermediate:
                for i, chunk in enumerate(row['chunks']):
                    if i > 0 and len(row['chunks']) > 1 and row['chunks'][i-1]['chunk_index'] == row['chunks'][i]['chunk_index'] - 1:
                        chunk_sentences = chunk_by_sentence(chunk['chunk_text'])
                        if len(chunk_sentences) >= 3:
                            chunk_sentences = chunk_sentences[1:]
                        row['chunks'][i]['chunk_text'] = '...'.join(chunk_sentences)
                row['text'] = '...'.join([c['chunk_text'] for c in row['chunks']])
                del row['chunks']

            if config.rerank and m_res_intermediate:
                print("> Reranking (Text)...")
                print("> Input size: ", len(m_res_intermediate))
                scores = await rerank(topic, [row['text'][:4096] for row in m_res_intermediate], max_results)
                for i in range(len(m_res_intermediate)):
                    m_res_intermediate[i]['rank_score'] = scores[i]
            else:
                for i in range(len(m_res_intermediate)):
                    m_res_intermediate[i]['rank_score'] = m_res_intermediate[i].get('score', 0)
            
            m_res_final = sorted(m_res_intermediate, key=lambda x: x['rank_score'], reverse=True)
            m_res_final = m_res_final[:max_results]

            print("Fusion Results (Text):")
            print(f"{'ID':<10}    {'Distance':<15}    {'Score':<15}    {'Rank Score':<15}    {'Title':<50}    {'URL':<50}    {'Text':<90}")
            print("-" * 200)
            for row in m_res_final:
                distance = f"{row['distance']:.4f}" if isinstance(row['distance'], (int, float)) else "N/A"
                score_val = f"{row['score']:.4f}" if isinstance(row['score'], (int, float)) else "N/A"
                url = row.get('url', 'N/A')
                title = f"{row['title'][:47]}..." if len(row['title']) >= 50 else row['title']
                url_display = f"{url[:47]}..." if len(url) >= 50 else url # Renamed to avoid conflict
                text_content = row['text'].replace('\n', ' ') # Renamed to avoid conflict
                start = max(0, (len(text_content) - 80) // 2)
                text_display = f"...{text_content[start:start + 80 - 6]}..." if len(text_content) > 80 else text_content
                print(
                    f"{row['id']:<10}    {distance:<15}    {score_val:<15}    {row['rank_score']:<15}    {title:<50}    {url_display:<50}    {text_display:<80}")
            for row in m_res_final: # Ensure final score is set
                row["score"] = row.get("rank_score", 0)
        else: # No text results from fusion
            m_res_final = []

    # Process image results if requested
    if config.search_type in ["image", "all"]:
        print("> Processing Image Results...")
        unique_img_results = {}
        # is_res_data is a list of lists of dicts from _perform_img_search
        for result_set in is_res_data: 
            for row in result_set:
                result_id = row['id']
                result_distance = row['distance']
                if result_id not in unique_img_results or result_distance < unique_img_results[result_id]['distance']:
                    unique_img_results[result_id] = row
        
        is_res_final = sorted(list(unique_img_results.values()), key=lambda x: x['distance'])
        is_res_final = is_res_final[:max_results]
        for row in is_res_final:
            row["score"] = row.get("similarity", 0) # Or calculate from distance if needed

    return m_res_final, is_res_final

async def image_encode_sig_from_file(uploaded_file: UploadFile) -> list:
    """
    Encodes an uploaded image file using the siglip2/encode_image endpoint.
    """
    form_data = aiohttp.FormData()
    form_data.add_field('files',
                       await uploaded_file.read(),
                       filename=uploaded_file.filename,
                       content_type=uploaded_file.content_type)
    
    # Ensure the file pointer is reset if read multiple times or passed around
    await uploaded_file.seek(0)

    async with aiohttp.ClientSession() as session:
        async with session.post(f"{SELF_HOST_MODEL_SERVER_URL_BASE}/siglip2/encode_image", data=form_data) as response:
            if response.status == 200:
                # Assuming model server returns List[List[float]] (list of embeddings)
                raw_embeddings = await response.json()
                return raw_embeddings
            else:
                print(f"Error encoding image {uploaded_file.filename}: {response.status} {await response.text()}")
                return []

async def image_query(image_file: UploadFile, max_results: int):
    """
    Performs an image-based query using an uploaded image file.
    Finds similar images from the IMAGE_TABLE_NAME.
    """

    # Dynamically construct the function name based on TABLE_NAME
    img_search_fn_name = f""f"tempalte_vector_search_{IMAGE_TABLE_NAME}"
    if not hasattr(db_helper, img_search_fn_name):
        raise AttributeError(f"Function '{img_search_fn_name}' not found in db_helper.")
    img_search_tmpl_builder = getattr(db_helper, img_search_fn_name)

    await init() # Ensure DB pool and other initializations are done

    print(f"Image query received for file: {image_file.filename}, max_results: {max_results}")

    img_embeddings = await image_encode_sig_from_file(image_file)

    if not img_embeddings:
        print("No embeddings generated for the image.")
        return [], []

    # Dynamically get the image search template builder
    img_search_fn_name = f"tempalte_vector_search_{IMAGE_TABLE_NAME}"
    if not hasattr(db_helper, img_search_fn_name):
        raise AttributeError(f"Function '{img_search_fn_name}' not found in db_helper for table {IMAGE_TABLE_NAME}.")
    img_search_tmpl_builder = getattr(db_helper, img_search_fn_name)

    tasks = []
    for emb_data in img_embeddings:
        sql, values = img_search_tmpl_builder(IMAGE_TABLE_NAME, emb_data)
        tasks.append(
            asyncio.create_task(
                execute_query(
                    sql,
                    values
                )
            )
        )
    
    query_results_list = await asyncio.gather(*tasks)

    # Process and merge results from potentially multiple embedding vectors (though usually one for a single image)
    unique_img_results = {}
    for result_set in query_results_list:
        for row in result_set:
            result_id = row['id'] # Assuming 'id' is the unique identifier for images
            result_distance = row.get('distance') # Assuming distance is returned

            if result_distance is None: # Skip if no distance
                continue

            if result_id not in unique_img_results or result_distance < unique_img_results[result_id]['distance']:
                unique_img_results[result_id] = row
    # Sort by distance and take top N
    sorted_images = sorted(list(unique_img_results.values()), key=lambda x: x['distance'])
    final_image_results = sorted_images[:max_results]

    # Add 'score' field, e.g., based on similarity (1 - normalized_distance) or directly from model if available
    for row in final_image_results:
        row["score"] = row.get("similarity", 1.0 - (row['distance'] if row['distance'] is not None else MAX_DISTANCE) / MAX_DISTANCE)
        # Ensure all required fields for SearchResultImageItem are present or have defaults
        row.setdefault('caption', row.get('title', 'N/A')) # Example: use title if caption missing
        row.setdefault('processed_storage_id', 'N/A')
        row.setdefault('aspect_ratio', 1.0)
        row.setdefault('exif', {})
        row.setdefault('source', [])
    # For image_query, text results are typically empty.
    return [], final_image_results


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(query("what is an ai model"))
