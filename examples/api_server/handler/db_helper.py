def escape_bm25_query_simple(q: str) -> str:
    if not isinstance(q, str):
        return q
    return q.replace("'", r"\'")

SUB_QUERY_COUNT = 300

def tempalte_vector_search_ts_ms_marco(table_name: str, e: any) -> tuple:
    sql = f"""SELECT id, item_id, answers, query, passage_text, is_selected, url, query_id, query_type, well_formed_answers, passage_id,
        (vector <=> %s::vector) as distance,
        ((2 - (vector <=> %s::vector)) / 2) as similarity
        FROM {table_name} order by (vector <=> %s::vector) ASC OFFSET %s LIMIT %s"""
    values = (e, e, e, 0, SUB_QUERY_COUNT)
    return sql, values
        
def tempalte_bm25_search_ts_ms_marco(table_name: str, b: any) -> tuple:
    b = escape_bm25_query_simple(b)
    sql = f"""SELECT id, item_id, answers, query, passage_text, is_selected, url, query_id, query_type, passage_id,
                    well_formed_answers, paradedb.score(passage_text) as score
                    FROM {table_name} WHERE passage_text @@@ %s
                    ORDER BY score DESC OFFSET %s LIMIT %s"""
    values = (b, 0, SUB_QUERY_COUNT)
    return sql, values


def tempalte_vector_search_ts_wikipedia_en_embed(table_name: str, e: any) -> tuple:
    sql = f"""SELECT id, title, url, snapshot, source_id, chunk_index, chunk_text,
            (vector <=> %s::vector) as distance,
            ((2 - (vector <=> %s::vector)) / 2) as similarity
            FROM {table_name} ORDER BY (vector <=> %s::vector) ASC OFFSET %s LIMIT %s"""
    values = (e, e, e, 0, SUB_QUERY_COUNT)

    return sql, values
 
def tempalte_bm25_search_ts_wikipedia_en_embed(table_name: str, b: any) -> tuple:
    b = escape_bm25_query_simple(b)
    sql = f"""SELECT id, title, url, snapshot, source_id, chunk_index, chunk_text, paradedb.score(id) as score
            FROM {table_name} WHERE title @@@ %s or chunk_text @@@ %s
            ORDER BY score DESC OFFSET %s LIMIT %s"""
    values = (b, b, 0, SUB_QUERY_COUNT)
    return sql, values


def tempalte_vector_search_is_pd12m(table_name: str, e: any) -> tuple:
    sql = f"""SELECT id, url, caption, processed_storage_id, aspect_ratio, exif, source,
            (vector <-> %s::vector) as distance,
            (1 / (1 + (vector <-> %s::vector))) AS similarity
            FROM {table_name} ORDER BY (vector <-> %s::vector) ASC OFFSET %s LIMIT %s"""
    values = (e, e, e, 0, SUB_QUERY_COUNT)

    return sql, values
