from lib.dataset import SQL_PROCESSED, init
from lib.s3 import get_url_by_key

SQL_SIMILARITY_CHECKED = 'similar_to != 0'
SQL_SIMILARITY_NOT_CHECKED = 'similar_to < 1'
SIMILARITY_THRESHOLD = 0.999
SIMILAR_BATCH = 1
MERGE_TARGET = 'alpha'


def pack_item(item):
    item['s3_url'] = get_url_by_key(item['processed_storage_id'])
    return item


def search_similar(
    ds,  vector, id=0, threshold=SIMILARITY_THRESHOLD, batch_size=SIMILAR_BATCH,
    pervious_result=[], recursion_depth=0, render=None, origin=None
):
    # Convert similarity threshold (0-1) to distance threshold (0-2)
    distance, ctn = 2 * (1 - threshold), False
    # str_time = time.time()
    resp = ds.query(
        f"""SELECT id, processed_storage_id,
        (vector <=> %s) as distance,
        ((2 - (vector <=> %s)) / 2) as similarity
        FROM {ds.get_table_name()}
        WHERE id > %s AND {SQL_PROCESSED} AND {SQL_SIMILARITY_NOT_CHECKED}
        ORDER BY (vector <=> %s) ASC OFFSET %s LIMIT %s""",
        (vector, vector, id, vector, len(pervious_result), batch_size)
    )
    # end_time = time.time()
    # print(f'>>> Query time: {end_time - str_time:.2f} seconds.')
    resp = [pack_item(item) for item in resp if item['distance'] <= distance]
    if resp:
        if resp[-1]['distance'] <= distance:
            ctn = True
        print(f'>>> Found {len(resp)} similar items in {ds.get_table_name()}'
              + f' with distance <= {distance} (threshold: {threshold})'
              + f' at recursion depth {recursion_depth}.')
        if render:
            render([*([origin] if origin else []), *resp])
    return search_similar(
        ds,  vector, id=id, threshold=threshold, batch_size=batch_size,
        pervious_result=pervious_result + resp,
        recursion_depth=recursion_depth + 1
    ) if ctn else pervious_result + resp


def search_similar_siglip(
    ds,  vector, id=0, threshold=0.1, batch_size=100,
    pervious_result=[], recursion_depth=0, render=None, origin=None
):
    # Convert similarity threshold (0-1) to distance threshold (0-2)
    distance, ctn = 2 * (1 - threshold), False
    # str_time = time.time()
    resp = ds.query(
        f"""SELECT id, processed_storage_id,
        (vector_siglip <=> %s) as distance,
        ((2 - (vector_siglip <=> %s)) / 2) as similarity
        FROM {ds.get_table_name()}
        ORDER BY (vector_siglip <=> %s) ASC OFFSET %s LIMIT %s""",
        (vector, vector, vector, len(pervious_result), batch_size)
    )
    # end_time = time.time()
    # print(f'>>> Query time: {end_time - str_time:.2f} seconds.')
    resp = [pack_item(item) for item in resp if item['distance'] <= distance]
    if resp:
        if resp[-1]['distance'] <= distance:
            ctn = True
        print(f'>>> Found {len(resp)} similar items in {ds.get_table_name()}'
              + f' with distance <= {distance} (threshold: {threshold})'
              + f' at recursion depth {recursion_depth}.')
        if render:
            render([*([origin] if origin else []), *resp])
    return search_similar_siglip(
        ds,  vector, id=id, threshold=threshold, batch_size=batch_size,
        pervious_result=pervious_result + resp,
        recursion_depth=recursion_depth + 1
    ) if ctn else pervious_result + resp


def get_next_item(ds, id=0):
    resp = ds.query(
        f'SELECT id, processed_storage_id, vector'
        + f' FROM {ds.get_table_name()}'
        + f' WHERE id > %s AND {SQL_PROCESSED} AND {SQL_SIMILARITY_NOT_CHECKED}'
        + ' ORDER BY id ASC LIMIT 1', (id,)
    )
    return pack_item(resp[0]) if resp else None


def restore_progress(ds):
    resp = ds.query(f'SELECT max(similar_to) FROM {ds.get_table_name()}')
    return max(resp[0]['max'] - 1, 0) if resp else 0


def reset_similarity(ds):
    print(f'>>> Resetting similarity data in {ds.get_table_name()}...')
    ds.query(
        f'UPDATE {ds.get_table_name()} SET similarity = 0, similar_to = 0'
        + f' WHERE {SQL_SIMILARITY_CHECKED}'
    )


def get_total_items(ds):
    resp = ds.query(f'SELECT max(id) FROM {ds.get_table_name()}')[0]['max']
    print(f'Total items in {ds.get_table_name()}: {resp}')
    return resp


def run_dataset(dataset, start_from=0, dry_run=True, reset=False, render=None):
    # https://stabilityai-public-packages.s3.us-west-2.amazonaws.com/Stable+Diffusion+3+Paper.pdf#page=13&zoom=100,48,457
    if dry_run and reset:
        raise ValueError("Dry run mode cannot be combined with reset option.")
    ds = init(dataset)
    if reset:
        reset_similarity(ds)
    total_items = get_total_items(ds)
    if not start_from:
        start_from = restore_progress(ds)
        # print(start_from)
        # import sys
        # sys.exit(0)
    cur_item = get_next_item(ds, id=start_from)
    while cur_item:
        progress = f"{round(cur_item['id'] / total_items * 100, 2)}%"
        print(
            f">>> Processing: {cur_item['id']} / {total_items}"
            + f" ({progress}) => {cur_item['s3_url']}"
        )
        resp = search_similar(
            ds, cur_item['vector'], cur_item['id'], render=render,
            origin=cur_item
        )
        for item in resp:
            print(f"    Found: {item['id']} ({item['similarity'] * 100:.2f}% "
                  + f"/ {item['distance']:.2f}) => {item['s3_url']}")
            if not dry_run:
                ds.query(
                    f'UPDATE {ds.get_table_name()}'
                    + f' SET similarity = %s, similar_to = %s'
                    + ' WHERE id = %s',
                    (item['similarity'], cur_item['id'], item['id'])
                )
        cur_item = get_next_item(ds, cur_item['id'])
    print('Done!')


def get_next_item_sc(ds, id=0):
    resp = ds.query(
        f'SELECT * FROM {ds.get_table_name()} WHERE id > %s '
        + ' ORDER BY id ASC LIMIT 1', (id,)
    )
    return pack_item(resp[0]) if resp else None


def check_similar(ds,  vector, threshold=SIMILARITY_THRESHOLD):
    # Convert similarity threshold (0-1) to distance threshold (0-2)
    distance = 2 * (1 - threshold)
    resp = ds.query(f"""SELECT *, (vector <=> %s) as distance,
        ((2 - (vector <=> %s)) / 2) as similarity FROM {ds.get_table_name()}
        ORDER BY (vector <=> %s) ASC LIMIT 1""", (vector, vector, vector))
    item = resp[0] if resp else None
    return pack_item(item) if item and (item['distance'] <= distance) else None


def expand_source(source):
    match source:
        case 'cc12m_cleaned' | 'cc12m_woman':
            return ['cc12m', source]
    return [source]


def merge_items(sc, tg, source):
    assert source, "Source dataset name must be provided."
    use_tg = tg['origin_width'] * tg['origin_height'] \
        >= sc['origin_width'] * sc['origin_height']
    print(
        f"    Merging: {source}[{sc['id']}] {['<=', '=>'][use_tg]} {MERGE_TARGET}[{tg['id']}]..."
    )
    return {
        'url': tg['url'] if use_tg else sc['url'],
        'caption': tg['caption'] or sc['caption'],
        'caption_long': tg['caption_long'] or sc['caption_long'],
        'origin_width': tg['origin_width'] if use_tg else sc['origin_width'],
        'origin_height': tg['origin_height'] if use_tg else sc['origin_height'],
        'origin_storage_id': tg['origin_storage_id'] if use_tg else sc['origin_storage_id'],
        'processed_storage_id': tg['processed_storage_id'] if use_tg else sc['processed_storage_id'],
        'processed_width': tg['processed_width'] if use_tg else sc['processed_width'],
        'processed_height': tg['processed_height'] if use_tg else sc['processed_height'],
        'aspect_ratio': tg['aspect_ratio'] if use_tg else sc['aspect_ratio'],
        'exif': {**sc['exif'], **tg['exif']},
        'meta': {**sc['meta'], **tg['meta']},
        'source': list(set(tg['source'] + expand_source(source))),
        'vector': tg['vector'] if use_tg else sc['vector'],
        'created_at': min(tg['created_at'], sc['created_at']),
        'updated_at': max(tg['updated_at'], sc['updated_at']),
    }


def absorb_dataset(dataset, target=MERGE_TARGET, start_from=0, dry_run=True, render=None):
    ds, tg = init(dataset), init(target)
    total_source_items, _ = get_total_items(ds), get_total_items(tg)
    cur_item = get_next_item_sc(ds, id=start_from)
    while cur_item:
        # print(cur_item)
        progress = f"{cur_item['id'] / total_source_items * 100:.2f}%"
        print(
            f">>> Processing: {cur_item['id']} / {total_source_items}"
            + f" ({progress}) => {cur_item['s3_url']}"
        )
        sim = check_similar(tg, cur_item['vector'])
        if sim:
            print(f"    Found: {sim['id']} ({sim['similarity'] * 100:.2f}% "
                  + f"/ {sim['distance']:.2f}) => {sim['s3_url']}")
            merged_item = merge_items(cur_item, sim, dataset)
            if not dry_run:
                tg.update_by_id(
                    sim['id'], merged_item, deplicate_ignore=['url']
                )
        else:
            print("    Not found, inserting...")
            if not dry_run:
                new_item = dict(cur_item)
                new_item['source'] = expand_source(dataset)
                new_item.pop('id', None)
                new_item.pop('s3_url', None)
                tg.insert(new_item, deplicate_ignore=['url'])
        cur_item = get_next_item_sc(ds, cur_item['id'])
    print('Done!')
    get_total_items(tg)


__all__ = [
    'run_dataset',
    'absorb_dataset',
    'search_similar',
    'search_similar_siglip',
]
