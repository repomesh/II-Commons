from lib.dataset import init
from lib.embedding import normalize
from lib.psql import batch_insert

ds = init('pd12m')
last_id = 0
BATCH_SIZE = 1000


def get_unprocessed():
    return ds.query(
        f"SELECT id, vector FROM {ds.get_table_name()}"
        + ' WHERE id > %s AND vector IS NOT NULL AND vector_2 IS NULL'
        + ' ORDER BY id ASC LIMIT %s',
        (last_id, BATCH_SIZE)
    )


def run():
    global last_id
    items = get_unprocessed()
    # print(items)
    while len(items) > 0:
        print(f"⚙️ Processing rows from {items[0]['id']} to {items[-1]['id']}")
        data = []
        vector_2 = normalize([x['vector'] for x in items])
        for i in range(len(items)):
            data.append((vector_2[i], items[i]['id']))
            # ds.update_by_id(items[i]['id'], {'vector_2': vector_2[i]})
        # print('✏️ Updating database...')
        batch_insert(
            f"UPDATE {ds.get_table_name()} SET vector_2 = %s WHERE id = %s",
            data
        )
        last_id = items[-1]['id']
        # print(f"✅ Done.")
        items = get_unprocessed()


if __name__ == '__main__':
    run()
