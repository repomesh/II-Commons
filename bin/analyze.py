from lib.dataset import analyze, get_datasets, init


def run():
    resp_ds = get_datasets()

    print('\nDataset Report')
    print(f'\nDatasets ({len(resp_ds)}):', ', '.join(resp_ds))

    for ds_name in resp_ds:
        print('\n\n>>> Dataset:', ds_name)
        ds = init(ds_name)
        resp_al = analyze(ds)
        for key, val in resp_al.items():
            print(f'  - {key}: {val}')


if __name__ == "__main__":
    run()

__all__ = [
    "run",
]
