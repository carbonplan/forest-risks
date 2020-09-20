import pathlib

def setup(store=None):
    if not home:
        raise ValueError('storage location not specified')
    if store == 'gcs':
        processed = pathlib.Path('gs://carbonplan-data/processed')
    elif store == 'az':
        processed = pathlib.Path('https://carbonplan.blob.core.windows.net/carbonplan-data/processed')
    elif store == 'local':
        processed = pathlib.Path('~/workdir/carbonplan')

    return processed