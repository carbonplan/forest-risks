import pathlib

import urlpath


def loading(store=None):
    if store is None:
        raise ValueError('data store not specified')
    if store == 'gcs':
        base = urlpath.URL('gs://carbonplan-data')
    elif store == 'az':
        base = urlpath.URL('https://carbonplan.blob.core.windows.net/carbonplan-data')
    elif store == 'local':
        base = pathlib.Path(pathlib.Path.home() / 'workdir/carbonplan-data')

    return base
