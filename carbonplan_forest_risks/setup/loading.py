import pathlib

import urlpath


def loading(store=None):
    if store is None:
        raise ValueError('data store not specified')
    if store == 'gs':
        base = urlpath.URL('gs://')
    elif store == 'az':
        base = urlpath.URL('https://carbonplan.blob.core.windows.net')
    elif store == 'local':
        base = pathlib.Path(pathlib.Path.home() / 'workdir')

    return base
