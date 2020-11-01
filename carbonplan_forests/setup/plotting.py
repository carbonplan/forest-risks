import altair as alt


def plotting(remote=False):
    alt.renderers.enable('default', embed_options={'actions': False})
    if remote:
        alt.data_transformers.enable('data_server_proxied', urlpath='/user-redirect')
    else:
        alt.data_transformers.enable('data_server')
    alt.data_transformers.disable_max_rows()
