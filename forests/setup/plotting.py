import altair as alt


def plotting():
    alt.renderers.enable('default', embed_options={'actions': False})
    alt.data_transformers.enable('data_server')
    alt.data_transformers.disable_max_rows()
