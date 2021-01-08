from .nlcd import nlcd


def mask(store='az', year=2001, coarsen=None):
    bands = nlcd(store=store, classes='all', year=year)
    if coarsen:
        bands = bands.coarsen(x=coarsen, y=coarsen, boundary='trim').mean()
    return bands.sum('band')
