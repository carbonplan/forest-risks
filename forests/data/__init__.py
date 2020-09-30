from intake import open_catalog
import pathlib


cat_dir = pathlib.Path(__file__)
cat_file = str(cat_dir.parent / "catalog.yaml")
cat = open_catalog(cat_file)
