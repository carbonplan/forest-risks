# carbonplan / forest-risks / R code

These two files implement our models related to insect and drought mortality. These analyses are implemented in R, whereas our fire and biomass models are implemented in Python.

Two scripts are included.

`Figure-1-Drought-Insect-Historical-Models.R` fits models of drought- and insect-related mortality to historical data. This script generates output files which we then consume to create Zarr files that support our final figures (see `notebooks/paper`).

`Figure-2-4-Drought-Insect-Projections.R` uses those models to project drought- and insect-related mortality in future climates. This script generates output files which we again consume to create Zarr files that support our final figures (see `notebooks/paper`).

The input data for these scripts is based on preprocessing versions of FIA, TerraClimate, and CMIP6. The code for that preprocessing is constained in this repository, and the files are available on cloud storage and referened directly within the scripts.
