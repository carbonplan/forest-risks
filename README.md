<img
  src='https://carbonplan-assets.s3.amazonaws.com/monogram/dark-small.png'
  height='48'
/>

# carbonplan / forest-risks

**forest carbon potential and risks**

_Note: This project is under active development. We expect to make many breaking changes to the utilities and APIs included in this repository. Feel free to look around, but use at your own risk._

[![GitHub][github-badge]][github]
[![Build Status]][actions]
![MIT License][]

[github]: https://github.com/carbonplan/forest-risks
[github-badge]: https://badgen.net/badge/-/github?icon=github&label
[build status]: https://github.com/carbonplan/forest-risks/actions/workflows/main.yaml/badge.svg
[actions]: https://github.com/carbonplan/forest-risks/actions/workflows/main.yaml
[mit license]: https://badgen.net/badge/license/MIT/blue


This repository includes our libraries and scripts for mapping forest carbon potential and risks.

## install

```shell
pip install carbonplan[forests]
```

## usage

This codebase is organized into modules that implement data loading and model fitting as well as utitlies for plotting and other common tasks. Most anlayses involve some combination of the `load` and `fit` modules.

There are four scripts in the `scripts` folder that use these tools to import data, run models, and parse results.

- `biomass.py`
- `fire.py`
- `drought.py`
- `insects.py`

And there are two additional scripts `regrid.py` and `convert.py` for converting the results to zarr files for storage and geojson for visualization purposes.

Several notebooks are additionally provided that show the use of these tools for fitting models and inspecting model outputs. Notebooks are organized by the model type, e.g. `biomass`, `fire`, etc.

## data products

As part of this project we have created derived data products for five key variables relevant to evaluating forest carbon storage potential and risks.
- `biomass` The potential carbon storage in forests assuming continued growth of existing forests.
- `drought` The projected future impacts of drought on forests.
- `fire` The risks associated with forest fires.
- `insects` The risk to forests from insect-related tree mortality.
- `biophysical` The climate impact of forests by reflecting/absorbing light and transpiration.

Gridded rasters for each of these layers are available for the continental United States at a 4km spatial scale from the 20th century through the end of the 21st century in decadal increments. All data are accessible via this [catalog](https://github.com/carbonplan/forests/blob/master/carbonplan_forests/data/catalog.yaml). Additional formats and download options will be provided in the future.

## license

All the code in this repository is [MIT](https://choosealicense.com/licenses/mit/) licensed. When possible, the data used by this project is licensed using the [CC-BY-4.0](https://choosealicense.com/licenses/cc-by-4.0/) license. We include attribution and additional license information for third party datasets, and we request that you also maintain that attribution if using this data.

## about us

CarbonPlan is a non-profit organization that uses data and science for climate action. We aim to improve the transparency and scientific integrity of carbon removal and climate solutions through open data and tools. Find out more at [carbonplan.org](https://carbonplan.org/) or get in touch by [opening an issue](https://github.com/carbonplan/forest-risks/issues/new) or [sending us an email](mailto:hello@carbonplan.org).

## contributors

This project is being developed by CarbonPlan staff and the following outside contributors:

- Bill Anderegg (@anderegg)
- Grayson Badgley (@badgley)
- Anna Trugman
