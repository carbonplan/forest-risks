<img
  src='https://carbonplan-assets.s3.amazonaws.com/monogram/dark-small.png'
  height='48'
/>

# carbonplan / forests

**forest carbon potential and risks**

_Note: This project is under active development. We expect to make many breaking changes to the utilities and APIs included in this repository. Feel free to look around, but use at your own risk._

[![GitHub][github-badge]][github]
![Build Status][]
![MIT License][]

[github]: https://github.com/carbonplan/forests
[github-badge]: https://flat.badgen.net/badge/-/github?icon=github&label
[build status]: https://flat.badgen.net/github/checks/carbonplan/forests
[mit license]: https://flat.badgen.net/badge/license/MIT/blue

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

## license

All the code in this repository is [MIT](https://choosealicense.com/licenses/mit/) licensed. Some of the data provided by this API is sourced from content made available under a [CC-BY-4.0](https://choosealicense.com/licenses/cc-by-4.0/) license. We include attribution for this content, and we please request that you also maintain that attribution if using this data.

## about us

CarbonPlan is a non-profit organization that uses data and science for carbon removal. We aim to improve the transparency and scientific integrity of carbon removal and climate solutions through open data and tools. Find out more at [carbonplan.org](https://carbonplan.org/) or get in touch by [opening an issue](https://github.com/carbonplan/forests/issues/new) or [sending us an email](mailto:hello@carbonplan.org).

## contributors

This project is being developed by CarbonPlan staff and the following outside contributors:

- Bill Anderegg (@anderegg)
- Grayson Badgley (@badgley)
- Anna Trugman
