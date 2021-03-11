# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
ARG BASE_CONTAINER=jupyter/minimal-notebook
FROM $BASE_CONTAINER

LABEL maintainer="Jupyter Project <jupyter@googlegroups.com>"

USER root

# R pre-requisites
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    fonts-dejavu \
    unixodbc \
    unixodbc-dev \
    r-cran-rodbc \
    gfortran \
    gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Fix for devtools https://github.com/conda-forge/r-devtools-feedstock/issues/4
RUN ln -s /bin/tar /bin/gtar

USER $NB_UID

# R packages
RUN conda install --quiet --yes \
    'r-base=4.0.3' \
    'r-caret=6.*' \
    'r-crayon=1.3*' \
    'r-devtools=2.3*' \
    'r-forecast=8.13*' \
    'r-hexbin=1.28*' \
    'r-htmltools=0.5*' \
    'r-htmlwidgets=1.5*' \
    'r-irkernel=1.1*' \
    'r-nycflights13=1.0*' \
    'r-randomforest=4.6*' \
    'r-rcurl=1.98*' \
    'r-rmarkdown=2.6*' \
    'r-rodbc=1.3*' \
    'r-rsqlite=2.2*' \
    'r-shiny=1.5*' \
    'r-tidyverse=1.3*' \
    'unixodbc=2.3.*' \
    'r-tidymodels=0.1*' \
    'r-rgdal=1.5*' \
    'r-rjson=0.2*' \
    'r-units=0.6*' \
    'r-rjags=4*' \
    'r-r2jags=0.6*' \
    'r-sf=0.9*' \
    'r-tweenr=1.0*' \
    'r-dtplyr=1.0*' \
    'r-gganimate=1.0*' \
    'r-remotes=2.2*' \
    'r-rgeos=0.5*' \
    'r-lsd=4.' \
    'r-mass=7.*' \
    'r-sdmtools=1.1*' \
    'r-raster=3.4*' \
    'r-mumin=1.43*' \
    'r-rnetcdf=2.4*' \
    'r-scales=1.1*' \
    'r-car=3.0*' \
    'r-pscl=1.5*' \
    'r-nlme=3.1*' \
    && \
    conda clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

RUN R -e "remotes::install_github('hunter-stanke/rFIA@aaa0ff429696a1c94f81525d142d269f6f7676a5')"
# RUN R -e "install.packages('rworldmap', dependencies=TRUE)"

# Install e1071 R package (dependency of the caret R package)
RUN conda install --quiet --yes r-e1071
