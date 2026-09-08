FROM rocker/verse:latest

# Install system dependencies required by rstan, biva, and other packages
RUN apt-get update && apt-get install -y \
    libv8-dev \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pak for fast, dependency-resolved R package installation
RUN Rscript -e "install.packages('pak', repos = sprintf('https://r-lib.github.io/p/pak/stable/%s/%s/%s', .Platform\$pkgType, R.Version()\$os, R.Version()\$arch))"

# Install all CRAN and GitHub packages required for the book
RUN Rscript -e "pak::pkg_install(c( \
    'knitr', 'rmarkdown', 'downlit', 'xml2', 'ggplot2', 'dplyr', \
    'tidyr', 'tibble', 'purrr', 'tidyverse', 'broom', 'scales', \
    'patchwork', 'zoo', 'ggiraph', 'ggpubr', 'stochtree', 'CausalImpact', \
    'bsynth', 'rstan', 'bayesplot', 'posterior', 'vizdraws', 'glossary', \
    'furrr', 'tictoc', 'glue', 'shinydashboard', 'remotes', \
    'MatchIt', 'coda', 'rpart', 'rpart.plot', 'shiny', 'shinybusy', \
    'arm', 'future', 'lubridate', 'brms', 'google/imt', 'google/biva', \
    'reticulate', \
    'shinylive@0.5.0' \
    ))"

# Bake the shinylive WebAssembly bundle into the image. The Quarto extension
# downloads it mid-render if it is missing, which turns every render into a
# network-dependent operation; doing it here makes local renders reproducible
# and offline-safe, and matches what CI pre-fetches.
RUN R -q -e "shinylive::assets_ensure()"

# LongBet is the JAX engine (ignacio82/longbet-jax): a Python sampler plus a
# thin R front door that reaches it through reticulate. The retired C++ package
# (ignacio82/longbet, <= v0.7.2) is deliberately not installed -- it shares the
# R package name, so installing both would silently decide which engine the
# chapter runs on. Pin LONGBET_REF to a commit for a reproducible build.
ARG LONGBET_REF=main
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev git \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/longbet-venv \
    && /opt/longbet-venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/longbet-venv/bin/pip install --no-cache-dir \
        "longbet-jax @ git+https://github.com/ignacio82/longbet-jax.git@${LONGBET_REF}"

# reticulate must resolve to that venv for every R session in the image.
ENV RETICULATE_PYTHON=/opt/longbet-venv/bin/python

# The R front door lives in the same repository as the Python engine, so both
# come from one ref and cannot drift apart.
RUN R -q -e "remotes::install_github('ignacio82/longbet-jax', ref = Sys.getenv('LONGBET_REF', 'main'), upgrade = 'never')"

# install.packages() does not set a non-zero exit status when a package fails
# to build, so a broken compile would otherwise produce an image that only
# fails much later, during the render, with a confusing error. Fail here.
# Both halves are checked: an R package that loads but cannot reach the Python
# engine passes every is.function() test and then fails at the first fit.
RUN R -q -e "stopifnot( \
    requireNamespace('longbet', quietly = TRUE), \
    packageVersion('longbet') >= '1.0.0', \
    is.function(longbet::get_catt), \
    is.function(longbet::att_stability), \
    is.function(longbet::get_att), \
    is.function(longbet::getTaus), \
    is.function(longbet::getMus), \
    is.function(longbet::plot_rollout), \
    is.function(longbet::longbet), \
    is.function(longbet::longbet_multi), \
    is.function(longbet::joint_prob), \
    is.function(longbet::outcome_correlation), \
    is.function(getS3method('predict', 'longbet', envir = asNamespace('longbet'))), \
    is.function(getS3method('predict', 'longbet_multi', envir = asNamespace('longbet'))), \
    'cache_forest_evaluations' %in% names(formals(getS3method('predict', 'longbet', envir = asNamespace('longbet')))) \
    ); \
    py <- reticulate::import('longbet'); \
    stopifnot(reticulate::py_has_attr(py, 'LongBet'), \
              reticulate::py_has_attr(py, 'LongBetMulti'), \
              reticulate::py_has_attr(py, 'att_stability')); \
    cat('longbet R', as.character(packageVersion('longbet')), \
        '/ python', py[['__version__']], 'ok\n')"

>>>>>>> longbet
WORKDIR /book
COPY . .

# Set default command to render the Quarto project
CMD ["quarto", "render"]
