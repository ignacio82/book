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
    'ignacio82/longbet@v0.7.2', \
    'shinylive@0.5.0' \
    ))"

# Bake the shinylive WebAssembly bundle into the image. The Quarto extension
# downloads it mid-render if it is missing, which turns every render into a
# network-dependent operation; doing it here makes local renders reproducible
# and offline-safe, and matches what CI pre-fetches.
RUN R -q -e "shinylive::assets_ensure()"

# install.packages() does not set a non-zero exit status when a package fails
# to build, so a broken compile would otherwise produce an image that only
# fails much later, during the render, with a confusing error. Fail here.
RUN R -q -e "stopifnot( \
    requireNamespace('longbet', quietly = TRUE), \
    packageVersion('longbet') >= '0.7.2', \
    is.function(longbet::get_catt), \
    is.function(longbet::att_stability), \
    is.function(longbet::longbet_multi), \
    is.function(longbet::joint_prob), \
    'longbet_multi_cpp' %in% ls(asNamespace('longbet')), \
    'treat_effect_re' %in% names(formals(longbet::longbet)), \
    is.function(longbet::predict.longbet), \
    'x_tv_trt' %in% names(formals(longbet::longbet)) \
    ); cat('longbet', as.character(packageVersion('longbet')), 'ok\n')"

WORKDIR /book
COPY . .

# Set default command to render the Quarto project
CMD ["quarto", "render"]
