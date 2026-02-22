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
    'patchwork', 'ggiraph', 'ggpubr', 'stochtree', 'CausalImpact', \
    'bsynth', 'rstan', 'bayesplot', 'posterior', 'vizdraws', 'glossary', \
    'furrr', 'tictoc', 'glue', 'shinydashboard', 'shinylive', 'remotes', \
    'google/imt', 'google/biva' \
))"

WORKDIR /book
COPY . .

# Set default command to render the Quarto project
CMD ["quarto", "render"]
