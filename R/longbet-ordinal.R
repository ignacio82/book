# Compact simulation extracts used by the independent ordinal chapter.
# All probabilities below are recomputed from paired draws at render time.
read_ordinal_example <- function(directory = "data/longbet-ordinal") {
  checks <- read.csv(file.path(directory, "checksums.csv"))
  for (j in seq_len(nrow(checks))) {
    actual <- digest::digest(file = file.path(directory, checks$file[j]), algo = "sha256")
    stopifnot(identical(actual, checks$sha256[j]))
  }
  panels <- read.csv(file.path(directory, "panel-results.csv"))
  draws <- read.csv(gzfile(file.path(directory, "decision-draws.csv.gz")))
  provenance <- jsonlite::fromJSON(file.path(directory, "provenance.json"))
  stopifnot(nrow(panels) == 180L,
            identical(sort(unique(panels$seed)), 91000:91019),
            all(table(panels$scenario, panels$method)[table(panels$scenario, panels$method) > 0] == 20L),
            nrow(draws) == 20000L,
            identical(draws$chain, rep(1:4, each = 5000)),
            identical(draws$iteration, rep(1:5000, 4)),
            all(is.finite(draws$top_att)), all(is.finite(draws$lowest_att)),
            all(abs(draws$top_att) <= 1), all(abs(draws$lowest_att) <= 1))
  rare <- subset(panels, scenario == "rare_top" & grepl("longbet", method))
  stopifnot(nrow(rare) == 40L, all(rare$diagnostic_failures == 0))
  list(panels = panels, draws = draws, provenance = provenance)
}
