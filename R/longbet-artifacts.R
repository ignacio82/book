# R/longbet-artifacts.R
# Load-or-build cache for fitted models. Chapters that share a panel share its fit: the first
# chapter to need it pays for it, the rest read it back. Every artifact is keyed on the engine
# source, the sampler settings and a digest of the data, so a fit from another revision, another
# budget or another data draw is never reused. Nothing is harvested from a knitr cache.

longbet_cache_dir <- function() {
  dir_path <- file.path("cache", if (longbet_quick()) "longbet-quick" else "longbet")
  if (!dir.exists(dir_path)) dir.create(dir_path, recursive = TRUE, showWarnings = FALSE)
  dir_path
}

lb_key <- function(...) {
  engine <- tryCatch(get_longbet_engine_fingerprint(), error = function(e) "unknown")
  digest::digest(list(schema = 2L, engine = engine, parts = list(...)), algo = "sha256")
}

write_artifact_atomic <- function(artifact, target_path) {
  tmp_file <- tempfile(pattern = paste0("tmp_", basename(target_path), "_"),
                       tmpdir = dirname(target_path))
  on.exit({ if (file.exists(tmp_file)) unlink(tmp_file) }, add = TRUE)
  saveRDS(artifact, tmp_file)
  if (is.null(tryCatch(readRDS(tmp_file), error = function(e) NULL))) {
    stop(sprintf("Failed to validate serialized artifact at '%s'", tmp_file))
  }
  file.rename(tmp_file, target_path)
  invisible(target_path)
}

# LONGBET_BUILD_ONLY=name1,name2 turns a render into a builder for just those artifacts: they are
# fitted as usual, cached artifacts are still reused, and an artifact that is neither listed nor
# cached comes back as NULL instead of being built. Code that uses such a placeholder fails, so
# this mode is meant for a throwaway copy rendered with `error: true`, where a failed chunk does
# not stop the ones after it: tools/prebuild-longbet-cache.sh uses it to spread the fits of one
# chapter over several containers or machines. A normal render never sets it.
lb_build_only <- function() {
  v <- Sys.getenv("LONGBET_BUILD_ONLY")
  if (!nzchar(v)) NULL else trimws(strsplit(v, ",")[[1]])
}

# Build `name` unless a cached copy carries the same key. `builder` runs in a fresh worker
# process by default, so the memory a fit and its prediction allocate is returned to the
# operating system before the chapter continues; `export` names the objects it needs.
lb_artifact <- function(name, key, builder, export = character(), in_worker = TRUE) {
  target_file <- file.path(longbet_cache_dir(), paste0(name, ".rds"))
  if (file.exists(target_file)) {
    cached <- tryCatch(readRDS(target_file), error = function(e) NULL)
    if (!is.null(cached) && identical(cached$key, key)) {
      message(sprintf("[Artifact] Reusing %s from %s", name, target_file))
      return(cached$payload)
    }
    message(sprintf("[Artifact] %s is stale; rebuilding", name))
  }
  only <- lb_build_only()
  if (!is.null(only) && !(name %in% only)) {
    message(sprintf("[Artifact] %s is not in LONGBET_BUILD_ONLY and not cached; returning NULL", name))
    return(NULL)
  }
  message(sprintf("[Artifact] Building %s ...", name))
  started <- Sys.time()
  payload <- if (in_worker) run_parallel(list(name), function(task) builder(), export = export)[[1]]
             else builder()
  envelope <- list(schema_version = 2L, artifact = name, key = key,
                   seconds = as.numeric(difftime(Sys.time(), started, units = "secs")),
                   created_at = Sys.time(), payload = payload)
  write_artifact_atomic(envelope, target_file)
  message(sprintf("[Artifact] Saved %s (%.0f s)", name, envelope$seconds))
  payload
}

# The marketplace rollout's fit, shared by the first three chapters. The fit object carries the
# serialized engine state, so predict() rehydrates it in whichever chapter reads it back.
lb_core_fit <- function(ro, budget = lb_budget(16)) {
  key <- lb_key("core_fit", budget, LB_MODEL,
                digest::digest(list(ro$y_train, ro$x, ro$z_train, ro$week_study), algo = "sha256"))
  lb_artifact("core_fit", key, builder = function() {
    do.call(longbet::longbet, c(list(y = ro$y_train, x = ro$x, z = ro$z_train, t = ro$week_study,
                                     random_seed = 42L), LB_MODEL, budget))
  })
}
