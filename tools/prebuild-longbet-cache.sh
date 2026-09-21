#!/usr/bin/env bash
# Build the LongBet fit cache for several chapters, or single fits, at once, without touching
# the project's docs/ or _freeze/.
#
#   tools/prebuild-longbet-cache.sh longbet.qmd longbet_encouragement.qmd
#   tools/prebuild-longbet-cache.sh longbet.qmd:core_fit longbet_did.qmd:hetero_fit ...
#
# Each job renders a throwaway copy of the book (no _freeze, so Quarto executes it) with the
# real cache/ directory mounted in its place, so the fitted models land in cache/longbet/
# keyed exactly as the real render expects. Jobs run concurrently, one Docker container each.
# A `chapter.qmd:artifact` job sets LONGBET_BUILD_ONLY=artifact (see R/longbet-artifacts.R)
# and renders the copy with `error: true`, so it fits only that artifact and tolerates the
# chunks that need the ones it skipped; that is how one chapter's fits are spread over many
# cores or machines. Afterwards `./render.sh <chapter>.qmd` reuses every fit and only re-runs
# predictions and the econometric estimators.
#
# LONGBET_QUICK=1 uses the smoke-test budgets and the cache/longbet-quick/ store.
# SCRATCH sets where the copies live (default: a directory under /tmp).
set -euo pipefail
[ "$#" -ge 1 ] || { echo "usage: $0 <chapter.qmd[:artifact]> [more jobs...]" >&2; exit 2; }
book=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
scratch="${SCRATCH:-/tmp/longbet-prebuild}"
image="${BOOK_IMAGE:-book}"
mkdir -p "$scratch/logs" "$book/cache"
pids=(); names=()
for job in "$@"; do
  chapter="${job%%:*}"; artifact=""; [ "$job" != "$chapter" ] && artifact="${job#*:}"
  name="${chapter%.qmd}${artifact:+-$artifact}"
  copy="$scratch/$name"
  mkdir -p "$copy"
  rsync -a --delete --exclude docs --exclude .git --exclude cache --exclude _freeze \
        --exclude '*_files' --exclude '*_cache' --exclude .quarto "$book/" "$copy/"
  if [ -n "$artifact" ]; then
    # knitr tolerates a failed chunk under `error: true` but not a failed inline expression, and
    # inline text that summarises a skipped fit would fail; the copy's output is thrown away, so
    # drop every inline R expression from the copy.
    perl -0pi -e 's/`r [^`]*`//g' "$copy/$chapter"
    # The project already has an `execute:` block; edit it rather than adding a duplicate key.
    if grep -q '^  freeze: auto$' "$copy/_quarto.yml"; then
      sed -i 's/^  freeze: auto$/  freeze: false\n  error: true/' "$copy/_quarto.yml"
    else
      printf '\nexecute:\n  error: true\n  freeze: false\n' >> "$copy/_quarto.yml"
    fi
  fi
  log="$scratch/logs/$name.log"
  echo "[$(date +%H:%M:%S)] $job -> $log"
  docker run --rm -v "$copy:/book" -v "$book/cache:/book/cache" \
         -e LONGBET_QUICK="${LONGBET_QUICK:-}" -e LONGBET_BUILD_ONLY="$artifact" \
         "$image" quarto render "$chapter" > "$log" 2>&1 &
  pids+=("$!"); names+=("$job")
done
rc=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "[$(date +%H:%M:%S)] ${names[$i]} ok"
  else echo "[$(date +%H:%M:%S)] ${names[$i]} FAILED (see $scratch/logs)"; rc=1; fi
done
exit $rc
