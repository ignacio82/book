#!/usr/bin/env bash
# Render the book (or one chapter) inside the `book` Docker image.
#
#   ./render.sh                  render the whole book
#   ./render.sh longbet.qmd      render one chapter
#   ./render.sh --build          build the image
#
# LONGBET_REPO=/path/to/longbet-jax renders against that working tree instead of
# the longbet-jax revision baked into the image: the tree's Python engine is put
# on PYTHONPATH and its R front door is installed into a temporary library for
# the duration of the render. Use it to render the LongBet chapters against an
# engine revision that is not yet published; the Dockerfile's LONGBET_REF is
# what CI and a plain `./render.sh` use.
set -euo pipefail

IMAGE_NAME="${BOOK_IMAGE:-book}"

if [ "${1:-}" = "--build" ]; then
    echo "Building Docker image ${IMAGE_NAME}..."
    docker build -t "${IMAGE_NAME}" .
    exit 0
fi

engine_args=()
render_cmd='quarto render "$@"'
if [ -n "${LONGBET_REPO:-}" ]; then
    engine_repo=$(cd -- "${LONGBET_REPO}" && pwd)
    engine_commit=$(git -C "${engine_repo}" rev-parse --short HEAD 2>/dev/null || echo "unknown")
    engine_hash=$(cd "${engine_repo}" && tar --exclude='__pycache__' --exclude='*.pyc' -cf - \
        R src inst man DESCRIPTION NAMESPACE | sha256sum | cut -d ' ' -f 1)
    echo "Using longbet-jax working tree ${engine_repo} (commit ${engine_commit}, source ${engine_hash:0:12})"
    engine_args=(-v "${engine_repo}:/engine:ro"
                 -e PYTHONPATH=/engine/src -e PYTHONDONTWRITEBYTECODE=1
                 -e LONGBET_ENGINE_COMMIT="${engine_commit}" -e LONGBET_SOURCE_SHA256="${engine_hash}"
                 -e R_LIBS=/tmp/longbet-r-library)
    render_cmd='mkdir -p /tmp/longbet-r-library && R CMD INSTALL --no-libs --library=/tmp/longbet-r-library /engine >/dev/null && quarto render "$@"'
fi

if [ -n "${1:-}" ]; then
    echo "Rendering target '$1' locally via Docker..."
else
    echo "Rendering full book locally via Docker..."
fi
docker run --rm -v "$(pwd):/book" -e LONGBET_QUICK="${LONGBET_QUICK:-}" "${engine_args[@]}" "${IMAGE_NAME}" bash -c "${render_cmd}" render "$@"

echo "Done! Render output saved in docs/ and _freeze/."
