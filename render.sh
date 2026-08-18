#!/usr/bin/env bash
set -e

IMAGE_NAME="book"

if [ "$1" = "--build" ]; then
    echo "Building Docker image ${IMAGE_NAME}..."
    docker build -t "${IMAGE_NAME}" .
    exit 0
fi

if [ -n "$1" ]; then
    echo "Rendering target '$1' locally via Docker..."
    docker run --rm -v "$(pwd):/book" "${IMAGE_NAME}" quarto render "$1"
else
    echo "Rendering full book locally via Docker..."
    docker run --rm -v "$(pwd):/book" "${IMAGE_NAME}" quarto render
fi

echo "Done! Render output saved in docs/ and _freeze/."
