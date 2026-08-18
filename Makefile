.PHONY: all build render render-chapter preview clean

IMAGE_NAME ?= book

all: render

build:
	docker build -t $(IMAGE_NAME) .

render:
	docker run --rm -v "$(CURDIR):/book" $(IMAGE_NAME) quarto render

render-chapter:
	@if [ -z "$(CHAPTER)" ]; then \
		echo "Usage: make render-chapter CHAPTER=phobart.qmd"; \
		exit 1; \
	fi
	docker run --rm -v "$(CURDIR):/book" $(IMAGE_NAME) quarto render $(CHAPTER)

preview:
	docker run --rm -it -p 4200:4200 -v "$(CURDIR):/book" $(IMAGE_NAME) quarto preview --port 4200 --host 0.0.0.0

clean:
	rm -rf _freeze cache docs
