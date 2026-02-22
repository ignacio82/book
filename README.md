# Business Data Science: A guide for data-driven decisions

[![Read the Book](https://img.shields.io/badge/Read-Book-blue?style=for-the-badge&logo=Read-the-Docs)](https://book.martinez.fyi)
[![Chat with Iggy](https://img.shields.io/badge/Chat-with_Iggy-purple?style=for-the-badge&logo=OpenAI)](https://iggy.martinez.fyi)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg?style=for-the-badge)](https://creativecommons.org/licenses/by-nc/4.0/)

> By **Ignacio Martinez**

This repository contains the source code for the book **[Business Data Science: A guide for data-driven decisions](https://book.martinez.fyi)**. The book provides a comprehensive guide to the principles and applications of business data science, with a focus on making sound, data-driven decisions through causal inference.

## 📖 About the Book

In the modern business landscape, data isn't just an asset – it's the raw material from which informed decisions are forged. Data, however, does not speak for itself. The extraction of actionable insights requires not only technical prowess, but a sophisticated understanding of causal inference. This is where the business data scientist steps in, acting as the voice of data, translating its complex signals into meaningful narratives that drive strategic decision-making.

Throughout this book, we navigate the core principles of causal inference, learning how to confidently identify cause-and-effect relationships within data. Our exploration emphasizes a "decisions first" philosophy, ensuring data analysis is always laser-focused on informing and optimizing decision-making. Topics include:

- The Potential Outcomes Framework & Causal Inference
- Randomized Controlled Trials (A/B testing, factorial designs)
- Observational methods (Matching, Causal Impact, Synthetic Control)
- Generalized Linear Models & Bayesian thinking
- Stochastic Trees (BART, BCF)

## 🤖 Meet Iggy: Your AI Data Science Companion

To further assist you, I have created **[Iggy](https://iggy.martinez.fyi)**, an AI Data Science agent that acts as a companion to this book. Iggy is designed to answer your data science questions using the contents of this book as its knowledge base, providing an interactive way to explore and clarify the concepts discussed. 

## 🛠️ Building the Book

This book is written and built using [Quarto](https://quarto.org/). To render the book locally:

1. Install [Quarto](https://quarto.org/docs/get-started/)
2. Clone this repository
3. Render the book:
   ```bash
   quarto render
   ```

### Using Docker

To avoid installing R and all its dependencies locally, you can build and render the book using Docker. This ensures an isolated environment mirroring the CI process.

1. Build the Docker image (this will take a few minutes as it installs all packages):
   ```bash
   docker build -t business-data-science-book .
   ```
2. Run the container to render the book. To prevent generated files (like caches and compiled models) from cluttering your repository, we render everything into a dedicated `_docker_build` subfolder that will be ignored by Git:
   ```bash
   mkdir -p _docker_build
   docker run --rm -v "$(pwd):/src:ro" -v "$(pwd)/_docker_build:/book" business-data-science-book bash -O extglob -O dotglob -c "cp -a /src/!(_docker_build) /book/ && quarto render"
   ```
   Once finished, you can find the fully rendered book at `_docker_build/docs/index.html`.

## 📄 License

This book is licensed under the [Creative Commons Attribution-NonCommercial 4.0](https://creativecommons.org/licenses/by-nc/4.0/) License.
