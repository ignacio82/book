# Business Data Science: A guide for data-driven decisions

[![Read the Book](https://img.shields.io/badge/Read-Book-blue?style=for-the-badge&logo=Read-the-Docs)](https://book.martinez.fyi)
[![Chat with Iggy](https://img.shields.io/badge/Chat-with_Iggy-purple?style=for-the-badge&logo=OpenAI)](https://iggy.martinez.fyi)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg?style=for-the-badge)](https://creativecommons.org/licenses/by-nc/4.0/)

> By **Ignacio Martinez**

This repository contains the source code for the book **[Business Data Science: A guide for data-driven decisions](https://book.martinez.fyi)**. The book provides a comprehensive guide to the principles and applications of business data science, with a focus on making sound, data-driven decisions through causal inference.

## 📖 About the Book

In the modern business landscape, data isn't just an asset – it's the raw material from which informed decisions are forged. Data, however, does not speak for itself. The extraction of actionable insights requires not only technical prowess, but a sophisticated understanding of causal inference. This is where the business data scientist steps in, acting as the voice of data, translating its complex signals into meaningful narratives that drive strategic decision-making.

Throughout this book, we navigate the core principles of causal inference, learning how to confidently identify cause-and-effect relationships within data. Our exploration emphasizes a "decisions first" philosophy, ensuring data analysis is always laser-focused on informing and optimizing decision-making. Topics include:

- The Potential Outcomes Framework & Causal Inference Foundations
- Randomized Experiments and Observational Comparisons (A/B testing, factorial designs, instrumental variables, matching)
- Generalized Linear Models & Bayesian Thinking
- Stochastic Trees & Heterogeneous Effects (BART, BCF, causal propensity, PhoBART, BAD)
- Longitudinal and Panel Causal Inference (CausalImpact, Synthetic Control, LongBet dynamic effects, decisions, and extensions)

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

To avoid installing R and all its dependencies directly on your host machine, you can build and render the book using Docker:

1. Build the Docker image (installs system libraries, R packages, and dependencies):
   ```bash
   make build
   # or: ./render.sh --build
   ```
2. Render a single modified chapter (fast, with caching and freeze output):
   ```bash
   make render-chapter CHAPTER=phobart.qmd
   # or: ./render.sh phobart.qmd
   ```
3. Render the full book:
   ```bash
   make render
   # or: ./render.sh
   ```
4. Live interactive preview:
   ```bash
   make preview
   ```

*Note: When rendering locally, Quarto saves execution outputs to `_freeze/`. Committing `_freeze/` allows GitHub Actions to publish in 1–2 minutes without re-running long MCMC computations in CI. After modifying shared LongBet R files, explicitly render all three LongBet chapters (`./render.sh longbet.qmd`, `./render.sh longbet_decisions.qmd`, `./render.sh longbet_extensions.qmd`) to refresh their frozen outputs.*

## 📄 License

This book is licensed under the [Creative Commons Attribution-NonCommercial 4.0](https://creativecommons.org/licenses/by-nc/4.0/) License.
