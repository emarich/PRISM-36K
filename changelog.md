# Changelog

All notable changes to this dataset will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Dataset releases are versioned as `MAJOR.MINOR.PATCH`:

- **MAJOR** — incompatible structural changes (e.g., renamed folders, changed filename convention, removed images)
- **MINOR** — backwards-compatible additions (e.g., new metadata columns, additional splits, new generator subset)
- **PATCH** — corrections to metadata, checksums, or documentation with no change to image content

---

## [1.0.0] — 2026-05-07

### Added

- **src/generation.py** — Main image generation script supporting 4 API models (NVIDIA Sana, PixArt-α, DALL·E 2, and Stable Diffusion v1.4) with batch generation capabilities
- **src/generateMetadata.py** — Metadata generation utility for processing generated images and their associated attributes
- **metadata/prompts.csv** — Comprehensive dataset of text prompts used for image generation
- **README.md** — Documentation for the PRISM framework, repository structure, installation instructions, and usage guidelines
- **requirements.txt** — Python dependencies for the generation pipeline and API-based models
- **.env-template** — Template file for environment variables (API keys: OPENAI_API_KEY, HUGGINGFACE_TOKEN, REPLICATE_API_TOKEN)
- **LICENSE** — MIT License for the project
- **changelog.md** — This file, documenting version changes and updates
