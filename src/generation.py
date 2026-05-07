#!/usr/bin/env python3
"""Generate images from prompts using different models."""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional
import json
import csv

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()



class ImageGenerator:
    """Base class for image generators."""

    def __init__(self, model_name: str, output_base_dir: str = "images"):
        self.model_name = model_name
        self.output_base_dir = Path(output_base_dir)
        self.model_output_dir = self.output_base_dir / model_name
        self.model_output_dir.mkdir(parents=True, exist_ok=True)

    def save_image(self, image_data: bytes, prompt_idx: int, response_idx: int) -> str:
        """Save image with naming convention: model_i_j.png"""
        filename = f"{self.model_name}_{prompt_idx}_{response_idx}.png"
        filepath = self.model_output_dir / filename
        with open(filepath, "wb") as f:
            f.write(image_data)
        return str(filepath)

    def generate(self, prompts: list[str], num_responses: int) -> None:
        """Generate images for all prompts."""
        raise NotImplementedError


class NvidiaSanaGenerator(ImageGenerator):
    """Generate images using NVIDIA SANA model."""

    def __init__(self, output_base_dir: str = "images"):
        super().__init__("nvidia-sana", output_base_dir)
        try:
            import replicate
            self.replicate = replicate
        except ImportError:
            print("Error: replicate package not installed. Install with: pip install replicate")
            sys.exit(1)

    def generate(self, prompts: list[str], num_responses: int) -> None:
        """Generate images using NVIDIA SANA."""
        model_id = "nvidia/sana:c6b5d2b7459910fec94432e9e1203c3cdce92d6db20f714f1355747990b52fa6"

        for i, prompt in enumerate(prompts, start=1):
            print(f"[NVIDIA SANA] Processing prompt {i}/{len(prompts)}: {prompt}")

            for j in range(1, num_responses + 1):
                input_params = {
                    "width": 512,
                    "height": 512,
                    "prompt": prompt,
                    "model_variant": "1600M-512px",
                }

                try:
                    output = self.replicate.run(model_id, input=input_params)
                    image_path = self.save_image(output.read(), i, j)
                    print(f"  [{j}/{num_responses}] Saved: {image_path}")
                except Exception as e:
                    print(f"  [{j}/{num_responses}] Error: {e}")


class PixArtGenerator(ImageGenerator):
    """Generate images using PixArt model."""

    def __init__(self, output_base_dir: str = "images"):
        super().__init__("pixart", output_base_dir)
        try:
            import replicate
            self.replicate = replicate
        except ImportError:
            print("Error: replicate package not installed. Install with: pip install replicate")
            sys.exit(1)

    def generate(self, prompts: list[str], num_responses: int) -> None:
        """Generate images using PixArt."""
        model_id = "lucataco/pixart-xl-2:816c99673841b9448bc2539834c16d40e0315bbf92fef0317b57a226727409bb"

        for i, prompt in enumerate(prompts, start=1):
            print(f"[PixArt] Processing prompt {i}/{len(prompts)}: {prompt}")

            for j in range(1, num_responses + 1):
                input_params = {"width": 512, "height": 512, "prompt": prompt}

                try:
                    output = self.replicate.run(model_id, input=input_params)

                    for img_idx, item in enumerate(output):
                        image_path = self.save_image(item.read(), i, j)
                        print(f"  [{j}/{num_responses}] Saved: {image_path}")
                except Exception as e:
                    print(f"  [{j}/{num_responses}] Error: {e}")


class DALLE2Generator(ImageGenerator):
    """Generate images using DALL-E 2 model."""

    def __init__(self, output_base_dir: str = "images"):
        super().__init__("dalle2", output_base_dir)
        try:
            from openai import OpenAI
            import requests
            self.requests = requests
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("Error: OPENAI_API_KEY not found in .env file")
                sys.exit(1)
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            print("Error: openai package not installed. Install with: pip install openai")
            sys.exit(1)

    def generate(self, prompts: list[str], num_responses: int) -> None:
        """Generate images using DALL-E 2."""
        for i, prompt in enumerate(prompts, start=1):
            print(f"[DALL-E 2] Processing prompt {i}/{len(prompts)}: {prompt}")

            for j in range(1, num_responses + 1):
                try:
                    response = self.client.images.generate(
                        model="dall-e-2",
                        prompt=prompt,
                        n=1,
                        size="512x512",
                    )

                    image_url = response.data[0].url
                    image_response = self.requests.get(image_url)
                    image_path = self.save_image(image_response.content, i, j)
                    print(f"  [{j}/{num_responses}] Saved: {image_path}")
                except Exception as e:
                    print(f"  [{j}/{num_responses}] Error: {e}")


class StableDiffusionGenerator(ImageGenerator):
    """Generate images using Stable Diffusion model."""

    def __init__(self, output_base_dir: str = "images"):
        super().__init__("stable-diffusion", output_base_dir)
        try:
            import torch
            from diffusers import StableDiffusionPipeline
            from torch import autocast
            from huggingface_hub import login

            self.torch = torch
            self.autocast = autocast

            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            if not hf_token:
                print("Error: HUGGINGFACE_TOKEN not found in .env file")
                sys.exit(1)

            login(token=hf_token, add_to_git_credential=False)

            self.pipe = StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4"
            ).to("cuda")
        except ImportError:
            print(
                "Error: Required packages not installed. Install with: "
                "pip install torch diffusers transformers huggingface-hub"
            )
            sys.exit(1)
        except Exception as e:
            print(f"Error initializing Stable Diffusion: {e}")
            sys.exit(1)

    def generate(self, prompts: list[str], num_responses: int) -> None:
        """Generate images using Stable Diffusion."""
        for i, prompt in enumerate(prompts, start=1):
            print(f"[Stable Diffusion] Processing prompt {i}/{len(prompts)}: {prompt}")

            for j in range(1, num_responses + 1):
                try:
                    with self.autocast("cuda"):
                        result = self.pipe(prompt)

                    image = result.images[0]

                    # Convert PIL image to bytes
                    from io import BytesIO
                    image_bytes = BytesIO()
                    image.save(image_bytes, format="PNG")
                    image_data = image_bytes.getvalue()

                    image_path = self.save_image(image_data, i, j)
                    print(f"  [{j}/{num_responses}] Saved: {image_path}")
                except Exception as e:
                    print(f"  [{j}/{num_responses}] Error: {e}")


def load_prompts(prompt_file: str) -> list[str]:
    """Load prompts from file (CSV or text file)."""
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            if prompt_file.endswith(".csv"):
                reader = csv.DictReader(f)
                prompts = [row["prompt"].strip() for row in reader if row["prompt"].strip()]
            else:
                prompts = [line.strip() for line in f if line.strip()]

        if not prompts:
            print(f"Error: No prompts found in {prompt_file}")
            sys.exit(1)
        return prompts
    except FileNotFoundError:
        print(f"Error: Prompt file '{prompt_file}' not found")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Generate images from prompts using different models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generation.py --models nvidia-sana pixart --prompts ../metadata/prompts.csv --responses 10
  python generation.py  # Uses default settings with ../metadata/prompts.csv
        """,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["nvidia-sana"],
        choices=["nvidia-sana", "pixart", "dalle2", "stable-diffusion"],
        help="Models to use for generation (default: all)",
    )

    parser.add_argument(
        "--prompts",
        default="../metadata/prompts.csv",
        help="Path to prompts file (CSV or text file). CSV should have 'prompt' column (default: ../metadata/prompts.csv)",
    )

    parser.add_argument(
        "--responses",
        type=int,
        default=150,
        help="Number of images to generate per prompt (default: 150)",
    )

    parser.add_argument(
        "--output",
        default="images",
        help="Output directory for generated images (default: images)",
    )

    parser.add_argument(
        "--env",
        default=".env",
        help="Path to .env file containing API keys (default: .env)",
    )

    args = parser.parse_args()

    # Load .env file
    if args.env and os.path.exists(args.env):
        load_dotenv(args.env)
    elif args.env:
        print(f"Warning: .env file '{args.env}' not found. Using environment variables.")

    # Load prompts
    prompts = load_prompts(args.prompts)
    print(f"Loaded {len(prompts)} prompts from {args.prompts}\n")

    # Initialize generators
    generators: dict[str, ImageGenerator] = {
        "nvidia-sana": NvidiaSanaGenerator(args.output),
        "pixart": PixArtGenerator(args.output),
        "dalle2": DALLE2Generator(args.output),
        "stable-diffusion": StableDiffusionGenerator(args.output),
    }

    # Generate images for selected models
    for model_name in args.models:
        if model_name in generators:
            print(f"\n{'='*60}")
            print(f"Starting generation for: {model_name}")
            print(f"Prompts: {len(prompts)}, Responses per prompt: {args.responses}")
            print(f"{'='*60}\n")

            try:
                generators[model_name].generate(prompts, args.responses)
            except KeyboardInterrupt:
                print(f"\nInterrupted during {model_name} generation")
                sys.exit(1)
            except Exception as e:
                print(f"Error during {model_name} generation: {e}")

    print(f"\n{'='*60}")
    print("Image generation complete!")
    print(f"Images saved to: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()