# PRISM: Phase-enhanced Radial-based Image Signature Mapping for AI-Generated Image Attribution

This repository contains the **image generation pipeline** for the PRISM framework, a method for AI-generated image attribution and fingerprinting.

**Citation:**
```
@article{ricco2025prism,
  author  = {Ricco, Emanuele and Onofri, Elia and Cima, Lorenzo
             and Cresci, Stefano and {Di Pietro}, Roberto},
  title   = {{PRISM}: Phase-enhanced Radial-based Image Signature Mapping
             for {AI}-Generated Image Attribution},
  journal = {arXiv preprint arXiv:2509.15270},
  year    = {2025},
  doi     = {10.48550/arXiv.2509.15270},
  url     = {https://arxiv.org/abs/2509.15270}
}

```

## Repository Structure

This repository is organized as follows:

```
PRISM-36K/
├── metadata/                           
│   ├── prompts.csv             # prompt_id, length, pair_id, prompt text
│   ├── images.csv              # filename, model, prompt_id, iter, sha256, width, height (after image generation) 
│── src/ 
│   ├── generateMetadata.py     # file to generate files in the folder metadata/
│   ├── generation.py           # Main image generation script (4 API models)
├── .env-template               # API keys (OPENAI_API_KEY, HUGGINGFACE_TOKEN, REPLICATE_API_TOKEN)
│── changelog.md                # version history
│── LICENSE.txt                 # CC-BY-4.0 license
├── README.md                   # this file
├── requirements.txt            # Python dependencies for API-based models
├── FuseDream/                  # FuseDream repository (cloned, separate environment)
├── VQGAN-CLIP/                 # VQGAN-CLIP repository (cloned, separate environment)
└── images/                     # Output directory (created automatically after generation.py)
    ├── dalle2/                 # DALL·E 2 images
    ├── nvidia-sana/            # NVIDIA Sana images
    ├── pixart/                 # PixArt images
    ├── stable-diffusion/       # Stable Diffusion images
    ├── fusedream/              # FuseDream images (generated from FuseDream)
    └── vqgan_clip/             # VQGAN-CLIP images (generated from VQGAN-CLIP)

```

## Supported Image Generation Models

The `src/generation.py` script supports 4 different image generation models:

- **NVIDIA Sana**: Efficient high-resolution image synthesis with linear diffusion transformer
- **PixArt-α**: Fast training diffusion transformer for photorealistic text-to-image synthesis
- **DALL·E 2** : Hierarchical text-conditional image generation (requires API key)
- **Stable Diffusion v1.4**: Pre-trained model for diffusion-based synthesis

### Output Directory Structure

The scripts generate an `images/` folder with subdirectories for each model. Images are named using the convention: `{model}_{i}_{j}.png` where `i` is the prompt index (1-40) and `j` is the response/image index (1-N) with N = 150 as default parameter.

```
images/
├── dalle2/
│   ├── dalle2_1_1.png
│   ├── dalle2_1_2.png
│   └── dalle2_1_N.png
├── nvidia-sana/
│   ├── nvidia-sana_1_1.png
│   ├── nvidia-sana_1_2.png
│   └── nvidia-sana_1_N.png
├── pixart/
│   ├── pixart_1_1.png
│   ├── pixart_1_2.png
│   └── pixart_1_N.png
├── stable-diffusion/
│   ├── stable-diffusion_1_1.png
│   ├── stable-diffusion_1_2.png
│   └── stable-diffusion_1_N.png
├── fusedream/
│   ├── fusedream_1_1.png
│   ├── fusedream_1_2.png
│   └── fusedream_1_N.png
└── vqgan_clip/
    ├── vqgan_clip_1_1.png
    ├── vqgan_clip_1_2.png
    └── vqgan_clip_1_N.png
```

**Naming Convention:**
- `{model}_{i}_{j}.png`
  - `{model}`: Model identifier (dalle2, nvidia-sana, pixart, stable-diffusion, fusedream, vqgan_clip)
  - `{i}`: Prompt index (1-40)
  - `{j}`: Response/image index (1-N)

**Example:** For 40 prompts with N=150 responses each:
- Total images per model: 6,000 (40 prompts × 150 responses)
- Total images across all 6 models: 36,000 images



## Quick Start

### API-Based Models (Sana, PixArt, DALL·E 2, Stable Diffusion)

The `requirements.txt` file in this directory contains dependencies for running the API-based models. These models share the same Python environment, recommended Python 3.11.

conda create -n prism python=3.11

Remember also to add the different tokens in .env (OPENAI_API_KEY, HUGGINGFACE_TOKEN, REPLICATE_API_TOKEN).

#### 1. Install Dependencies

```bash
conda activate prism
python -m pip install -r requirements.txt
```

#### 2. Generate Images

Run the main generation script:

```bash
python src/generation.py --responses N
```

**Parameters:**
- `--prompts`: Path to prompts file (CSV or text file). Default: `../metadata/prompts.csv`
- `--responses`: Number of image variants to generate per prompt (default: N)

---

### FuseDream (Separate Environment)

FuseDream requires its own Python virtual environment due to specific dependencies.
The original `fusedream_generator.py` needs to be modified to process multiple prompts in batch.

#### 1. Clone and Setup

```bash
git clone https://github.com/gnobitab/FuseDream
cd FuseDream

# Create a separate virtual environment for FuseDream (recommended python 3.9)
conda create -n fusedream python=3.9  
conda activate fusedream
```

#### 2. Install Dependencies

Refer to the FuseDream repository's `README.md` for its specific requirements.

#### 3. Download Pretrained Weights

Install gdown:
```bash
pip install gdown
```

Download the BigGAN checkpoints:
```bash
gdown "https://drive.google.com/uc?id=1sOZ9og9kJLsqMNhaDnPJgzVsBZQ1sjZ5" -O BigGAN_utils/weights/biggan-512.pth
```

#### 4. Modify `fusedream_generator.py` for Batch Processing

The original `fusedream_generator.py` processes a single prompt per run. Modify it to read from `../metadata/prompts.csv` and generate batch images:

**Replace the argument parsing and single-prompt logic with:**

```python
import torch
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torchvision
import BigGAN_utils.utils as utils
import clip
import torch.nn.functional as F
from DiffAugment_pytorch import DiffAugment
import numpy as np
from fusedream_utils import FuseDreamBaseGenerator, get_G, save_image
import os
import csv

# Read prompts from the CSV file
prompts_file = '../metadata/prompts.csv'
prompts = []
with open(prompts_file, 'r') as f:
    reader = csv.DictReader(f)
    prompts = [row['prompt'].strip() for row in reader if row['prompt'].strip()]

# Configuration
NUM_REPETITIONS = N
INIT_ITERS = 50
OPT_ITERS = 50
output_dir = '../images/fusedream'

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Initialize the model once
G, config = get_G(512)  # Choose from 256 and 512
generator = FuseDreamBaseGenerator(G, config, 10)

# Process each prompt
for i, sentence in enumerate(prompts):
    prompt_index = i + 1
    print(f"\nProcessing prompt {prompt_index}/{len(prompts)}: {sentence}")
    
    for j in range(1, NUM_REPETITIONS + 1):
        # Set a different seed for each repetition
        seed = prompt_index * 1000 + j
        utils.seed_rng(seed)
        
        # Generate basis
        z_cllt, y_cllt = generator.generate_basis(sentence, init_iters=INIT_ITERS, num_basis=5)
        z_cllt_save = torch.cat(z_cllt).cpu().numpy()
        y_cllt_save = torch.cat(y_cllt).cpu().numpy()
        
        # Optimize CLIP score
        img, z, y = generator.optimize_clip_score(z_cllt, y_cllt, sentence, latent_noise=True, augment=True, opt_iters=OPT_ITERS, optimize_y=True)
        score = generator.measureAugCLIP(z, y, sentence, augment=True, num_samples=10)
        
        # Save the image with consistent naming
        output_filename = f"fusedream_{prompt_index}_{j}.png"
        output_path = os.path.join(output_dir, output_filename)
        save_image(img, output_path)
        
        if j % 10 == 0:
            print(f"  Generated {j}/{NUM_REPETITIONS} images (CLIP score: {score:.4f})")

print("\nAll FuseDream images generated successfully!")
```

#### 5. Running FuseDream

From the FuseDream directory (with your venv activated):

```bash
python fusedream_generator.py
```

This will:
- Read all prompts from `../metadata/prompts.csv`
- Generate N images per prompt
- Save to `../images/fusedream/` with naming: `fusedream_i_j.png`, etc.
- Display progress and CLIP scores for each batch

---

### VQGAN-CLIP (Separate Environment)

VQGAN-CLIP also requires its own Python virtual environment (Python 3.9).

#### 1. Clone and Setup

```bash
git clone https://github.com/nerdyrodent/VQGAN-CLIP
cd VQGAN-CLIP

# Create a separate virtual environment for VQGAN-CLIP
conda create -n vqgan python=3.9
conda activate vqgan
```

#### 2. Install Dependencies

First, install PyTorch with CUDA 11.1 support:

```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Then install other required packages (refer to VQGAN-CLIP's `requirements.txt` or use):

```bash
pip install ftfy regex tqdm omegaconf pytorch-lightning IPython kornia imageio imageio-ffmpeg einops torch_optimizer setuptools==59.5.0
```

#### 3. Clone Required Repositories

Inside the VQGAN-CLIP directory, clone CLIP and taming-transformers:

```bash
git clone https://github.com/openai/CLIP
git clone https://github.com/CompVis/taming-transformers
```

#### 4. Download Pretrained Models

Create a checkpoints directory and download VQGAN models:

```bash
mkdir checkpoints

# Example: Download ImageNet 16384 model
curl -L -o checkpoints/vqgan_imagenet_f16_16384.yaml -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1'
curl -L -o checkpoints/vqgan_imagenet_f16_16384.ckpt -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1'
```

For additional models, see the [taming-transformers](https://github.com/CompVis/taming-transformers#overview-of-pretrained-models) repository.

#### 5. Generate Images from Prompts

Create a bash script in the VQGAN-CLIP directory to generate images from prompts stored in `../metadata/prompts.csv`:

```bash
#!/bin/bash

# This script should be placed in the VQGAN-CLIP directory
# It reads prompts from ../metadata/prompts.csv and generates images

# Set generation parameters
SIZE="512 512"           # Output image size (512x512)
ITERATIONS=250           # Number of iterations per image
SAVE_EVERY=50            # Save intermediate results every N iterations
NUM_RESPONSES=N          # Number of responses
SEED=42                  # Random seed


# Create output directory
OUTPUT_DIR="../images/vqgan_clip"
mkdir -p "$OUTPUT_DIR"

# Read prompts from CSV file (skip header)
PROMPTS_FILE="../metadata/prompts.csv"

# Counter for prompt index
prompt_index=0

# Read CSV file, skip header, and extract prompt column (last column)
tail -n +2 "$PROMPTS_FILE" | while IFS=',' read -r prompt_id length pair_id prompt; do
    # Remove quotes if present and trim whitespace
    prompt=$(echo "$prompt" | sed 's/^"//;s/"$//' | xargs)
    
    prompt_index=$((prompt_index + 1))
    
    echo "Processing prompt $prompt_index: \"$prompt\""
    
    # Generate N images for this prompt
    for j in $(seq 1 $NUM_RESPONSES); do
        # Create unique seed for each image      
        echo "  Generating image $j/$NUM_RESPONSES prompt $prompt_index"
        
        # Run VQGAN-CLIP generation
        python generate.py \
            -p "$prompt" \
            -s $SIZE \
            -i $ITERATIONS \
            -se $SAVE_EVERY \
            -o "$OUTPUT_DIR/vqgan_clip_${prompt_index}_${j}.png" \
            -sd $SEED
    done
    
    echo "Completed prompt $prompt_index"
done

echo "VQGAN-CLIP image generation complete!"
echo "Generated images saved to: $OUTPUT_DIR"
```

Save this script (e.g., as `generate_vqgan.sh`) in the VQGAN-CLIP directory and run it:

```bash
cd VQGAN-CLIP
bash generate_vqgan.sh
```

**Parameters explained:**
- `-p`: Prompt text
- `-s`: Image size (width height, e.g., "512 512")
- `-i`: Number of iterations (higher = more refined, but slower)
- `-se`: Save interval (save intermediate images every N iterations)
- `-o`: Output file path
- `-sd`: Random seed for reproducibility

#### 6. Hardware Requirements

VQGAN-CLIP requires significant GPU memory:
- **380x380 images**: ~8 GB VRAM
- **512x512 images**: ~10 GB VRAM
- **900x900 images**: ~24 GB VRAM

For A100 GPUs, the provided SLURM parameters are:
```
--gpus=1
--mem=NG
--cpus-per-task=32
--constraint=a100
```
This will:
- Read all prompts from `../metadata/prompts.csv`
- Generate N images per prompt
- Save to `../images/vqgan_clip/` with naming: `vqgan_clip_i_j.png`, etc.
- Display progress and CLIP scores for each batch

---

## Generate Metadata

After generating all images from the above models, run the metadata generation script to create `metadata/images.csv`:

```bash
python src/generateMetadata.py
```

This script will:
- Scan the `images/` directory for all generated images
- Extract image properties (filename, model, dimensions, hash)
- Create `metadata/images.csv` with columns: `filename, model, prompt_id, iter, sha256, width, height`

This CSV file is required for subsequent analysis and attribution tasks.

---

### References

VQGAN - CLIP:
```
@inproceedings{esser2021taming,
  title={Taming transformers for high-resolution image synthesis},
  author={Esser, Patrick and Rombach, Robin and Ommer, Bjorn},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={12873--12883},
  year={2021},
  note={\doi{10.1109/CVPR46437.2021.01268}}
}
```

FuseDream:
```
@article{liu2021fusedream,
  title={Fusedream: Training-free text-to-image generation with improved clip+ gan space optimization},
  author={Liu, Xingchao and Gong, Chengyue and Wu, Lemeng and Zhang, Shujian and Su, Hao and Liu, Qiang},
  journal={arXiv preprint arXiv:2112.01573},
  year={2021},
  note={\doi{10.48550/arXiv.2112.01573}}
}
```

DALL·E 2:
```
@misc{ramesh2022dalle2,
  author = {Ramesh, Aditya and Dhariwal, Prafulla and Nichol, Alex and Chu, Casey and Chen, Mark},
  title = {DALL·E 2: Hierarchical Text-Conditional Image Generation with CLIP Latents},
  year = {2022},
  publisher = {OpenAI},
  howpublished = {\url{https://platform.openai.com/docs/models/dall-e-2}},
  note = {Accessed: 2025-05-18}
}
```

Sana:
```
@misc{xie2024sana,
  title={Sana: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformer},
  author={Enze Xie and Junsong Chen and Junyu Chen and Han Cai and Haotian Tang and Yujun Lin and Zhekai Zhang and Muyang Li and Ligeng Zhu and Yao Lu and Song Han},
  year={2024},
  eprint={2410.10629},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2410.10629}
}
```

PixArt-α:
```
@inproceedings{chen2024pixart,
  title={PixArt-$\alpha$: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis},
  author={Chen, Junsong and Yu, Jincheng and Ge, Chongjian and Yao, Lewei and Xie, Enze and Wang, Zhongdao and Kwok, James T and Luo, Ping and Lu, Huchuan and Li, Zhenguo},
  booktitle={ICLR},
  year={2024},
  note={\doi{10.48550/arXiv.2310.00426}}
}
```

Stable Diffusion v1.4:
- Pre-trained model available at [Hugging Face](https://huggingface.co/CompVis/stable-diffusion-v1-4)


## Companion Dataset

The PRISM-36K dataset is publicly available on Zenodo:

[![Zenodo](https://img.shields.io/badge/Zenodo-20065919-1b60c2?logo=zenodo)](https://zenodo.org/records/20065919)

The dataset contains 36,000 AI-generated images (6,000 images per model) along with corresponding metadata for fingerprinting and attribution research.

---

## Contact / Issues

For bug reports, questions about the generation, or collaboration enquiries, please contact:

**Emanuele Ricco** — `emanuele[dot]ricco[at]kaust[dot]edu[dot]sa`  
Cybersecurity Research and Innovation Laboratory (CRI-Lab)  
King Abdullah University of Science and Technology (KAUST), Saudi Arabia