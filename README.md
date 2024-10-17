# ColFlor: Towards BERT-Size Vision-Language Document Retrieval Models
---

[[Model card]](https://huggingface.co/ahmed-masry/ColFlor)
[[Demo]](https://huggingface.co/spaces/ahmed-masry/ColFlor-Demo)
[[Blog Post]](https://huggingface.co/blog/ahmed-masry/colflor)

## Credits. 
This read me file (and github repo in general) was adapted from the ColPali original github repo (https://github.com/illuin-tech/colpali)

## Associated BlogPost

This repository contains the code used for training the ColFlor model described in the [*ColFlor: Towards BERT-Size Vision-Language Document Retrieval Models*](https://huggingface.co/blog/ahmed-masry/colflor) blogpost.

## Introduction

With our new model *ColPali*, we propose to leverage VLMs to construct efficient multi-vector embeddings in the visual space for document retrieval. By feeding the ViT output patches from PaliGemma-3B to a linear projection, we create a multi-vector representation of documents. We train the model to maximize the similarity between these document embeddings and the query embeddings, following the ColBERT method.

Using ColPali removes the need for potentially complex and brittle layout recognition and OCR pipelines with a single model that can take into account both the textual and visual content (layout, charts, ...) of a document.

![ColPali Architecture](assets/colpali_architecture.webp)

## List of ColVision models

| Model                                                        | Score on [ViDoRe](https://huggingface.co/spaces/vidore/vidore-leaderboard) 🏆 | License    | Comments                                                     | Currently supported |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---------- | ------------------------------------------------------------ | ------------------- |
| [vidore/colpali](https://huggingface.co/vidore/colpali)      | 81.3                                                         | Gemma      | • Based on `google/paligemma-3b-mix-448`.<br />• Checkpoint used in the ColPali paper. | ❌                   |
| [vidore/colpali-v1.1](https://huggingface.co/vidore/colpali-v1.1) | 81.5                                                         | Gemma      | • Based on `google/paligemma-3b-mix-448`.                    | ✅                   |
| [vidore/colpali-v1.2](https://huggingface.co/vidore/colpali-v1.2) | 83.1                                                         | Gemma      | • Based on `google/paligemma-3b-mix-448`.                    | ✅                   |
| [vidore/colqwen2-v0.1](https://huggingface.co/vidore/colqwen2-v0.1) | 86.6                                                         | Apache 2.0 | • Based on `Qwen/Qwen2-VL-2B-Instruct`.<br />• Supports dynamic resolution.<br />• Trained using 768 image patches per page. | ✅                   |

## Setup

We used Python 3.11.6 and PyTorch 2.2.2 to train and test our models, but the codebase is compatible with Python >=3.9 and recent PyTorch versions. To install the package, run:

```bash
pip install colpali-engine
```

> [!WARNING]
> For ColPali versions above v1.0, make sure to install the `colpali-engine` package from source or with a version above v0.2.0.

## Usage

### Quick start

```python
import torch
from PIL import Image

from colpali_engine.models import ColPali, ColPaliProcessor

model_name = "vidore/colpali-v1.2"

model = ColPali.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  # or "mps" if on Apple Silicon
).eval()

processor = ColPaliProcessor.from_pretrained(model_name)

# Your inputs
images = [
    Image.new("RGB", (32, 32), color="white"),
    Image.new("RGB", (16, 16), color="black"),
]
queries = [
    "Is attention really all you need?",
    "Are Benjamin, Antoine, Merve, and Jo best friends?",
]

# Process the inputs
batch_images = processor.process_images(images).to(model.device)
batch_queries = processor.process_queries(queries).to(model.device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**batch_images)
    query_embeddings = model(**batch_queries)

scores = processor.score_multi_vector(query_embeddings, image_embeddings)

```

### Inference

You can find an example [here](https://github.com/illuin-tech/colpali/blob/main/scripts/infer/run_inference_with_python.py). If you need an indexing system, we recommend using [`byaldi`](https://github.com/AnswerDotAI/byaldi) - [RAGatouille](https://github.com/AnswerDotAI/RAGatouille)'s little sister 🐭 - which share a similar API and leverages our `colpali-engine` package.

### Benchmarking

To benchmark ColPali to reproduce the results on the [ViDoRe leaderboard](https://huggingface.co/spaces/vidore/vidore-leaderboard), it is recommended to use the [`vidore-benchmark`](https://github.com/illuin-tech/vidore-benchmark) package.

### Training

To keep a lightweight repository, only the essential packages were installed. In particular, you must specify the dependencies to use the training script for ColPali. You can do this using the following command:

```bash
pip install "colpali-engine[train]"
```

All the model configs used can be found in `scripts/configs/` and rely on the [configue](https://github.com/illuin-tech/configue) package for straightforward configuration. They should be used with the `train_colbert.py` script.

#### Example 1: Local training

```bash
USE_LOCAL_DATASET=0 python scripts/train/train_colbert.py scripts/configs/pali/train_colpali_docmatix_hardneg_model.yaml
```

or using `accelerate`:

```bash
accelerate launch scripts/train/train_colbert.py scripts/configs/pali/train_colpali_docmatix_hardneg_model.yaml
```

#### Example 2: Training on a SLURM cluster

```bash
sbatch --nodes=1 --cpus-per-task=16 --mem-per-cpu=32GB --time=20:00:00 --gres=gpu:1  -p gpua100 --job-name=colidefics --output=colidefics.out --error=colidefics.err --wrap="accelerate launch scripts/train/train_colbert.py scripts/configs/pali/train_colpali_docmatix_hardneg_model.yaml"

sbatch --nodes=1  --time=5:00:00 -A cad15443 --gres=gpu:8  --constraint=MI250 --job-name=colpali --wrap="python scripts/train/train_colbert.py scripts/configs/pali/train_colpali_docmatix_hardneg_model.yaml"
```

## Citation
If you plan to use ColFlor in your research, please consider citing us as follows:
```latex
@misc{masry2024colflor,
    title={ColFlor: BERT-Size Vision-Language Document Retrieval Models},
    url={https://huggingface.co/blog/ahmed-masry/colflor},
    author={Masry, Ahmed},
    month={October},
    year={2024}
}
```
