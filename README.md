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

## Usage

### Quick start

First, clone this github repo and install dependencies using the following command: 

```bash
pip install -e .
```

Afetr that, you can run the code below for inference. 

```python
import torch
from PIL import Image

from colpali_engine.models import ColFlor, ColFlorProcessor

model_name = "ahmed-masry/ColFlor"

model = ColFlor.from_pretrained(
    model_name,
    device_map="cuda", 
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

### Benchmarking

To reproduce the results reported in the blogpost, you can simple run the following colab notebook on free T4 gpu! 
[[Colab Evaluation Notebook]](https://colab.research.google.com/drive/1fvLP5WLKssg9yEtkwVdG5yxMBGhrcjGZ?usp=sharing )

The notebook mainly utilizes the evaluation codes from this github repo: [[Vidore Benchmark colflor]](https://github.com/AhmedMasryKU/vidore-benchmark-colflor)

### Training

First, clone this repo and run the following command to install dependencies: 

```bash
pip install . -e 
```
Then, you can start the training process by running this command: 

```bash
python scripts/train/train_colbert.py scripts/configs/pali/train_colpali_docmatix_hardneg_model.yaml
```
Make sure to modify the yaml file based on your dataset and training setup!

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
