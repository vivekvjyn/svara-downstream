# Svara Representation Learning for Carnatic Music Transcription

Self-supervised learning for svara representation, primarily aimed at Carnatic music transcription and related tasks such as performance analysis and imputation, using contrastive learning on unannotated pitch contours followed by supervised fine-tuning on annotated data.

## Setup

```bash
git clone https://github.com/vivekvjyn/svara-representation.git
cd svara-representation
pip install -r requirements.txt
pip install -e .
```

## Run experiments
1. Prepare dataset:

```bash
./scripts/pitch.sh
```

2. Pre-train the model on [Carnatic Music Rhythm (CMR)](https://zenodo.org/records/1264394) dataset:

```bash
./scripts/ssl.sh
```

> We pretrain an InceptionTime encoder using the InfoNCE loss on unannotated pitch contours from the CMR dataset. Positive pairs are created by applying data augmentations such as time warping and pitch drifting.

![alt](.github/images/simclr.png)

3. Fine-tune the pretrained model on annotated [Carnatic Varnam](https://doi.org/10.5281/zenodo.1257117) dataset using LoRA and report F1 score:

```bash
./scripts/svaras.sh
```

> We finetune the pretrained model on annotated data using cross-entropy loss for svara classification. Low-rank adaptation (LoRA) is used for efficient fine-tuning. we report F1 score for baseline and fine-tuned models.

![alt](.github/images/lora.png)

4. Cluster svara embeddings on the Carnatic Varnam dataset using HDBSCAN and report Normalized Mutual Information (NMI):

```bash
./scripts/gamakas.sh
```

> We evaluate the learned representations by clustering svara embeddings to identify distinct svara forms (gamaka realizations). HDBSCAN is applied independently for each svara, and the resulting clusters are compared against expert-provided svara-form annotations using the Normalized Mutual Information (NMI) score.
