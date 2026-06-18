# 🧠 Federated Multimodal Learning for Early MCI Detection

> **Research Project** — *Federated Multimodal Learning Framework for Early Mild Cognitive Impairment (MCI) Detection using Speech, Text, MRI and Electronic Health Records*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface)](https://huggingface.co/)
[![Flower](https://img.shields.io/badge/Federated-Flower%20FL-pink)](https://flower.dev/)
[![PEFT](https://img.shields.io/badge/LoRA-PEFT-green)](https://github.com/huggingface/peft)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Modules](#-modules)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Data Preparation](#-data-preparation)
- [Usage](#-usage)
  - [Centralised Training](#1-centralised-training)
  - [Federated Training](#2-federated-training)
  - [LoRA Fine-Tuning](#3-lora-fine-tuning)
  - [Inference / Prediction](#4-inference--prediction)
  - [Knowledge Graph + RAG](#5-knowledge-graph--rag)
- [Configuration](#-configuration)
- [Metrics](#-metrics)
- [Research Background](#-research-background)
- [Roadmap](#-roadmap)

---

## 🔍 Overview

This framework enables **privacy-preserving, multimodal detection** of Mild Cognitive Impairment (MCI) — an early precursor to Alzheimer's Disease — by combining four complementary data modalities:

| Modality | Model | Input | Output |
|---|---|---|---|
| 🎙️ Speech | `Wav2Vec2` | `.wav` audio files | Acoustic speech embeddings |
| 📝 Text | `BERT` | Transcript `.txt` files | Semantic language embeddings |
| 🧲 MRI | 3D CNN | `.nii` brain volumes | Structural neuroimaging embeddings |
| 📊 EHR | MLP | Tabular clinical data | Clinical feature embeddings |

All four embeddings are **concatenated and passed through a fusion classifier** to predict:
- `0` → **Healthy**
- `1` → **Mild Cognitive Impairment (MCI)**

Training is conducted using **Federated Learning** (via [Flower](https://flower.dev/)) so that raw patient data never leaves individual hospital sites. **LoRA (PEFT)** is applied to BERT and Wav2Vec2 to enable efficient fine-tuning of large pre-trained models. A **Knowledge Graph + RAG** module provides contextual medical knowledge augmentation via Neo4j and LangChain.

---

## 🏗️ Architecture

```
┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
│  .wav File │  │  .txt File │  │  .nii File │  │ EHR Table  │
└─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
      │               │               │               │
      ▼               ▼               ▼               ▼
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ Wav2Vec2 │   │   BERT   │   │  3D CNN  │   │   MLP    │
│ Encoder  │   │ Encoder  │   │ Encoder  │   │ Encoder  │
│ (+LoRA)  │   │ (+LoRA)  │   │          │   │          │
└────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘
     │              │              │              │
     └──────────────┴──────────────┴──────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Concat Fusion  │  ← all embeddings concatenated
                  │  (Linear + BN + │
                  │   Dropout)      │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   Classifier    │
                  │ Healthy │  MCI  │
                  └─────────────────┘


Federated Learning (Flower / FedAvg):

  ┌──────────────────────────────────────────┐
  │             Central Server               │
  │         FedAvg Aggregation               │
  └────────────┬───────────────┬─────────────┘
               │               │
        ┌──────▼──────┐ ┌──────▼──────┐
        │  Client 0   │ │  Client 1   │  ... (one per hospital site)
        │ (Hospital A)│ │ (Hospital B)│
        │ local train │ │ local train │
        └─────────────┘ └─────────────┘
          (data stays local — only weights are shared)
```

---

## 📁 Project Structure

```
project/
│
├── configs/
│   └── config.yaml              # Master config: paths, hyperparams, FL, LoRA, KG
│
├── data/
│   ├── audio/                   # Patient .wav speech recordings
│   ├── transcripts/             # Patient .txt speech transcripts
│   ├── mri/                     # Patient .nii MRI volumes
│   ├── ehr/                     # EHR CSV files
│   └── metadata.csv             # Index CSV: maps patient_id → all file paths + label
│
├── datasets/
│   ├── __init__.py
│   └── multimodal_dataset.py    # PyTorch Dataset for all 4 modalities
│
├── models/
│   ├── __init__.py
│   ├── audio_encoder.py         # Wav2Vec2 speech encoder
│   ├── text_encoder.py          # BERT text encoder
│   ├── mri_encoder.py           # 3D CNN MRI encoder
│   ├── ehr_encoder.py           # MLP EHR encoder
│   └── fusion_model.py          # Multimodal concat + classification head
│
├── training/
│   ├── __init__.py
│   ├── metrics.py               # Accuracy, Precision, Recall, F1, ROC-AUC
│   ├── evaluate.py              # Full validation / test inference loop
│   └── train.py                 # Training pipeline (AdamW, CosineAnnealingLR, checkpointing)
│
├── federated/
│   ├── __init__.py
│   ├── client.py                # Flower FL client (per hospital site)
│   └── server.py                # Flower FL server (FedAvg aggregation)
│
├── lora/
│   ├── __init__.py
│   └── lora_setup.py            # PEFT LoRA injection for BERT and Wav2Vec2
│
├── rag/
│   ├── __init__.py
│   ├── knowledge_graph.py       # Neo4j Medical Knowledge Graph (MCI/AD schema)
│   └── retriever.py             # LangChain-compatible RAG retriever
│
├── predict.py                   # CLI inference for a single patient
├── requirements.txt
└── README.md
```

---

## 🧩 Modules

### A. Speech Encoder — `models/audio_encoder.py`
- Backbone: [`facebook/wav2vec2-base`](https://huggingface.co/facebook/wav2vec2-base)
- Input: raw waveform tensor `(B, T)` at 16 kHz
- Output: mean-pooled hidden states → embedding `(B, 768)`
- LoRA-ready: pass `apply_lora=True` or use `lora/lora_setup.py`

### B. Text Encoder — `models/text_encoder.py`
- Backbone: [`bert-base-uncased`](https://huggingface.co/bert-base-uncased)
- Input: tokenised transcript `input_ids` + `attention_mask` `(B, 128)`
- Output: `[CLS]` pooler output → embedding `(B, 768)`
- LoRA targets: `query` and `value` projection matrices

### C. MRI Encoder — `models/mri_encoder.py`
- Architecture: 3-layer 3D CNN (`Conv3d → ReLU → MaxPool3d`) + `AdaptiveAvgPool3d(4,4,4)`
- Input: normalised NIfTI volume `(B, 1, 64, 64, 64)`
- Output: embedding `(B, 128)`

### D. EHR Encoder — `models/ehr_encoder.py`
- Architecture: `Linear(in) → ReLU → BatchNorm1d → Dropout(0.3) → Linear(out) → ReLU`
- Input: scaled tabular features `(B, n_ehr_features)`
- Output: embedding `(B, 64)`

### E. Fusion Classifier — `models/fusion_model.py`
- Concatenates all 4 embeddings → `(B, 768+768+128+64 = 1728)`
- Passes through `Linear → ReLU → BatchNorm → Dropout(0.4) → Linear → Linear`
- Output: logits `(B, 2)` → `argmax` gives 0=Healthy or 1=MCI

### F. Training Pipeline — `training/`
- **Optimizer**: AdamW with weight decay `1e-4`
- **Scheduler**: CosineAnnealingLR
- **Gradient Clipping**: `max_norm=1.0`
- **Checkpointing**: best model saved by F1-score to `checkpoints/best_model.pt`
- **Reproducibility**: explicit `random_state=42` everywhere

### G. Federated Learning — `federated/`
- Framework: [Flower (flwr)](https://flower.dev/)
- Strategy: **FedAvg** with dataset-size weighted metric aggregation
- `client.py` — locally trains for N epochs, serialises weights as numpy arrays
- `server.py` — orchestrates rounds, logs weighted-average accuracy/F1/AUC per round

### H. LoRA Fine-Tuning — `lora/lora_setup.py`
- Library: [PEFT](https://github.com/huggingface/peft)
- Default rank `r=8`, `alpha=16`, `dropout=0.1`
- Freezes all base model weights; only adapter layers are trained
- Dramatically reduces GPU memory and per-client compute in the FL setting

### I. Knowledge Graph + RAG — `rag/`
- **Graph DB**: Neo4j (`bolt://localhost:7687`)
- **Schema**: `Patient`, `Diagnosis`, `Biomarker`, `Gene`, `Drug`, `Symptom` nodes with typed relationships
- **Seed data**: MCI & Alzheimer's diagnoses, Amyloid-β42, Total Tau, Hippocampal Volume, APOE gene
- **Retriever**: LangChain `BaseRetriever` subclass; degrades gracefully without LangChain installed
- **Offline mode**: static clinical context strings returned when Neo4j is unreachable

---

## 📦 Requirements

```
torch
torchaudio
torchvision
transformers
peft
flwr
scikit-learn
pandas
numpy
nibabel
neo4j
langchain
langchain-core
pyyaml
```

Python **3.10+** is required. A CUDA-capable GPU is strongly recommended for encoder fine-tuning.

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/multimodal-mci-detection.git
cd multimodal-mci-detection/project

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note**: Installing `torch` with CUDA support may require a platform-specific command.
> See [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

---

## 📂 Data Preparation

### 1. Organise raw data
```
data/
├── audio/
│   ├── patient_001.wav
│   └── patient_002.wav
├── transcripts/
│   ├── patient_001.txt
│   └── patient_002.txt
├── mri/
│   ├── patient_001.nii
│   └── patient_002.nii
```

### 2. Create `data/metadata.csv`

This is the index file the `MultimodalDataset` uses to load everything.

```csv
patient_id,audio_path,text_path,mri_path,ehr_age,ehr_sex,ehr_mmse,ehr_education_years,label
patient_001,audio/patient_001.wav,transcripts/patient_001.txt,mri/patient_001.nii,72,1,26,16,0
patient_002,audio/patient_002.wav,transcripts/patient_002.txt,mri/patient_002.nii,68,0,23,12,1
```

- All `ehr_*` prefixed columns are automatically detected and passed through `StandardScaler`.
- `label`: `0` = Healthy, `1` = MCI
- Paths are **relative to `data/`** (set as `data_dir` in config).

### 3. Update `configs/config.yaml`
```yaml
model:
  ehr_in_features: 4    # Must match number of ehr_* columns in your CSV
```

---

## 🚀 Usage

All commands should be run from the **`project/`** root directory.

### 1. Centralised Training

Train all encoders + fusion head together on local data:

```bash
python -m training.train --config configs/config.yaml
```

Best model is saved to `checkpoints/best_model.pt`.

---

### 2. Federated Training

Simulates multi-site training across multiple clients. Each client keeps its data local.

**Terminal 1 — Start the aggregation server:**
```bash
python -m federated.server --config configs/config.yaml
```

**Terminal 2+ — Start each hospital client:**
```bash
# Site A
python -m federated.client --config configs/config.yaml --client-id 0

# Site B (separate terminal)
python -m federated.client --config configs/config.yaml --client-id 1
```

> In a real multi-site deployment, each `--client-id` maps to a different `metadata.csv`
> pointing to that site's local patient data.

FL training runs for `num_rounds` (set in `config.yaml`). Per-round weighted metrics
are logged on the server.

---

### 3. LoRA Fine-Tuning

Apply LoRA adapters to BERT and/or Wav2Vec2 before training to reduce trainable parameters by ~95%:

```python
from models.text_encoder import TextEncoder
from models.audio_encoder import AudioEncoder
from lora.lora_setup import apply_lora_to_text, apply_lora_to_audio

text_enc = TextEncoder()
text_enc.bert = apply_lora_to_text(text_enc.bert, r=8, alpha=16)
# Output: trainable params: 294,912 / 109,482,240 (0.27%)

audio_enc = AudioEncoder()
audio_enc.wav2vec2 = apply_lora_to_audio(audio_enc.wav2vec2, r=8, alpha=16)
```

---

### 4. Inference / Prediction

Run a single-patient prediction from the command line:

```bash
python predict.py \
  --config configs/config.yaml \
  --checkpoint checkpoints/best_model.pt \
  --audio data/audio/patient_003.wav \
  --text data/transcripts/patient_003.txt \
  --mri data/mri/patient_003.nii \
  --ehr "72,1,24.5,16,0,120,80"
```

**Example output:**
```
==================================================
Prediction     : Mild Cognitive Impairment (MCI)
Confidence     : 83.7%
  P(Healthy)   : 16.3%
  P(MCI)       : 83.7%
==================================================
```

---

### 5. Knowledge Graph + RAG

```python
from rag.knowledge_graph import MedicalKnowledgeGraph
from rag.retriever import MedicalGraphRetriever

# Connect to Neo4j and initialise schema (run once)
with MedicalKnowledgeGraph(uri="bolt://localhost:7687") as kg:
    kg.initialize_schema()

    # Query by medical concept
    retriever = MedicalGraphRetriever(kg)
    context = retriever.retrieve("What biomarkers are associated with MCI?")
    print(context.context_text)

    # Query by patient ID
    patient_ctx = retriever.retrieve_for_patient("patient_001")
    print(patient_ctx.context_text)
```

**As a LangChain retriever:**
```python
lc_retriever = retriever.as_langchain_retriever()
chain = RetrievalQA.from_chain_type(llm=your_llm, retriever=lc_retriever)
answer = chain.run("What are the early signs of MCI in speech patterns?")
```

> Neo4j is **optional**. When unreachable, the retriever falls back to a curated
> static MCI/AD knowledge base automatically.

---

## ⚙️ Configuration

All settings live in [`configs/config.yaml`](configs/config.yaml):

```yaml
data:
  metadata_path: "data/metadata.csv"
  data_dir: "data"

model:
  audio_model_name: "facebook/wav2vec2-base"
  text_model_name:  "bert-base-uncased"
  ehr_in_features: 50          # ← update to match your CSV
  fusion_hidden_dim: 256
  num_classes: 2

training:
  batch_size: 8
  learning_rate: 1.0e-4
  epochs: 20
  seed: 42

federated:
  server_address: "127.0.0.1:8080"
  num_rounds: 10
  min_clients: 2
  local_epochs: 2

lora:
  rank: 8
  alpha: 16
  dropout: 0.1

knowledge_graph:
  uri: "bolt://localhost:7687"
  user: "neo4j"
  password: "password"
```

---

## 📊 Metrics

The following metrics are computed on the validation / test set after every epoch and every FL round:

| Metric | Description |
|---|---|
| **Accuracy** | Overall fraction of correct predictions |
| **Precision** | TP / (TP + FP) — how many predicted MCI cases are true MCI |
| **Recall** | TP / (TP + FN) — how many true MCI cases are detected |
| **F1 Score** | Harmonic mean of Precision and Recall |
| **ROC-AUC** | Area under the Receiver Operating Characteristic curve |

> Best model is saved based on **F1-Score** (configurable via `checkpointing.save_best_metric`).

---

## 🔬 Research Background

Mild Cognitive Impairment is a transitional stage between normal aging and dementia. Early detection is clinically critical because:

- MCI-to-Alzheimer's conversion rate is **10–15% per year** (vs. 1–2% in healthy adults)
- Early intervention with cholinesterase inhibitors (e.g., Donepezil) can slow progression
- Multimodal biomarkers outperform single-modality approaches in sensitivity and specificity

**Key biomarkers addressed by this framework:**
- 🔊 **Speech**: reduced lexical diversity, increased pause duration, word-finding difficulties
- 📝 **Language**: semantic coherence, syntactic complexity, narrative structure
- 🧲 **MRI**: hippocampal atrophy, entorhinal cortex thinning, white matter hyperintensities
- 📊 **EHR**: MMSE score, age, APOE-ε4 status, education level, comorbidities

**Federated Learning motivation:**
- Patient neuroimaging and health records are subject to strict privacy regulations (HIPAA, GDPR)
- Multi-site collaboration is essential for diverse, representative datasets
- FedAvg enables model training across institutions without raw data sharing

---

## 🗺️ Roadmap

- [x] Wav2Vec2 audio encoder
- [x] BERT text encoder
- [x] 3D CNN MRI encoder
- [x] MLP EHR encoder
- [x] Multimodal fusion classifier
- [x] AdamW + CosineAnnealingLR training loop
- [x] Accuracy, Precision, Recall, F1, ROC-AUC metrics
- [x] Flower FedAvg federated training (server + client)
- [x] LoRA / PEFT adapter fine-tuning
- [x] Neo4j Knowledge Graph schema + Cypher queries
- [x] LangChain RAG retriever
- [x] Single-patient CLI inference
- [ ] Differential Privacy (DP-SGD) in FL client
- [ ] ADNI-specific MRI preprocessing (skull stripping, spatial normalisation)
- [ ] Docker Compose for local Neo4j + server deployment
- [ ] pytest unit test suite for all encoder modules
- [ ] Experiment tracking with MLflow or Weights & Biases
- [ ] Web dashboard for federated round monitoring

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [ADNI — Alzheimer's Disease Neuroimaging Initiative](http://adni.loni.usc.edu/)
- [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base) — Meta AI
- [bert-base-uncased](https://huggingface.co/bert-base-uncased) — Google Research
- [Flower Federated Learning](https://flower.dev/) — Adap
- [PEFT / LoRA](https://github.com/huggingface/peft) — HuggingFace
- [LangChain](https://www.langchain.com/)
- [Neo4j](https://neo4j.com/)
