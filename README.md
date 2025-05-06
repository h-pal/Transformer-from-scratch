# Transformers from Scratch

This repository contains an implementation of a Transformer model from scratch and a custom trainer designed for masked language modeling (MLM). The project demonstrates the fundamental concepts of Transformer architectures and provides a complete pipeline for training and evaluation.

## Files in this Repository

### 1. `model.py`

* Implements the Transformer model from scratch.
* Key Features:

  * Encoder and Decoder blocks with multi-head attention.
  * Position-wise feedforward networks.
  * Scaled dot-product attention mechanism.

### 2. `trainer.py`

* Provides a custom training pipeline for the Transformer model.
* Includes:

  * Data preprocessing and vocabulary creation.
  * Dataset wrapper for handling masked language modeling tasks.
  * Training loop with logging using [Weights & Biases](https://wandb.ai/).
  * Validation with perplexity and top-k accuracy metrics.

## Key Highlights

### Model Architecture

The Transformer is implemented following the original architecture outlined in the "Attention is All You Need" paper. The model supports customization for:

* Number of encoder/decoder blocks.
* Attention heads.
* Feedforward dimensions.
* Input and output vocabularies.

### Training Pipeline

The training process involves:

1. **Data Preprocessing**:

   * Tokenizes and cleans text data.
   * Builds a vocabulary with special tokens (`<START>`, `<END>`, `<MSK>`, `<UNK>`).
2. **Masked Language Modeling**:

   * Randomly masks input tokens for predicting their values.
3. **Metrics and Logging**:

   * Tracks loss, perplexity, and top-k accuracy during training and validation.

### Dataset

The training uses the Wikitext-103 dataset, downloaded via the Hugging Face `datasets` library. Preprocessing includes lowercasing, punctuation removal, and filtering by a predefined English vocabulary.

### Requirements

* Python 3.8 or higher
* CUDA-compatible GPU (optional but recommended)
* Required libraries:

  ```
  torch
  nltk
  transformers
  wandb
  datasets
  tqdm
  ```

### Usage

#### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/transformers-from-scratch.git
   cd transformers-from-scratch
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

#### Running the Code

1. **Preprocess the Data**:
   The data is preprocessed automatically during the training phase in `trainer.py`.

2. **Train the Model**:
   Run the `trainer.py` script to train the Transformer model:

   ```bash
   python trainer.py
   ```

3. **Monitoring Training**:
   Track progress on Weights & Biases by creating a project and using your API key.

#### Customization

Adjust hyperparameters such as:

* Number of encoder/decoder layers.
* Feedforward dimensions.
* Vocabulary size.
  Modify these directly in `trainer.py`.

### Outputs

* Logs for training and validation.
* Metrics like loss, perplexity, and top-k accuracy.
* Model checkpoints (optional, if implemented).

### Acknowledgments

* The Transformer model architecture is based on the original ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) paper by Vaswani et al.
* Thanks to Hugging Face for the Wikitext-103 dataset and tooling.
