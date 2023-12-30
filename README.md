# Fine-tune Phi-2

## Notebooks

- `nb_dataset.ipynb`: Create a synthetic conversational dataset using a seed of riddles
- `nb_qlora.ipynb`: Fine-tune Phi-2 using QLoRA 

## `qlora.py`: Train on multiple GPUs with accelerate

### Overview

1. **Setup and Initialization**: Import necessary libraries, set up Weights and Biases (wandb) for tracking, and initialize a unique run identifier.

2. **Configuration and Seeds**: Set the seed for reproducibility and configure model and dataset paths, learning rate, batch sizes, epochs, and maximum token length.

3. **LoRA Configuration**: Define the Low-Rank Adaptation (LoRA) configuration for efficient model fine-tuning.

4. **Model Preparation**: Load the Phi-2 model with quantization settings for 4-bit training, and resize token embeddings to accommodate new special tokens.

5. **Tokenizer Preparation**: Load and configure the tokenizer, adding necessary special tokens for ChatML formatting.

6. **Dataset Loading and Preparation**: Load the dataset from Hugging Face, split it into training and test sets, and apply ChatML formatting and tokenization.

7. **Data Collation**: Define a collation function to transform individual data samples into batched data suitable for training.

8. **Training Configuration**: Set up training arguments with specified hyperparameters like batch sizes, learning rate, and gradient accumulation steps.

9. **Trainer Initialization**: Initialize the Trainer object with the model, tokenizer, training arguments, and data collator.

10. **Training Execution**: Launch the training process, optionally with Weights and Biases tracking for the main process.

### Set parameters

```python
modelpath="microsoft/phi-2"
dataset_name="g-ronimo/riddles_evolved"
lr=0.00002		# low but works for this dataset
bs=1		# batch size for training
bs_eval=16		# batch size for evals
ga_steps=16		# gradient acc. steps
epochs=20		# dataset is small, many epochs needed
max_length=1024		# samples will be cut beyond this number of tokens
output_dir=f"out"
```

### Run

```bash
accelerate launch qlora.py
```

