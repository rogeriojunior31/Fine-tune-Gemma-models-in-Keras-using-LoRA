# Fine-tune Gemma Model with Keras using LoRA

This Colab Notebook demonstrates how to fine-tune the Gemma language model using Keras and Low Rank Adaptation (LoRA) on the Databricks Dolly 15k dataset.

## Table of Contents

- [Installation and Setup](#installation-and-setup)
    - [Install Dependencies](#install-dependencies)
    - [Kaggle Configuration](#kaggle-configuration)
    - [Backend Selection](#backend-selection)
    - [Import Dependencies](#import-dependencies)    
- [Load Dataset](#load-dataset)
- [Load Model](#load-model)
- [Inference before Fine-tuning](#inference-before-fine-tuning)
- [LoRA Fine-tuning](#lora-fine-tuning)
- [Inference after Fine-tuning](#inference-after-fine-tuning)

## Installation and Setup

## Install Dependencies

The notebook starts by installing the necessary dependencies, including `kaggle`, `keras-nlp`, `keras`, and `datasets`.

## Kaggle Configuration

The Kaggle API is configured by copying the `kaggle.json` file to the appropriate directory and setting the required permissions.

## Backend Selection

The backend for Keras is set to `jax`, and the `XLA_PYTHON_CLIENT_MEM_FRACTION` environment variable is set to `1.00`.

## Import Dependencies

The required libraries, `keras` and `keras_nlp`, are imported.

## Load Dataset

The Databricks Dolly 15k dataset is loaded from a JSON lines file. Examples with empty context are filtered out, and the data is formatted as a single string template.

## Load Model

The Gemma language model is loaded using `GemmaCausalLM.from_preset("gemma_2b_en")`. The model summary is displayed, showing the total number of parameters and the number of trainable and non-trainable parameters.

## Inference before Fine-tuning

Two example prompts, "Brazil Trip" and "ELI5 Photosynthesis," are used to generate responses from the model before fine-tuning.

## LoRA Fine-tuning

LoRA is enabled for the model's backbone with a rank of 4. The number of trainable parameters is significantly reduced when LoRA is enabled.The input sequence length is limited to 512 to control memory usage. The AdamW optimizer is used with a learning rate of 5e-4 and weight decay of 0.01. Layer normalization and bias terms are excluded from weight decay.The model is compiled with sparse categorical cross-entropy loss and sparse categorical accuracy metric. Fine-tuning is performed for one epoch with a batch size of 1.

## Inference after Fine-tuning

The same example prompts, "Brazil Trip" and "ELI5 Photosynthesis," are used to generate responses from the model after fine-tuning, demonstrating the improved quality of the generated text.