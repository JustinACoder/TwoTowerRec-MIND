# TwoTowerRec-MIND

A news recommendation system implementation comparing Two-Tower neural architecture against Matrix Factorization baseline on the Microsoft News Dataset (MIND).

This project was developed as part of the course "Computation Intensive Statistics" (MATH 680) at McGill University.

## Overview

This project implements and evaluates a Two-Tower recommender system for news article recommendations using the MIND dataset. The Two-Tower architecture uses separate neural networks to learn embeddings for users and items in a shared latent space, enabling efficient similarity-based recommendations.

## Dataset

The project uses the MIND (Microsoft News Dataset) which contains:
- User click histories and impression logs
- News article metadata (titles, abstracts, categories, subcategories)
- Entity embeddings from knowledge graphs

Both small and large versions of the dataset are supported.

## Models

**Two-Tower Model**: Neural network with separate towers for users and items. User features include history embeddings and history length. Item features use text embeddings from MiniLM on combined title and abstract. Loss function is InfoNCE adapted for multi-positive impressions.

**Matrix Factorization Baseline**: Alternating Least Squares (ALS) implementation using the `implicit` library for comparison.

## Project Structure

- `01_data_preprocessing.ipynb` - Downloads MIND dataset and creates user/item feature CSVs
- `02_training_twotower.ipynb` - Trains the Two-Tower model
- `03_training_mf.ipynb` - Trains the Matrix Factorization baseline
- `04_evaluation.ipynb` - Evaluates both models on validation set
- `two_tower_model.py` - Two-Tower architecture implementation
- `dataset.py` - PyTorch Dataset and DataLoader utilities

## Requirements

```
numpy
sentence-transformers
pandas
torch
implicit
matplotlib
torchmetrics
scikit-learn
```

Install with: `pip install -r requirements.txt`

Note: On Windows, Visual Studio is required to install the `implicit` library.

Note2: To use GPU acceleration, ensure PyTorch is installed with CUDA support.

Note3: In some environments, I had some issues with the `sentence-transformers` library. If you encounter problems, try changing the version.

## Usage

Run the notebooks in order (01 → 02 → 03 → 04) to preprocess data, train models, and evaluate results.

**Warning**: Some scripts are computationally intensive and may require significant time and resources, especially on the large MIND dataset. I recommend using a cloud service with GPU support (like Lambda GPU). If you have memory issues, consider reducing batch sizes or using the small MIND dataset.

## Conclusion
Although the Two-Tower model achieved better performance than the Matrix Factorization baseline, the improvement was modest.
I suspect many reasons for this, including limited hyperparameter tuning, extremely sparse user-item interactions (around 0.005% positive user-item pairs), and the lack of richer user features beyond click history limiting the model's ability to capture user preferences and context effectively. Further improvements could be explored by incorporating additional user features, experimenting with different architectures, and conducting more extensive
hyperparameter tuning.