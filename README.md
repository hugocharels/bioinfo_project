# DeepLoc Reimplementation

This project re-implements the **DeepLoc** model: a deep learning method for predicting protein subcellular localization from amino acid sequences only, using convolutional and recurrent neural networks with attention mechanisms.

## Overview

DeepLoc is an end-to-end deep learning model that processes entire protein sequences to predict one of ten subcellular localization classes, as well as whether the protein is membrane-bound or soluble.

## Architecture Summary

- **Input**: Protein sequence encoded as a profile or BLOSUM62 matrix.
- **CNN Layer**: Motif detection with filters of sizes [1, 3, 5, 9, 15, 21].
- **Conv Layer**: 128 filters of size (3 × 120).
- **BiLSTM Layer**: 256 units per direction (output: 1000 × 512).
- **Attention Layer**: LSTM with 512 units for 10 decoding steps.
- **Dense Layers**: 512 units fully connected.
- **Output**:
    - 10-class softmax for localization
    - Binary sigmoid for membrane/soluble prediction

## Reimplementation Steps

### 1. 📦 Dependencies

Install the required Python packages:

```bash
python -m venv venv
source venv/bin/activate
pip install numpy scipy theano lasagne biopython pandas scikit-learn
```

> 📌 Note: The original implementation used Python 2.7, Theano 0.9.0, and Lasagne 0.2. You may use PyTorch or TensorFlow if reimplementing from scratch.

### 2. 📄 Data Preparation

#### Download Dataset

- Download the DeepLoc dataset: https://services.healthtech.dtu.dk/services/DeepLoc-2.1/

#### Filtering Criteria

- Eukaryotic
- Non-fragments
- Nuclear-encoded
- > 40 amino acids
- Experimental annotation (ECO:0000269)
- Single localization label

#### Encoding

- **Preferred**: Protein profile (e.g., from PSI-BLAST or TOPCONS)
- **Alternative**: BLOSUM62 or one-hot encoding

#### Sequence Handling

- Pad/truncate sequences to max length 1000
- Middle truncation for long sequences (preserve N- and C-terminal)

### 3. 🧠 Model Architecture

Implement the network as:

Input → CNN (6 filter sizes × 20 filters) → Conv Layer (128 filters) →
BiLSTM (256×2) → Attention LSTM (512) × 10 steps →
Dense (512) → [Softmax (10) & Sigmoid (1)]

- Attention mechanism follows Bahdanau-style soft attention.
- Hierarchical tree classifier (optional): Use if mimicking the full DeepLoc behavior.

### 4. 🏋️ Training

#### Hyperparameters

- Optimizer: SGD or Adam
- Loss: Cross-entropy (multi-class) + Binary cross-entropy (membrane predictor)
- Epochs: 150
- Max sequence length: 1000
- Batch size: 32–64
- Cross-validation: 5-fold homology-partitioned

#### Class Imbalance

- Use a cost-sensitive loss or reweighting

### 5. 🧪 Evaluation

#### Metrics

- **Subcellular Localization**: Accuracy, Gorodkin score
- **Membrane/Soluble**: Accuracy, Matthews Correlation Coefficient (MCC)

#### Baseline Comparison

- Compare with models like LocTree2, MultiLoc2, iLoc-Euk, etc.

### 6. 📊 Visualization (Optional)

- Use t-SNE to project the attention-based context vectors.
- Visualize attention weights over sequence positions to interpret biological signal focus (e.g., N-terminal peptides).

## Resources

- 🔗 [Original GitHub (archived)](https://github.com/JJAlmagro/subcellular_localization)
- 📄 [Paper](https://academic.oup.com/bioinformatics/article/33/21/3387/3931857)
- 🧬 [DeepLoc Webserver](http://www.cbs.dtu.dk/services/DeepLoc/)

## Acknowledgments

This reimplementation is based on the architecture and methodology described in:

**Almagro Armenteros et al.,** DeepLoc: prediction of protein subcellular localization using deep learning. _Bioinformatics_ (2017).
