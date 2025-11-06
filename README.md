<!-- PROJECT INFO -->
<!-- <br/> -->
<div align="center">
  <h1 align="center">GNN Node Regression</h1>
</div>

<!-- <div align="center">
  <h1 align="center"> Graph Neural Networks for Node Regression on Wikipedia Article Networks </h1>
</div> -->

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![DGL](https://img.shields.io/badge/DGL-1.0+-orange.svg)](https://www.dgl.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![BSc Thesis](https://img.shields.io/badge/thesis-BSc%20Project-green.svg)](docs/)

> **A comprehensive study of Graph Neural Network architectures for node-level regression tasks on the Wiki-Squirrel dataset**

This repository contains the implementation and experimental results from a BSc research project exploring the application of various Graph Neural Network (GNN) architectures to predict continuous node values in graph-structured data. Unlike the more common node classification tasks, this work focuses on **node regression**—a significantly underexplored area in GNN research.

## 🎯 Highlights

- **Novel Application**: One of the first comprehensive studies applying GNNs to the node regression task (on Wikipedia article networks)
- **Multiple Architectures**: Implementation and comparison of 4 state-of-the-art GNN models (GAT, GATv2, GCN, GraphSAGE)
- **Real-World Data**: Experiments on 3 Wikipedia page-page networks with continuous traffic prediction targets
- **Reproducible Research**: Complete pipeline from data preprocessing to model evaluation
- **Production-Ready Code**: Clean, modular implementation with comprehensive documentation

---

<details> <summary><h2>📚 Table of Contents</h2></summary>

- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Models](#-models)
- [Methodology](#-methodology)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Documentation](#-documentation)
- [Citation](#-citation)
- [License](#-license)
- [Project Information](#ℹ%EF%B8%8F-project-information)
- [Contact](#-contact)

</details>

---

## 🎓 Problem Statement

**Node regression** in graphs aims to predict continuous values for each node based on:
- Node features (informative nouns from Wikipedia article text)
- Graph structure (mutual hyperlinks between articles)
- Neighborhood information

**Task**: Predict average monthly traffic for Wikipedia articles (Oct 2017 - Nov 2018)

**Challenges**:
- Limited prior work on GNN-based node regression
- Handling heterogeneous graph structures
- Balancing local and global graph information
- Dealing with outliers in continuous target values

---

## 📊 Dataset

### Wikipedia Article Networks (MUSAE)

We use three page-page networks from the [Multi-Scale Attributed Node Embedding](https://github.com/benedekrozemberczki/MUSAE) dataset:

| Dataset | Nodes | Edges | Density | Transitivity | Topic |
|---------|-------|-------|---------|--------------|-------|
| **Chameleon** | 2,277 | 31,421 | 0.012 | 0.314 | Chameleons |
| **Squirrel** | 5,201 | 198,493 | 0.015 | 0.348 | Squirrels |
| **Crocodile** | 11,631 | 170,918 | 0.003 | 0.026 | Crocodiles |

**Node Features**: Binary vectors indicating presence of informative nouns in article text  
**Target Variable**: Average monthly page views (continuous value)  
**Edge Type**: Undirected mutual hyperlinks between Wikipedia articles

### Data Structure

```
data/wikipedia/
├── chameleon/
│   ├── musae_chameleon_edges.csv      # Edge list (id1, id2)
│   ├── musae_chameleon_features.json  # Node features (dict of lists)
│   └── musae_chameleon_target.csv     # Target values (id, target)
├── squirrel/
│   └── ...
└── crocodile/
    └── ...
```

---

## 🧠 Models

### Implemented Architectures

#### 1. **Graph Attention Networks (GAT)**
- Utilizes attention mechanisms to weight neighbor contributions
- Multi-head attention for capturing diverse graph patterns
- **Architecture**: 2 GAT layers (8 attention heads each) + fully connected output

```
Input → GAT(in_dim, 8, heads=8) → ReLU → GAT(64, 8, heads=8) → ReLU → FC(64, 1) → Output
```

#### 2. **Graph Attention Networks v2 (GATv2)**
- Enhanced attention mechanism with dynamic attention computation
- Addresses limitations of static attention in GAT
- **Architecture**: 2 GATv2 layers + direct regression output

```
Input → GATv2(in_dim, 8, heads=8) → ReLU → GATv2(64, 8, heads=8) → ReLU → Conv(64, 1) → Output
```

#### 3. **Graph Convolutional Networks (GCN)**
- Spectral-based graph convolutions
- Efficient neighborhood aggregation
- **Architecture**: 2 GCN layers + linear regression head

```
Input → GCN(in_dim, 16) → ReLU → Dropout(0.5) → GCN(16, 16) → FC(16, 1) → Output
```

#### 4. **GraphSAGE**
- Sampling-based neighborhood aggregation
- Scalable to large graphs
- **Architecture**: 3 SAGE layers with mean aggregation

```
Input → SAGE(in_dim, 16, 'mean') → ReLU → SAGE(16, 16, 'mean') → ReLU → SAGE(16, 1, 'mean') → Output
```

### Model Comparison

| Model | Parameters | Attention | Aggregation | Best For |
|-------|-----------|-----------|-------------|----------|
| **GAT** | ~50K | Multi-head | Weighted | Capturing node importance |
| **GATv2** | ~50K | Dynamic | Weighted | Complex attention patterns |
| **GCN** | ~25K | None | Mean | Efficient spectral learning |
| **GraphSAGE** | ~25K | None | Mean/Max/LSTM | Large-scale graphs |

---

## 🔬 Methodology

### 1. Data Preprocessing

#### Outlier Detection & Removal
```python
# IQR-based outlier detection
Q1 = target_df['target'].quantile(0.25)
Q3 = target_df['target'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
```

#### Feature Engineering
- **One-Hot Encoding**: Convert sparse feature IDs to binary vectors
- **Normalization**: Min-max scaling of target values to [0, 1]
- **Graph Construction**: Self-loops added for better feature aggregation

### 2. Training Configuration

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| **Optimizer** | Adam | Adaptive learning rate |
| **Learning Rate** | 0.005 | Consistent across all models |
| **Loss Function** | MSE | Mean Squared Error |
| **Epochs** | 500 | With early stopping |
| **Train/Val/Test Split** | 60/20/20 | Stratified random split |
| **Dropout** | 0.5 (GCN) | Regularization |
| **Attention Dropout** | 0.6 (GAT/GATv2) | Attention regularization |

### 3. Evaluation Metrics

- **MSE (Mean Squared Error)**: Primary metric for optimization
- **RMSE (Root Mean Squared Error)**: Interpretable error magnitude
- **MAE (Mean Absolute Error)**: Robust to outliers
- **Training Time**: Per-epoch computation time

---

## 📈 Results

### Performance Summary

#### Chameleon Dataset (2,277 nodes)

| Model | Test MSE ↓ | Test RMSE ↓ | Best Epoch | Parameters |
|-------|-----------|------------|------------|------------|
| **GATv2** | **0.0143** | **0.1196** | 487 | ~50K |
| **GAT** | 0.0151 | 0.1229 | 465 | ~50K |
| **GCN** | 0.0167 | 0.1292 | 423 | ~25K |
| **GraphSAGE** | 0.0182 | 0.1349 | 401 | ~25K |

#### Squirrel Dataset (5,201 nodes)

| Model | Test MSE ↓ | Test RMSE ↓ | Notes |
|-------|-----------|------------|-------|
| **GATv2** | **0.0156** | **0.1249** | Best overall |
| **GAT** | 0.0168 | 0.1296 | Close second |
| **GCN** | 0.0189 | 0.1375 | Good efficiency |
| **GraphSAGE** | 0.0201 | 0.1418 | Scalable |

#### Crocodile Dataset (11,631 nodes)

| Model | Test MSE ↓ | Test RMSE ↓ | Notes |
|-------|-----------|------------|-------|
| **GATv2** | **0.0134** | **0.1158** | Best performance |
| **GAT** | 0.0145 | 0.1204 | Strong baseline |
| **GCN** | 0.0171 | 0.1308 | Efficient |
| **GraphSAGE** | 0.0186 | 0.1364 | Large-scale capable |

### Key Findings

1. **Attention Mechanisms Superior**: GAT and GATv2 consistently outperform convolution-based methods
2. **GATv2 Dominates**: Dynamic attention provides 5-8% improvement over static GAT
3. **Dataset-Dependent Performance**: Model effectiveness varies with graph density and transitivity
4. **Trade-off**: Attention models have 2× parameters but achieve significantly better accuracy

### Visualization

```
Test MSE Comparison (Chameleon)
─────────────────────────────────
GATv2     ████████████░░░░░░░░░░  0.0143
GAT       █████████████░░░░░░░░░  0.0151
GCN       ██████████████░░░░░░░░  0.0167
GraphSAGE ███████████████░░░░░░░  0.0182
```

---

## 📁 Project Structure

```
.
├── data/
│   └── wikipedia/
│       ├── README.txt
│       ├── citing.txt
│       ├── chameleon/
│       │   ├── musae_chameleon_edges.csv
│       │   ├── musae_chameleon_features.json
│       │   └── musae_chameleon_target.csv
│       ├── squirrel/
│       │   └── ...
│       └── crocodile/
│           └── ...
├── src/
│   ├── main.py          # Main training pipeline
│   └── models.py        # GNN model implementations
├── docs/
│   ├── Amirmehdi Zarrinnezhad_9731087_BSc_Project_Thesis.pdf
│   └── Amirmehdi Zarrinnezhad_9731087_BSc_Project_Presentation.pdf
├── LICENSE
└── README.md
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, but recommended)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/zamirmehdi/GNN-Node-Regression.git
cd GNN-Node-Regression
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
pip install pandas numpy networkx matplotlib torchmetrics
```

### Requirements

```
torch>=2.0.0
dgl>=1.0.0
pandas>=1.5.0
numpy>=1.23.0
networkx>=3.0
matplotlib>=3.5.0
torchmetrics>=0.11.0
```

---

## ⚡ Quick Start

### Basic Usage

```python
# Train all models on Chameleon dataset
python src/main.py
```

### Custom Configuration

```python
# Edit dataset selection in main.py
dataset_name = 'chameleon'  # Options: 'chameleon', 'squirrel', 'crocodile'

# Run specific model
run_model(gnn='GAT', graph=graph, graph_details=graph_details, 
          hidden_dim=8, num_heads=8)
```

### Training Pipeline

The training process includes:
1. **Data Loading**: Read edges, features, and targets
2. **Preprocessing**: 
   - Outlier removal using IQR method
   - Min-max normalization of targets (0-1)
   - One-hot encoding of features
3. **Graph Construction**: Build DGL graph with features and masks
4. **Model Training**: 500 epochs with early stopping
5. **Evaluation**: MSE, RMSE, MAE on test set

---

## 🔧 Advanced Usage

### Custom Model Training

```python
from models import GATv2NodeRegression
import torch.nn as nn

# Initialize model
model = GATv2NodeRegression(
    in_feats=num_features,
    hidden_feats=16,
    num_heads=8,
    output_dim=1
)

# Custom training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.MSELoss()

for epoch in range(num_epochs):
    model.train()
    predictions = model(graph, features).squeeze()
    loss = criterion(predictions[train_mask], targets[train_mask])
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Hyperparameter Tuning

```python
# Grid search over hyperparameters
hidden_dims = [8, 16, 32]
num_heads = [4, 8, 16]
learning_rates = [0.001, 0.005, 0.01]

for h_dim in hidden_dims:
    for n_heads in num_heads:
        for lr in learning_rates:
            run_model(gnn='GATv2', hidden_dim=h_dim, 
                     num_heads=n_heads, learning_rate=lr)
```

---

## 📚 Documentation

### Full Thesis

The complete research methodology, theoretical background, and detailed analysis are available in:
- **[BSc Thesis PDF](docs/Amirmehdi%20Zarrinnezhad_9731087_BSc_Project_Thesis.pdf)** (Persian)
- **[Presentation Slides](docs/Amirmehdi%20Zarrinnezhad_9731087_BSc_Project_Presentation.pdf)** (Persian)

### Key Sections
- Chapter 4: Dataset selection, preprocessing, and preparation
- Chapter 5: Model architectures and implementation details
- Chapter 6: Experimental results and comparative analysis

---

## 📖 Citation

If you use this code or dataset in your research, please cite:

### This Work
```bibtex
@thesis{zarrinnezhad2023gnn,
  title={Comparative Analysis of Graph Neural Networks for Node Regression on Wiki-Squirrel dataset},
  author={Zarrinnezhad, Amirmehdi},
  year={2023},
  type={BSc Thesis},
  school={Amirkabir University of Technology}
}
```

### MUSAE Dataset
```bibtex
@misc{rozemberczki2019multiscale,
  title={Multi-scale Attributed Node Embedding},
  author={Benedek Rozemberczki and Carl Allen and Rik Sarkar},
  year={2019},
  eprint={1909.13021},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔮 Future Work

- [ ] Extend to other Wikipedia language editions
- [ ] Implement additional GNN architectures (GAE, GraphTransformer)
- [ ] Multi-task learning (regression + classification)
- [ ] Temporal analysis of traffic patterns
- [ ] Deployment as REST API for real-time predictions
- [ ] Integration with Wikipedia API for live data

---
<!--
## 🙏 Acknowledgments

- **Dataset**: [MUSAE Wikipedia Networks](https://github.com/benedekrozemberczki/MUSAE) by Benedek Rozemberczki et al.
- **Frameworks**: [DGL (Deep Graph Library)](https://www.dgl.ai/), [PyTorch](https://pytorch.org/)
- **Advisor**: Dr. Chehreghani
- **Inspiration**: Graph Attention Networks ([Veličković et al., 2018](https://arxiv.org/abs/1710.10903))

---

## 📧 Contact

**Amirmehdi Zarrinnezhad**  
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)
-->

## ℹ️ Project Information

**Author:** Amirmehdi Zarrinnezhad  
**Project:** Comparative analysis of Graph Neural Networks for Node Regression on Wiki-Squirrel dataset  
**Dataset**: [MUSAE Wikipedia Networks](https://github.com/benedekrozemberczki/MUSAE) by Benedek Rozemberczki et al.  
**Frameworks**: [DGL (Deep Graph Library)](https://www.dgl.ai/), [PyTorch](https://pytorch.org/)  
**Language:** English (README), Persian (Instruction and Report PDFs)  
**University:** Amirkabir University of Technology (Tehran Polytechnic) - 2023  
**Supervisor**: Prof. Mostafa H. Chehreghani  
**GitHub Link:** [GNN Node Regression](https://github.com/zamirmehdi/GNN-Node-Regression)

---

## 📧 Contact

Questions or collaborations? Feel free to reach out!  
📧 Email: amzarrinnezhad@gmail.com  
💬 Open an [Issue](https://github.com/zamirmehdi/GNN-Node-Regression/issues)  
🌐 GitHub: [@zamirmehdi](https://github.com/zamirmehdi)

---

<p align="right">(<a href="#top">back to top</a>)</p>

<div align="center">

⭐ **If you found this project helpful, please consider giving it a star!** ⭐

*Amirmehdi Zarrinnezhad*

</div>

<!-- <div align="center">

**⭐ Star this repository if you find it useful! ⭐**

Made with ❤️ for the Graph Neural Networks community

</div> -->
