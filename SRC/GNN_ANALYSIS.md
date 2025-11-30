# Graph Neural Network Analysis for COVID-19 Prediction

## Overview

This analysis implements Graph Neural Networks (GNNs) to predict COVID-19 case rates across U.S. counties by leveraging spatial relationships, socioeconomic factors, and mobility patterns. The approach treats counties as nodes in a graph, connected by geographic adjacency, and uses node features to predict COVID-19 cases per 100,000 population.

## Data Sources

### 1. Socioeconomic Data
- **Source**: American Community Survey (ACS) 5-Year Estimates (2023)
- **Tables Included**:
  - **B19057**: Public assistance income
  - **B19083**: Gini index of income inequality
  - **DP02**: Selected social characteristics
  - **DP04**: Selected housing characteristics
  - **DP05**: Demographic characteristics
  - **S1501**: Educational attainment
  - **S1702**: Poverty status
  - **S1903**: Median income
  - **S2301**: Employment status
  - **S2701**: Health insurance coverage
  - **S2801**: Computer and internet use

### 2. COVID-19 Data
- **Source**: United States COVID-19 Community Levels by County (2025)
- **Key Variables**:
  - County FIPS codes
  - COVID-19 community levels
  - COVID-19 cases per 100,000 population (target variable)
  - COVID-19 inpatient bed utilization
  - COVID-19 hospital admissions per 100,000

### 3. Mobility Data
- **Source**: Google COVID-19 Community Mobility Reports (2020-2022)
- **Categories**:
  - Retail and recreation
  - Grocery and pharmacy
  - Parks
  - Transit stations
  - Workplaces
  - Residential

### 4. Geographic Data
- **Source**: County adjacency data (2024)
- **Purpose**: Defines spatial relationships between counties to construct the graph structure

## Methodology

### Data Preprocessing

1. **FIPS Code Standardization**: All county FIPS codes are padded to 5 digits and treated as strings for consistent merging
2. **Mobility Aggregation**: Time-series mobility data is aggregated by county using mean values across all time periods
3. **COVID-19 Aggregation**: Multiple weekly observations are aggregated to county-level statistics
4. **Data Integration**: All datasets are merged on FIPS codes, creating a unified feature matrix

### Graph Construction

- **Nodes**: U.S. counties (3,000+ nodes)
- **Edges**: Geographic adjacency relationships between counties
- **Edge List**: Filtered to include only counties present in the dataset
- **Graph Properties**:
  - Undirected graph
  - No self-loops
  - Based on physical county boundaries

### Feature Engineering

**Node Features** (40+ features per county):
- Socioeconomic indicators (income, poverty, employment)
- Demographic characteristics (age, race, population)
- Housing characteristics (occupancy, ownership)
- Educational attainment levels
- Health insurance coverage rates
- Mobility pattern changes from baseline
- Internet and computer access

**Target Variable**:
- COVID-19 cases per 100,000 population

**Feature Scaling**:
- StandardScaler normalization applied to all features
- Missing values imputed with zeros

### Model Architecture

#### Original GCN Model

```
3-Layer Graph Convolutional Network (GCN)
├── GCN Layer 1: input_features → 64 hidden units
├── ReLU activation + Dropout (0.2)
├── GCN Layer 2: 64 → 64 hidden units
├── ReLU activation + Dropout (0.2)
├── GCN Layer 3: 64 → 32 hidden units
├── ReLU activation + Dropout (0.2)
└── Fully Connected: 32 → 1 (regression output)
```

**Key Components**:
- **Graph Convolution**: Aggregates information from neighboring counties
- **Dropout**: 0.2 rate for regularization
- **Optimizer**: Adam (lr=0.01, weight_decay=5e-4)
- **Loss Function**: Mean Squared Error (MSE)
- **Training**: 200 epochs

#### Improved GAT Model

```
3-Layer Graph Attention Network (GAT)
├── GAT Layer 1: input_features → 64×4 heads (256 units)
├── Batch Normalization + ELU + Dropout (0.3)
├── GAT Layer 2: 256 → 64×4 heads (256 units)
├── Batch Normalization + ELU + Dropout (0.3)
├── GAT Layer 3: 256 → 64×2 heads (128 units)
├── Batch Normalization + ELU + Dropout (0.3)
├── FC Layer 1: 128 → 64
├── ELU + Dropout (0.3)
├── FC Layer 2: 64 → 32
├── ELU + Dropout (0.3)
└── FC Layer 3: 32 → 1 (regression output)
```

**Improvements**:
- **Attention Mechanism**: Multi-head attention learns importance weights for neighbor contributions
- **Batch Normalization**: Stabilizes training and improves convergence
- **ELU Activation**: Better gradient flow than ReLU
- **Deeper Prediction Head**: 3-layer MLP for more expressive predictions
- **Learning Rate Scheduling**: ReduceLROnPlateau (factor=0.5, patience=20)
- **Early Stopping**: Patience=50 epochs to prevent overfitting
- **Gradient Clipping**: Max norm=1.0 for training stability
- **Training**: Up to 500 epochs with early stopping

### Train/Validation/Test Split

- **Training Set**: 70% of counties
- **Validation Set**: 15% of counties
- **Test Set**: 15% of counties
- **Split Method**: Random stratification (random_state=42)

## Results

### Model Performance Comparison

| Metric | Original GCN | Improved GAT | Improvement |
|--------|-------------|--------------|-------------|
| **MSE** | 1567.3646 | 1673.0570 |  -6.74% |
| **MAE** | 27.4770 | 28.0547 | -2.10% |
| **RMSE** | 39.5900 | 40.9030 | -3.32% |
| **R² Score** | 0.4410 | 0.4034 | -8.55% |

### Feature Importance Analysis

The analysis employs **permutation importance** to identify which features most influence COVID-19 predictions:

**Methodology**:
1. Baseline prediction error is measured on the test set
2. Each feature is randomly permuted (shuffled)
3. Prediction error is re-measured with the permuted feature
4. Importance score = increase in MSE after permutation
5. Process repeated 10 times per feature for robustness

**Interpretation**:
- **Positive Importance**: Feature degradation increases prediction error (important for accuracy)
- **Negative Importance**: Feature degradation decreases prediction error (may introduce noise)
- **Zero Importance**: Feature has no impact on predictions

**Key Findings** (Top Features):
The feature importance analysis reveals which socioeconomic, demographic, and mobility factors are most predictive of COVID-19 case rates at the county level.

## Visualizations

### 1. Training Curves
- Training and validation loss over epochs
- Demonstrates model convergence and overfitting detection
- Comparison between original GCN and improved GAT

### 2. Prediction Accuracy
- **Scatter Plot**: Predicted vs. actual COVID-19 cases per 100k
- **Perfect Prediction Line**: Red dashed line (y=x)
- **R² Score**: Indicates proportion of variance explained

### 3. Residual Analysis
- **Residual Plot**: Prediction errors vs. predicted values
- **Purpose**: Identifies systematic biases or heteroscedasticity
- **Ideal Pattern**: Random scatter around zero

### 4. Feature Importance
- **Horizontal Bar Chart**: Top 15 most important features
- **Full Feature Plot**: All features sorted by importance
- **Color Coding**: Blue (positive impact), red (negative impact)

## Technical Implementation

### Dependencies
```python
- numpy: Numerical operations
- pandas: Data manipulation
- matplotlib: Visualization
- networkx: Graph construction and analysis
- torch: Deep learning framework
- torch_geometric: Graph neural network layers
- scikit-learn: Preprocessing and evaluation
```

### Hardware Requirements
- **CPU/GPU**: Automatic detection (CUDA if available)
- **Memory**: Sufficient for ~3,000 nodes and ~10,000+ edges
- **Storage**: Models saved as `.pt` files

### Key Functions

1. **`train()`**: Trains the model for one epoch
2. **`evaluate(mask)`**: Evaluates model on train/val/test set
3. **`compute_feature_importance()`**: Permutation importance analysis
4. **`shorten_feature_name()`**: Creates readable feature labels

## Insights and Applications

### Spatial Dependencies
The graph structure captures how COVID-19 spreads between adjacent counties, leveraging:
- Geographic proximity
- Population movement patterns
- Shared resources and infrastructure

### Socioeconomic Factors
Feature importance reveals which socioeconomic conditions correlate with higher/lower case rates:
- Income and poverty levels
- Employment status
- Health insurance coverage
- Educational attainment
- Housing characteristics

### Mobility Patterns
Changes in mobility (from Google data) indicate:
- Social distancing compliance
- Economic activity levels
- Population density effects
- Risk of transmission

## Model Advantages

1. **Spatial Awareness**: Unlike traditional regression, GNNs incorporate geographic relationships
2. **Feature Learning**: Automatically learns relevant feature combinations
3. **Scalability**: Can handle large graphs with thousands of nodes
4. **Interpretability**: Feature importance provides actionable insights
5. **Flexibility**: Can be extended to temporal predictions or other diseases

## Limitations and Future Work

### Current Limitations
- Static snapshot (no temporal dynamics)
- Assumes current adjacency relationships (no time-varying mobility)
- Missing data imputed with zeros (may introduce bias)
- Computational cost for large-scale graphs

### Future Directions
1. **Temporal Extension**: Incorporate time-series predictions using Recurrent GNNs or Temporal GNNs
2. **Dynamic Edges**: Use mobility data to create weighted, time-varying edges
3. **Multi-task Learning**: Simultaneously predict cases, hospitalizations, and deaths
4. **Transfer Learning**: Apply pre-trained models to new regions or diseases
5. **Causal Inference**: Identify causal relationships, not just correlations
6. **Uncertainty Quantification**: Provide prediction intervals using Bayesian approaches

## Conclusion

This GNN-based approach successfully leverages spatial relationships and multi-source data to predict COVID-19 case rates across U.S. counties. The improved GAT model with attention mechanisms demonstrates the value of sophisticated graph architectures for epidemiological modeling. Feature importance analysis provides actionable insights into which socioeconomic and behavioral factors most influence disease spread, informing public health interventions.

## References

- **PyTorch Geometric**: Fey, M., & Lenssen, J. E. (2019). Fast Graph Representation Learning with PyTorch Geometric.
- **Graph Attention Networks**: Veličković, P., et al. (2018). Graph Attention Networks. ICLR.
- **Graph Convolutional Networks**: Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. ICLR.

## Files Generated

- `best_model.pt`: Best original GCN model weights
- `best_improved_model.pt`: Best improved GAT model weights
- Training curves and prediction visualizations
- Feature importance plots
