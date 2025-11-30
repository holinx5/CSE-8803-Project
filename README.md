# Socioeconomic Determinants and Mobility Patterns in COVID-19 Risk: A Spatial Network Analysis

[![GitHub Pages](https://img.shields.io/badge/GitHub-Pages-blue)](https://holinx5.github.io/CSE-8803-Project)
[![License](https://img.shields.io/badge/License-Research-purple)](LICENSE)

## Project Overview

This project addresses the problem of **attributing socioeconomic and mobility factors to COVID-19 transmission** by quantifying how these structural determinants influence disease spread across U.S. counties. We employ a complementary two-model framework combining mechanistic SIR models and Graph Neural Networks to provide both causal and predictive attribution of transmission risk.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Methods](#methods)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Documentation](#documentation)
- [Authors](#authors)
- [License](#license)

## Overview

This study develops spatial epidemic models to analyze how socioeconomic characteristics and mobility patterns influence COVID-19 transmission dynamics across 3,222 U.S. counties. The project combines:

1. **Baseline Spatial SIR Model**: Mechanistic model with constant transmission rates and spatial diffusion
2. **Extended Socio-mobility SIR Model**: County-specific transmission rates parameterized from 18 socioeconomic features
3. **Graph Neural Network (GNN)**: Data-driven approach learning non-linear relationships between factors and disease burden

## Key Findings

- **GNN achieves the only positive explanatory power** (R² = 0.441) among all models
- **Top influential factors**: Median household income, residential mobility, single-parent households, mortgage cost, overcrowding, education level, and workplace mobility
- **Mechanistic models reveal limitations**: Linear parameterization fails to capture complex non-linear interactions
- **Dual attribution framework**: SIR models provide causal attribution (how factors influence transmission rates), while GNN provides predictive attribution (which factors are most influential)

### Model Performance Comparison

| Model | MAE (cases/100k) | RMSE (cases/100k) | R² |
|-------|------------------|-------------------|-----|
| Baseline Spatial SIR | 1.26 | 7.65 | -0.028 |
| Extended Socio-mobility SIR | 13.53 | 17.88 | -4.609 |
| **Graph Neural Network** | **27.48** | **39.59** | **0.441** |

## Methods

### Data Sources
- **COVID-19 outcomes**: CDC Community Levels dataset
- **Socioeconomic indicators**: U.S. Census Bureau (18 features including income, education, poverty, housing, etc.)
- **Mobility data**: Google COVID-19 Community Mobility Reports
- **Spatial structure**: County adjacency network

### Models

1. **Baseline Spatial SIR**: Constant transmission rate β with degree-normalized spatial diffusion
2. **Extended SIR**: County-specific $\beta_i = \exp(\alpha_0 + \sum \alpha_j \cdot X_{ij})$ parameterized from socioeconomic features
3. **GNN**: Graph neural network with counties as nodes, adjacency as edges, learning non-linear feature interactions

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/holinx5/CSE-8803-Project.git
cd CSE-8803-Project

# Install required packages
pip install numpy pandas scipy networkx scikit-learn matplotlib seaborn
pip install torch torch-geometric  # For GNN model (if implemented)
```

### Data Setup
Place data files in the `Data/` directory:
- `county_adjacency2024.txt`
- `United_States_COVID-19_Community_Levels_by_County_20251102.csv`
- `socioeconomic_data.csv`

## Usage

### Running the Models

1. **Baseline Spatial SIR Model**:
   ```bash
   jupyter notebook spatial_sir_baseline_models.ipynb
   ```
   Execute all cells to run baseline model calibration and evaluation.

2. **Extended Socio-mobility Model**:
   Continue in the same notebook to run the extended model.

3. **GNN Model**:
      ```bash
   jupyter notebook GNN.ipynb
   ```


## Project Structure

```
├── README.md                         # This file
├── Data Compilation/                 # Raw Data
│   ├── covid/
│   │   ├── United_States_COVID-19_Community_Levels_by_County_20251102.csv
│   │   └── data dictionary.png
│   ├── mobility/
│   │   ├── 2020_US_Region_Mobility_Report.csv
│   │   ├── 2021_US_Region_Mobility_Report.csv
│   │   └── 2022_US_Region_Mobility_Report.csv
│   ├── socioeconomic/
│   │   ├── B19057/
│   │   │   ├── ACSDT5Y2023.B19057-Column-Metadata.csv
│   │   │   ├── ACSDT5Y2023.B19057-Data.csv
│   │   │   └── ACSDT5Y2023.B19057-Table-Notes.txt
│   │   ├── B19083/
│   │   │   ├── ACSDT5Y2023.B19083-Column-Metadata.csv
│   │   │   ├── ACSDT5Y2023.B19083-Data.csv
│   │   │   └── ACSDT5Y2023.B19083-Table-Notes.txt
│   │   ├── DP02/
│   │   │   ├── ACSDP5Y2023.DP02-Column-Metadata.csv
│   │   │   ├── ACSDP5Y2023.DP02-Data.csv
│   │   │   └── ACSDP5Y2023.DP02-Table-Notes.txt
│   │   ├── DP04/
│   │   │   ├── ACSDP5Y2023.DP04-Column-Metadata.csv
│   │   │   ├── ACSDP5Y2023.DP04-Data.csv
│   │   │   └── ACSDP5Y2023.DP04-Table-Notes.txt
│   │   ├── DP05/
│   │   │   ├── ACSDP5Y2023.DP05-Column-Metadata.csv
│   │   │   ├── ACSDP5Y2023.DP05-Data.csv
│   │   │   └── ACSDP5Y2023.DP05-Table-Notes.txt
│   │   ├── S1501/
│   │   │   ├── ACSST5Y2023.S1501-Column-Metadata.csv
│   │   │   ├── ACSST5Y2023.S1501-Data.csv
│   │   │   └── ACSST5Y2023.S1501-Table-Notes.txt
│   │   ├── S1702/
│   │   │   ├── ACSST5Y2023.S1702-Column-Metadata.csv
│   │   │   ├── ACSST5Y2023.S1702-Data.csv
│   │   │   └── ACSST5Y2023.S1702-Table-Notes.txt
│   │   ├── S1903/
│   │   │   ├── ACSST5Y2023.S1903-Column-Metadata.csv
│   │   │   ├── ACSST5Y2023.S1903-Data.csv
│   │   │   └── ACSST5Y2023.S1903-Table-Notes.txt
│   │   ├── S2301/
│   │   │   ├── ACSST5Y2023.S2301-Column-Metadata.csv
│   │   │   ├── ACSST5Y2023.S2301-Data.csv
│   │   │   └── ACSST5Y2023.S2301-Table-Notes.txt
│   │   ├── S2701/
│   │   │   ├── ACSST5Y2023.S2701-Column-Metadata.csv
│   │   │   ├── ACSST5Y2023.S2701-Data.csv
│   │   │   └── ACSST5Y2023.S2701-Table-Notes.txt
│   │   ├── S2801/
│   │   │   ├── ACSST5Y2023.S2801-Column-Metadata.csv
│   │   │   ├── ACSST5Y2023.S2801-Data.csv
│   │   │   └── ACSST5Y2023.S2801-Table-Notes.txt
│   │   ├── .Rhistory
│   │   ├── Variable Dictionary.docx
│   │   ├── data_compilation.R
│   │   ├── socioeconomic.Rproj
│   │   └── socioeconomic_data.csv
│   └── county_adjacency2024.txt
├── Data/                             # Data directory
│   ├── county_adjacency2024.txt
│   ├── United_States_COVID-19_Community_Levels_by_County_20251102.csv
│   └── socioeconomic_data.csv
├── SRC/                              # Source code
│   ├── spatial_sir_baseline_models.ipynb
|   ├── GNN.ipynb
│   ├── BASELINE_SIR_DOCUMENTATION.md
│   └── EXTENDED_SIR_DOCUMENTATION.md
└── DOC/                              # Documentation
    ├── final_report.pdf
    └── poster.pdf

```

## Results

### Key Contributions

1. **Comprehensive Attribution Framework**: Combines mechanistic understanding (SIR) with predictive identification (GNN)
2. **Identified Top Risk Factors**: Income, education, housing, and mobility patterns emerge as strongest predictors
3. **Spatial Insights**: Reveals how neighboring counties influence each other's transmission risk
4. **Methodological Advances**: Demonstrates limitations of linear parameterization and need for flexible modeling

### Detailed Results

See the [final report](DOC/final_report.pdf) for complete results, analysis, and discussion.

## Documentation

- **Model Documentation**: 
  - [Baseline SIR Model](SRC/BASELINE_SIR_DOCUMENTATION.md)
  - [Extended SIR Model](SRC/EXTENDED_SIR_DOCUMENTATION.md)
- **Final Report**: [DOC/final_report.pdf](DOC/final_report.pdf)
- **Poster**: [DOC/poster.pdf](DOC/poster.pdf)

## Authors

- **Holin Xue** - hxue49@gatech.edu
- **Priscilla Zhang** - zzhang3100@gatech.edu

**Institution**: Georgia Institute of Technology, H. Milton Stewart School of Industrial and Systems Engineering  
**Course**: CSE 8803: Data Science for Epidemiology

## Citation

If you use this work, please cite:

```bibtex
@article{Xue,Zhang2025socioeconomic,
  title={Socioeconomic Determinants and Mobility Patterns in COVID-19 Risk: A Spatial Network Analysis of the U.S. Counties},
  author={Xue, Holin and Zhang, Priscilla},
  journal={CSE 8803: Data Science for Epidemiology},
  year={2025},
  institution={Georgia Institute of Technology}
}
```

## License

This project is provided for research purposes. See project documentation for details.

## Acknowledgments

- Data sources: CDC, U.S. Census Bureau, Google COVID-19 Community Mobility Reports
- Course instructors and TAs for valuable feedback

---

**Note**: This is a course project for CSE 8803: Data Science for Epidemiology at Georgia Institute of Technology.

