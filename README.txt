================================================================================
Socioeconomic Determinants and Mobility Patterns in COVID-19 Risk:
A Spatial Network Analysis of the U.S. Counties
================================================================================

PACKAGE DESCRIPTION
-------------------

This package implements spatial epidemic models to analyze how socioeconomic 
characteristics and mobility patterns influence COVID-19 transmission dynamics 
across U.S. counties. The package includes:

1. Baseline Spatial SIR Model: A mechanistic model with constant transmission 
   rates and spatial diffusion via county adjacency.

2. Extended Socio-mobility SIR Model: Extends the baseline by allowing 
   county-specific transmission rates parameterized from 18 socioeconomic 
   features.

3. Graph Neural Network (GNN) Model: A data-driven approach that learns 
   non-linear relationships between socioeconomic/mobility factors and disease 
   burden.

The models are designed to address the problem of attributing socioeconomic 
and mobility factors to COVID-19 transmission by quantifying how these 
structural determinants influence disease spread.

INSTALLATION
------------

1. Prerequisites:
   - Python 3.7 or higher
   - pip package manager

2. Install required packages:
   
   pip install numpy pandas scipy networkx scikit-learn matplotlib seaborn
   pip install torch torch-geometric  # For GNN model (if implemented)

3. Data Setup:
   - Place data files in the Data/ directory:
     * county_adjacency2024.txt
     * United_States_COVID-19_Community_Levels_by_County_20251102.csv
     * socioeconomic_data.csv

USAGE
-----

1. Baseline Spatial SIR Model:
   
   Open and run the Jupyter notebook:
   spatial_sir_baseline_models.ipynb
   
   The notebook contains:
   - Data loading and preprocessing
   - Model implementation (simulate_spatial_sir function)
   - Baseline model calibration
   - Performance evaluation
   
   See SRC/BASELINE_SIR_DOCUMENTATION.md for detailed documentation.

2. Extended Socio-mobility SIR Model:
   
   Continue in the same notebook (spatial_sir_baseline_models.ipynb):
   - Extended model calibration
   - Feature weight analysis
   - Comparison with baseline
   
   See SRC/EXTENDED_SIR_DOCUMENTATION.md for detailed documentation.

3. Running a Demo:
   
   To run a quick demonstration:
   
   a. Open spatial_sir_baseline_models.ipynb in Jupyter
   b. Execute all cells sequentially
   c. The notebook will:
      - Load and preprocess data
      - Build spatial adjacency network
      - Calibrate baseline model
      - Calibrate extended model
      - Generate performance metrics and visualizations
   
   Expected runtime: ~5-10 minutes depending on data size and optimization 
   iterations.

DIRECTORY STRUCTURE
------------------

model/
├── README.txt                    # This file
├── Data/                         # Data directory
│   ├── county_adjacency2024.txt
│   ├── United_States_COVID-19_Community_Levels_by_County_20251102.csv
│   └── socioeconomic_data.csv
├── SRC/                          # Source code directory
│   ├── spatial_sir_baseline_models.ipynb  # Main implementation notebook
│   ├── BASELINE_SIR_DOCUMENTATION.md      # Baseline model docs
│   └── EXTENDED_SIR_DOCUMENTATION.md      # Extended model docs
└── DOC/                          # Documentation directory
    ├── final_report.pdf          # Final project report
    └── poster.pdf                # Presentation poster

KEY FUNCTIONS
-------------

1. simulate_spatial_sir(beta_vec, gamma, m, neighbor_lists, degrees_arr, 
                        S0, I0, R0, N, T=60, dt=1.0)
   
   Simulates spatial SIR model with degree-normalized diffusion.
   Returns final S, I, R populations.

2. loss_baseline(params)
   
   Loss function for baseline model calibration.
   Returns MAE between predicted and observed disease burden.

3. loss_extended(params, X_features, gamma, m)
   
   Loss function for extended model calibration.
   Returns MAE between predicted and observed disease burden.

MODEL PARAMETERS
----------------

Baseline Model:
- beta: Transmission rate [0.001, 0.1]
- gamma: Recovery rate [1/21, 1/3] days^-1 (3-21 day infectious period)
- m: Spatial diffusion [0, 0.2]

Extended Model:
- alpha_0, alpha_1, ..., alpha_18: Feature weights and intercept
- gamma: Recovery rate [1/21, 1/3] days^-1
- m: Spatial diffusion [0, 0.2]

OUTPUT
------

The models generate:
- Calibrated parameter values
- Predicted disease burden (cases per 100,000 population)
- Performance metrics (MAE, RMSE, R²)
- Visualization plots (if implemented in notebook)

TROUBLESHOOTING
---------------

1. Import errors:
   - Ensure all required packages are installed
   - Check Python version (3.7+)

2. Data loading errors:
   - Verify data files are in Data/ directory
   - Check file names match exactly (case-sensitive)

3. Optimization convergence issues:
   - Try different initial parameter guesses
   - Increase maxiter in optimization options
   - Check parameter bounds are reasonable

4. Memory issues:
   - Reduce number of counties in analysis
   - Use smaller time horizon (T parameter)

CONTACT
-------

For questions or issues, please refer to the main project report in DOC/ 
directory or contact the project authors.

LICENSE
-------

This software is provided for research purposes. See project documentation 
for details.

================================================================================

