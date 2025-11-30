# Baseline Spatial SIR Model Documentation

## Overview

The Baseline Spatial SIR Model implements a spatially-structured Susceptible-Infected-Recovered (SIR) epidemic model for COVID-19 transmission across U.S. counties. This model serves as a baseline that incorporates only geographic adjacency and reported case data, without socioeconomic or mobility covariates.

## Model Description

### Mathematical Framework

The model represents each county $i$ as a node in a weighted, undirected network $G=(V,E)$, where edges encode spatial adjacency between counties. The dynamics are governed by:

$$
\frac{dS_i}{dt} = -\beta_i S_i \sum_{j \in \mathcal{N}(i)} w_{ij} \frac{I_j}{N_j} + m \sum_{j \in \mathcal{N}(i)} w_{ij} \left(\frac{S_j}{N_j} - \frac{S_i}{N_i}\right)
$$

$$
\frac{dI_i}{dt} = \beta_i S_i \sum_{j \in \mathcal{N}(i)} w_{ij} \frac{I_j}{N_j} - \gamma I_i + m \sum_{j \in \mathcal{N}(i)} w_{ij} \left(\frac{I_j}{N_j} - \frac{I_i}{N_i}\right)
$$

$$
\frac{dR_i}{dt} = \gamma I_i + m \sum_{j \in \mathcal{N}(i)} w_{ij} \left(\frac{R_j}{N_j} - \frac{R_i}{N_i}\right)
$$

where:
- $S_i, I_i, R_i$: Susceptible, infected, and recovered populations in county $i$
- $N_i$: Total population of county $i$
- $\mathcal{N}(i)$: Set of neighboring counties
- $w_{ij}$: Spatial connectivity weights (normalized by node degree for diffusion terms)
- $\beta_i$: County-specific transmission rate (constant $\beta$ for baseline model)
- $\gamma$: Recovery rate (constant across all counties)
- $m$: Spatial diffusion parameter controlling cross-county movement

### Key Features

- **Constant transmission rate**: $\beta_i = \beta$ for all counties $i$
- **Spatial diffusion**: Incorporates county adjacency relationships through degree-normalized diffusion terms
- **No socioeconomic covariates**: Baseline model uses only geographic and epidemiological data

## Implementation

### Core Function: `simulate_spatial_sir`

```python
def simulate_spatial_sir(beta_vec, gamma, m, neighbor_lists, degrees_arr, 
                         S0, I0, R0, N, T=60, dt=1.0):
    """
    Simulate spatial SIR model with degree-normalized diffusion.
    
    Parameters:
    -----------
    beta_vec : array-like
        Transmission rate per county (array of length n_counties)
    gamma : float
        Recovery rate (scalar, constant across counties)
    m : float
        Spatial diffusion coefficient
    neighbor_lists : list of lists
        List of neighbor indices per county (neighbor_lists[i] contains indices of neighbors of county i)
    degrees_arr : array-like
        Degree of each county (number of neighbors)
    S0, I0, R0 : array-like
        Initial conditions for susceptible, infected, and recovered populations
    N : array-like
        Total population per county
    T : int, default=60
        Number of simulation time steps
    dt : float, default=1.0
        Time step size
    
    Returns:
    --------
    S, I, R : tuple of arrays
        Final susceptible, infected, and recovered populations after T time steps
    """
```

### Calibration Function: `loss_baseline`

```python
def loss_baseline(params):
    """
    Loss function for baseline model calibration.
    
    Parameters:
    -----------
    params : array-like
        [beta, gamma, m] - transmission rate, recovery rate, spatial diffusion
    
    Returns:
    --------
    mae : float
        Mean Absolute Error between predicted and observed disease burden
    """
```

### Parameter Bounds

- $\beta$: [0.001, 0.1] - Transmission rate bounds
- $\gamma$: [1/21, 1/3] days$^{-1}$ - Recovery rate (corresponds to infectious periods of 3-21 days)
- $m$: [0, 0.2] - Spatial diffusion parameter

### Optimization

The model is calibrated using L-BFGS-B optimization to minimize Mean Absolute Error (MAE) between predicted and observed disease burden (cases per 100,000 population).

## Usage Example

```python
# Load data and prepare initial conditions
# ... (data loading code) ...

# Set initial conditions
I0 = np.clip((y / 1e5) * N * 0.01, 1.0, N * 0.02)
S0 = N - I0
R0 = np.zeros_like(N)

# Calibrate baseline model
from scipy.optimize import minimize

bounds = [(0.001, 0.1), (1/21, 1/3), (0, 0.2)]
x0 = [0.01, 1/7, 0.02]  # Initial guess: β=0.01, γ=1/7 (7 days), m=0.02

result = minimize(loss_baseline, x0, method='L-BFGS-B', bounds=bounds, 
                  options={'maxiter': 100, 'disp': False})

beta_baseline, gamma_baseline, m_baseline = result.x
beta_vec_baseline = np.ones(len(fips_list)) * beta_baseline

# Simulate model
_, I_baseline, _ = simulate_spatial_sir(beta_vec_baseline, gamma_baseline, m_baseline,
                                        neighbor_lists, degrees_arr, S0, I0, R0, N, T=60)

# Evaluate performance
N_safe = np.where(N > 0, N, 1.0)
y_baseline = (I_baseline / N_safe) * 1e5
mae_baseline = mean_absolute_error(y, y_baseline)
```

## Expected Performance

Based on calibration results:
- **MAE**: ~1.26 cases per 100,000
- **RMSE**: ~7.65 cases per 100,000
- **R²**: ~-0.028 (negative R² indicates worse performance than mean predictor)

## Limitations

1. **Homogeneous transmission**: Constant $\beta$ across all counties fails to capture socioeconomic heterogeneity
2. **Simple spatial structure**: Degree-normalized diffusion may not reflect true mobility patterns
3. **No feature integration**: Does not incorporate socioeconomic or mobility covariates

## Files

- **Notebook**: `spatial_sir_baseline_models.ipynb`
- **Main implementation**: Cells 4-5 in the notebook
- **Data requirements**: 
  - County adjacency data (`Data/county_adjacency2024.txt`)
  - COVID-19 case data (`Data/United_States_COVID-19_Community_Levels_by_County_20251102.csv`)
  - Population data (from COVID dataset)

## Dependencies

- numpy
- pandas
- scipy
- networkx
- scikit-learn
- matplotlib
- seaborn

## References

See main report (`final_report.pdf`) Section 4.1 for detailed model description and Section 5.1 for results.

