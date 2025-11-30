# Extended Socio-mobility SIR Model Documentation

## Overview

The Extended Socio-mobility SIR Model augments the baseline spatial SIR model by allowing county-specific transmission rates $\beta_i$ to vary as a function of socioeconomic and mobility covariates. This model tests whether incorporating socioeconomic heterogeneity improves predictive performance compared to the baseline.

## Model Description

### Mathematical Framework

The model uses the same spatial SIR framework as the baseline, but parameterizes the transmission rate as:

$$
\beta_i = \exp\left(\alpha_0 + \sum_{j=1}^{p} \alpha_j \cdot X_{ij}\right)
$$

where:
- $X_{ij}$: The $j$-th standardized socioeconomic feature for county $i$
- $\alpha_0$: Intercept term
- $\alpha_j$: Feature-specific coefficients ($p=18$ socioeconomic features)
- The exponential transformation ensures $\beta_i > 0$ for all counties
- Resulting values are clipped to $[0.001, 0.1]$ to maintain realistic transmission rates

### Key Features

- **County-specific transmission rates**: $\beta_i$ varies across counties based on socioeconomic features
- **Linear parameterization**: Transmission rate is a linear combination of features in log-space
- **18 socioeconomic features**: Includes income, education, poverty, housing conditions, demographic composition, etc.
- **Spatial diffusion**: Same degree-normalized spatial diffusion as baseline model

### Socioeconomic Features

The model incorporates 18 standardized socioeconomic indicators:
1. Percent population aged 25+ without High School diploma
2. Median household income
3. Income disparity (Gini Index)
4. Median home value
5. Median gross rent
6. Median monthly mortgage
7. Percent of owner-occupied housing units
8. Percent unemployed
9. Percent of families below poverty level
10. Percent of single-parent households
11. Percent of households without motor vehicle
12. Percent of households with overcrowding (>1 person per room)
13. Percent non-white
14. Percent of vacant housing units
15. Percent of households with rent >35% of income
16. Percent of households receiving public assistance
17. Percent of individuals without health insurance
18. Percent of households without internet access

## Implementation

### Core Function: `loss_extended`

```python
def loss_extended(params, X_features, gamma, m):
    """
    Loss function for extended model calibration.
    
    Parameters:
    -----------
    params : array-like
        [alpha_0, alpha_1, ..., alpha_p, gamma, m]
        - alpha_0: intercept
        - alpha_1 to alpha_p: feature weights (p=18 features)
        - gamma: recovery rate
        - m: spatial diffusion parameter
    X_features : array-like, shape (n_counties, n_features)
        Standardized socioeconomic features for each county
    gamma : float
        Recovery rate (can be fixed or optimized)
    m : float
        Spatial diffusion parameter (can be fixed or optimized)
    
    Returns:
    --------
    mae : float
        Mean Absolute Error between predicted and observed disease burden
    """
    w = params[:-1]  # Feature weights (including intercept)
    b = params[-1]   # Intercept (if not included in w)
    
    # Compute β_i from features
    beta_logit = X_features @ w + b
    beta_vec = np.clip(np.exp(beta_logit), 0.001, 0.1)
    
    # Simulate and compute loss
    # ... (simulation code) ...
    
    return mae
```

### Parameter Bounds

- $\alpha_0, \alpha_1, \ldots, \alpha_p$: Unbounded (typically $[-10, 10]$ in practice)
- $\gamma$: [1/21, 1/3] days$^{-1}$ (same as baseline)
- $m$: [0, 0.2] (same as baseline)

### Optimization

The model is calibrated using L-BFGS-B optimization to minimize MAE. The optimization simultaneously adjusts all feature weights, intercept, recovery rate, and spatial diffusion parameter.

## Usage Example

```python
# Load and standardize features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_features = scaler.fit_transform(socio_df[feature_cols])

# Prepare initial conditions
I0 = np.clip((y / 1e5) * N * 0.01, 1.0, N * 0.02)
S0 = N - I0
R0 = np.zeros_like(N)

# Calibrate extended model
n_features = X_features.shape[1]
bounds = [(-10, 10)] * (n_features + 1) + [(1/21, 1/3), (0, 0.2)]  # weights + intercept + gamma + m
x0 = np.zeros(n_features + 1) + [1/7, 0.02]  # Initial guess

result = minimize(loss_extended, x0, args=(X_features, None, None), 
                  method='L-BFGS-B', bounds=bounds, 
                  options={'maxiter': 100, 'disp': False})

# Extract parameters
alpha_params = result.x[:-2]  # Feature weights
gamma_extended = result.x[-2]
m_extended = result.x[-1]

# Compute county-specific beta values
beta_logit = X_features @ alpha_params[:-1] + alpha_params[-1]
beta_vec_extended = np.clip(np.exp(beta_logit), 0.001, 0.1)

# Simulate model
_, I_extended, _ = simulate_spatial_sir(beta_vec_extended, gamma_extended, m_extended,
                                        neighbor_lists, degrees_arr, S0, I0, R0, N, T=60)

# Evaluate performance
y_extended = (I_extended / N_safe) * 1e5
mae_extended = mean_absolute_error(y, y_extended)
```

## Expected Performance

Based on calibration results:
- **MAE**: ~13.53 cases per 100,000
- **RMSE**: ~17.88 cases per 100,000
- **R²**: ~-4.609 (negative R² indicates worse performance than mean predictor)

**Note**: Despite theoretical motivation, the extended model performs worse than the baseline, suggesting limitations of linear parameterization.

## Limitations

1. **Linear parameterization**: The exponential-linear form may not capture complex non-linear interactions between features
2. **Parameter convergence**: All $\beta_i$ values often converge to the upper bound (0.1), eliminating county heterogeneity
3. **Feature interactions**: Cannot model interactions between socioeconomic factors
4. **Counterintuitive coefficients**: Negative weights for traditionally disadvantaged indicators (unemployment, poverty) suggest model misspecification

## Key Findings

1. **Performance degradation**: Extended model performs substantially worse than baseline (MAE ~10.7x higher)
2. **Parameter convergence issues**: All county-specific $\beta_i$ values converge to upper bound despite county-specific feature inputs
3. **Linear limitations**: Simple linear parameterization fails to capture true relationship between socioeconomic factors and transmission rates
4. **Motivation for GNN**: These limitations motivate the use of Graph Neural Networks, which can learn non-linear feature interactions

## Files

- **Notebook**: `spatial_sir_baseline_models.ipynb`
- **Main implementation**: Cells 6-7 in the notebook
- **Data requirements**: 
  - All baseline model data requirements
  - Socioeconomic data (`Data/socioeconomic_data.csv`)

## Dependencies

Same as baseline model, plus:
- Feature standardization (scikit-learn StandardScaler)

## References

See main report (`final_report.pdf`) Section 4.2 for detailed model description and Section 5.1 for results and discussion of limitations.

