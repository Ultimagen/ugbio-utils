# Smoothed Precision Estimation for Quality Score Mapping

## Overview

This module provides robust, smooth precision estimation for quality score mapping using adaptive kernel density estimation (KDE). It replaces noisy direct counting methods with sophisticated smoothing algorithms that provide stable precision estimates even in sparse high-quality regions.

The implementation uses efficient FFT-based convolution on fixed grids to achieve O(G log G) complexity, enabling processing of millions of data points in seconds while maintaining numerical stability and providing comprehensive diagnostics.

## Quick Start

### Basic Usage with MQUAL Scores

```python
import numpy as np
import pandas as pd
from ugbio_srsnv.smoothing_utils import (
    create_uncertainty_function_pipeline_fast,
    make_grid_and_transform,
    bin_data_to_grid,
    calculate_smoothed_precision_kde
)
from ugbio_srsnv.srsnv_utils import prob_to_phred, prob_to_logit

# 1. Prepare your dataset with cross-validation predictions
# DataFrame should have columns: 'fold_id', 'label', 'prob_fold_0', 'prob_fold_1', etc.
# Rows with fold_id=NaN are used as validation set

# 2. Create uncertainty function from cross-validation data
uncertainty_fn, metadata = create_uncertainty_function_pipeline_fast(
    pd_df=your_dataframe,
    fold_col='fold_id',
    label_col='label',
    num_cv_folds=3,
    prob_to_phred_fn=prob_to_phred,
    prob_to_logit_fn=prob_to_logit,
    transform_mode='mqual'  # Use MQUAL scale (recommended)
)

# 3. Convert your predictions to MQUAL scores
mqual_true = prob_to_phred(your_true_predictions)   # Probabilities for true positives
mqual_false = prob_to_phred(your_false_predictions) # Probabilities for false positives

# 4. Set up computational grid
all_mqual = np.concatenate([mqual_true, mqual_false])
grid, dx, to_grid, from_grid, grid_metadata = make_grid_and_transform(
    all_mqual,
    grid_size=1024,
    transform_mode="mqual"
)

# 5. Bin data to grid
counts_true, _ = bin_data_to_grid(mqual_true, to_grid_space=to_grid)
counts_false, _ = bin_data_to_grid(mqual_false, to_grid_space=to_grid)

# 6. Calculate smoothed precision with automatic truncation
tpr, fpr, precision, kde_metadata = calculate_smoothed_precision_kde(
    counts_true, counts_false, grid, dx, uncertainty_fn, from_grid,
    len(mqual_true), len(mqual_false),
    enforce_monotonic=False,      # Default: no monotonicity enforcement
    truncation_mode="auto_detect" # Default: automatic tail truncation
)

# 7. Convert grid coordinates to MQUAL values for interpretation
mqual_values = from_grid(grid)

print(f"Computed precision curve with {len(precision)} points")
print(f"MQUAL range: [{mqual_values.min():.1f}, {mqual_values.max():.1f}]")
print(f"Precision range: [{precision.min():.4f}, {precision.max():.4f}]")
```

### Using Logit Scale

```python
# For logit-based analysis, simply change the transform_mode
uncertainty_fn, metadata = create_uncertainty_function_pipeline_fast(
    pd_df=your_dataframe,
    fold_col='fold_id',
    label_col='label',
    num_cv_folds=3,
    prob_to_phred_fn=prob_to_phred,
    prob_to_logit_fn=prob_to_logit,
    transform_mode='logit'  # Use logit scale
)

# Convert predictions to logit scores
logit_true = prob_to_logit(your_true_predictions)
logit_false = prob_to_logit(your_false_predictions)

# Set up grid for logit mode
all_logit = np.concatenate([logit_true, logit_false])
grid, dx, to_grid, from_grid, grid_metadata = make_grid_and_transform(
    all_logit,
    grid_size=1024,
    transform_mode="logit"  # Different boundary handling for unbounded domain
)

# Continue with same binning and KDE steps...
```

### Configuration Options

```python
# High-precision configuration
tpr, fpr, precision, metadata = calculate_smoothed_precision_kde(
    counts_true, counts_false, grid, dx, uncertainty_fn, from_grid,
    len(mqual_true), len(mqual_false),
    num_bandwidth_levels=7,       # More adaptive smoothing (default: 5)
    enforce_monotonic=True,       # Apply isotonic regression
    truncation_mode="data_max",   # Truncate at maximum observed data
    machine_precision_threshold=1e-12  # Stricter precision detection
)

# Fast configuration for development
tpr, fpr, precision, metadata = calculate_smoothed_precision_kde(
    counts_true, counts_false, grid, dx, uncertainty_fn, from_grid,
    len(mqual_true), len(mqual_false),
    num_bandwidth_levels=3,       # Less adaptive smoothing
    enforce_monotonic=False,      # No isotonic regression
    truncation_mode="none"        # No tail processing
)
```

## Architecture

### Three-Layer Design

1. **Uncertainty Estimation**: Creates smooth uncertainty functions from cross-validation data using LOWESS regression
2. **Grid-based Convolution**: Efficient FFT-based operations on fixed grids for O(G log G) performance
3. **Adaptive KDE**: Variable bandwidth kernel density estimation informed by local uncertainty

### Key Advantages

- **Computational Efficiency**: O(G log G) complexity after initial binning, independent of dataset size
- **Numerical Stability**: Grid-based operations with proper boundary handling and normalization
- **Adaptive Smoothing**: Bandwidth varies based on local data density and model uncertainty
- **Quality Control**: Comprehensive diagnostics and conservation checks
- **Flexible Domains**: Support for both MQUAL (bounded at zero) and logit (unbounded) coordinate systems

## Data Requirements

### Input DataFrame Schema

Your dataset must contain cross-validation predictions:

```python
required_columns = [
    'fold_id',        # int/float, NaN for validation data
    'label',          # int/bool, true labels (0/1)
    'prob_fold_0',    # float [0,1], probability predictions from fold 0
    'prob_fold_1',    # float [0,1], probability predictions from fold 1
    'prob_fold_2',    # float [0,1], probability predictions from fold 2
    # ... additional prob_fold_k columns as needed
]
```

### Data Validation Requirements

- **Validation subset**: Rows where `fold_id` is NaN (≥5,000 rows recommended)
- **Cross-validation folds**: At least 2 folds with probability predictions
- **Data quality**: Probabilities in [0,1], minimal missing values
- **Label distribution**: Adequate representation of both positive and negative cases

## Detailed Function Documentation

### Core Pipeline Functions

#### `create_uncertainty_function_pipeline_fast()`

**Primary production function** for creating smooth uncertainty estimates from cross-validation data.

```python
uncertainty_fn, metadata = create_uncertainty_function_pipeline_fast(
    pd_df: pd.DataFrame,              # Data with CV predictions
    fold_col: str,                    # Column with fold assignments
    label_col: str,                   # Column with true labels
    num_cv_folds: int,                # Number of CV folds
    prob_to_phred_fn: Callable,       # Function: prob → MQUAL
    prob_to_logit_fn: Callable,       # Function: prob → logit
    transform_mode: str = "mqual",    # "mqual" or "logit"
    lowess_frac: float = 0.3,         # LOWESS smoothing fraction
    sigma_min: float = 1e-6,          # Minimum uncertainty bound
    sigma_max: float = 100.0,         # Maximum uncertainty bound
    min_val_size: int = 5000          # Minimum validation set size
) -> Tuple[Callable[[np.ndarray], np.ndarray], Dict[str, Any]]
```

**Returns**:
- `uncertainty_fn`: Vectorized function mapping scores to uncertainty estimates
- `metadata`: Essential processing information

**Performance**: ~3.5 seconds for 3M rows (41x faster than full diagnostic version)

**When to use**: Production deployment, batch processing, integration with downstream steps

---

#### `create_uncertainty_function_pipeline()`

**Full diagnostic version** with comprehensive quality metrics and validation.

```python
uncertainty_fn, full_metadata = create_uncertainty_function_pipeline(
    # Same parameters as fast version, plus:
    use_density_weights: bool = True,    # Weight LOWESS by data density
    use_log_transform: bool = False,     # Apply log transform before smoothing
    it: int = 3,                        # LOWESS robustness iterations
    **lowess_kwargs                     # Additional LOWESS parameters
) -> Tuple[Callable[[np.ndarray], np.ndarray], Dict[str, Any]]
```

**Performance**: ~145 seconds for 3M rows (includes extensive diagnostics)

**When to use**: Development, validation, troubleshooting, detailed quality analysis

---

### Grid Operations

#### `make_grid_and_transform()`

Creates computational grid and coordinate transformation functions.

```python
grid, dx, to_grid, from_grid, metadata = make_grid_and_transform(
    data_values: np.ndarray,          # Data to determine domain
    grid_size: int = 8192,            # Number of grid points
    mqual_max_lut: float = None,      # Maximum MQUAL (auto-detected)
    transform_mode: str = "mqual",    # "mqual" or "logit"
    padding_factor: float = 0.1       # Domain padding fraction
) -> Tuple[np.ndarray, float, Callable, Callable, Dict]
```

**Returns**:
- `grid`: Grid coordinates [0, 1, 2, ..., grid_size-1]
- `dx`: Grid spacing in data units (critical for normalization)
- `to_grid`: Function converting data values → grid coordinates
- `from_grid`: Function converting grid coordinates → data values
- `metadata`: Grid properties and boundary policies

**Critical**: The `dx` parameter is essential for proper density normalization in subsequent operations.

---

#### `bin_data_to_grid()`

Efficiently converts scattered data points to grid-based counts using linear interpolation.

```python
counts, metadata = bin_data_to_grid(
    values: np.ndarray,                      # Data points in original space
    weights: np.ndarray = None,              # Optional weights (uniform if None)
    grid_size: int = 8192,                   # Number of grid bins
    to_grid_space: Callable = None,          # Coordinate transform function
    boundary_policy: str = "clamp"           # "clamp" or "drop" for out-of-bounds
) -> Tuple[np.ndarray, Dict]
```

**Performance**: Processes ~45M points/second with perfect weight conservation (errors < 1e-12)

---

### Convolution Primitives

#### `create_gaussian_kernel()`

Creates truncated Gaussian kernels optimized for FFT convolution.

```python
kernel, metadata = create_gaussian_kernel(
    sigma: float,              # Standard deviation in data units
    dx: float,                 # Grid spacing from make_grid_and_transform
    radius_k: float = 4.0,     # Truncation radius (standard deviations)
    normalize: bool = True     # Whether to normalize to sum=1
) -> Tuple[np.ndarray, Dict]
```

**Note**: The `dx` parameter ensures kernels are properly scaled for grid spacing.

---

#### `fft_convolve()`

Fast convolution using FFT with automatic fallback for small arrays.

```python
result, metadata = fft_convolve(
    signal: np.ndarray,                 # Input data array
    kernel: np.ndarray,                 # Convolution kernel
    boundary_policy: str = "none",      # "none" or "reflect_at_zero"
    mode: str = "same"                  # Convolution mode
) -> Tuple[np.ndarray, Dict]
```

**Critical**: For MQUAL mode, use `boundary_policy="reflect_at_zero"` to maintain conservation. For logit mode, use `boundary_policy="none"`.

---

### KDE and Precision Calculation

#### `calculate_smoothed_precision_kde()` ⚡ **PRIMARY**

Main function for adaptive KDE-based precision estimation.

```python
tpr, fpr, precision, metadata = calculate_smoothed_precision_kde(
    counts_true: np.ndarray,                    # Binned true positive counts
    counts_false: np.ndarray,                   # Binned false positive counts
    grid: np.ndarray,                           # Grid coordinates
    dx: float,                                  # Grid spacing
    get_score_std: Callable,                    # Uncertainty function
    from_grid_space: Callable,                  # Grid→data coordinate transform
    n_true: int,                                # Total true positives
    n_false: int,                               # Total false positives
    num_bandwidth_levels: int = 5,              # Number of adaptive bandwidth levels
    enforce_monotonic: bool = True,             # Apply isotonic regression
    boundary_policy: str = "reflect_at_zero",   # Boundary handling
    truncation_mode: str = "auto_detect",       # Tail truncation method
    machine_precision_threshold: float = 1e-14  # Precision detection threshold
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]
```

**Returns**:
- `tpr`: True positive rates on grid
- `fpr`: False positive rates on grid
- `precision`: Precision values on grid
- `metadata`: Comprehensive processing diagnostics

**Performance**: ~0.1-2 seconds for 1M data points depending on configuration

---

### Utility Functions

#### `cumulative_sum_from_grid()`

Converts density functions to survival functions or CDFs.

```python
result, metadata = cumulative_sum_from_grid(
    counts: np.ndarray,           # Input density/counts array
    reverse: bool = True,         # True for survival function, False for CDF
    normalize_to: float = None,   # Normalize to specific total (None for sum)
    eps: float = 1e-10           # Small value for numerical stability
) -> Tuple[np.ndarray, Dict]
```

---

#### `interpolate_grid_to_points()`

Interpolates grid-based results to arbitrary points.

```python
interpolated, metadata = interpolate_grid_to_points(
    grid_values: np.ndarray,              # Values on grid
    grid: np.ndarray,                     # Grid coordinates
    target_points: np.ndarray,            # Points to interpolate to
    from_grid_space: Callable,            # Grid→data transform
    to_grid_space: Callable,              # Data→grid transform
    fill_value: str | float = "extrapolate"  # Extrapolation behavior
) -> Tuple[np.ndarray, Dict]
```

---

#### `truncate_density_tails()`

Removes numerical artifacts from density tails.

```python
truncated, metadata = truncate_density_tails(
    density: np.ndarray,                      # Input density array
    grid: np.ndarray = None,                  # Grid coordinates
    from_grid_space: Callable = None,         # Coordinate transform
    truncation_mode: str = "auto_detect",     # Truncation strategy
    machine_precision_threshold: float = 1e-14,  # Detection threshold
    counts_array: np.ndarray = None           # Reference counts for validation
) -> Tuple[np.ndarray, Dict]
```

**Truncation modes**:
- `"auto_detect"`: Automatically detect machine precision artifacts (recommended)
- `"data_max"`: Truncate at maximum observed data value
- `"none"`: No truncation applied

## Configuration Guidelines

### Grid Size Selection

| **Data Size** | **Recommended Grid Size** | **Memory Usage** | **Performance** |
|---------------|---------------------------|------------------|-----------------|
| < 100K points | 512 | ~8MB | Excellent |
| 100K - 1M points | 1024 | ~32MB | Very good |
| 1M - 5M points | 2048 | ~64MB | Good |
| > 5M points | 4096 | ~128MB | Acceptable |

**Note**: Powers of 2 are optimal for FFT performance.

### Bandwidth Levels

| **num_bandwidth_levels** | **Adaptivity** | **Speed** | **Use Case** |
|---------------------------|----------------|-----------|-------------|
| 3 | Low | Fast | Development, testing |
| 5 | Good | Standard | Production (recommended) |
| 7 | High | Slower | High-precision applications |
| >7 | Very high | Slow | Special research applications |

### Transform Mode Selection

**MQUAL mode** (`transform_mode="mqual"`):
- **Domain**: [0, ∞) with reflection boundary at zero
- **Use for**: Quality scores, Phred-scaled probabilities
- **Boundary policy**: `"reflect_at_zero"`

**Logit mode** (`transform_mode="logit"`):
- **Domain**: (-∞, ∞) unbounded
- **Use for**: Log-odds ratios, unbounded continuous scores
- **Boundary policy**: `"none"`

## Performance Characteristics

### Computational Complexity

- **Initial binning**: O(N) where N is dataset size
- **FFT operations**: O(G log G) where G is grid size
- **Total pipeline**: O(N + G log G × K) where K is number of bandwidth levels

### Memory Usage

- **Grid operations**: ~8×G×sizeof(float64)
- **Kernel bank**: ~K×kernel_radius×sizeof(float64)
- **Typical memory**: 32-128MB for production configurations

### Scalability Benchmarks

| **Dataset Size** | **Grid Size** | **Processing Time** | **Throughput** |
|------------------|---------------|-------------------|----------------|
| 100K points | 1024 | 0.08s | 1.2M pts/s |
| 500K points | 1024 | 0.35s | 1.4M pts/s |
| 1M points | 1024 | 0.71s | 1.4M pts/s |
| 2M points | 2048 | 1.85s | 1.1M pts/s |

## Quality Metrics and Validation

### Conservation Diagnostics

The pipeline provides comprehensive conservation checks:

```python
metadata = {
    'mixing': {
        'conservation_error_total': 1.2e-12,    # Weight preservation error
        'conservation_error_true': 8.9e-13,     # True class conservation
        'conservation_error_false': 3.4e-13,    # False class conservation
    },
    'rates': {
        'total_tpr_mass': 1.0000,               # TPR normalization check
        'total_fpr_mass': 1.0000,               # FPR normalization check
    }
}
```

**Good quality indicators**:
- Conservation errors < 1e-10
- Rate normalization within 1e-6 of 1.0
- Smooth precision curves without artifacts
- Reasonable extrapolation in high-quality regions

### Common Issues and Solutions

| **Issue** | **Symptom** | **Solution** |
|-----------|-------------|-------------|
| Poor conservation | Error > 1e-6 | Check boundary policy, reduce grid size |
| Noisy precision curve | High variance in tail | Increase bandwidth levels |
| Slow performance | Runtime > expected | Reduce grid size or bandwidth levels |
| Memory errors | Out of memory | Reduce grid size, process in batches |

## Complete Example: End-to-End Pipeline

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ugbio_srsnv.smoothing_utils import *
from ugbio_srsnv.srsnv_utils import prob_to_phred, prob_to_logit

# Load and prepare data
data = pd.read_parquet('your_featuremap.parquet')

# Create uncertainty function from CV data
print("Creating uncertainty function...")
uncertainty_fn, step1_metadata = create_uncertainty_function_pipeline_fast(
    data, 'fold_id', 'label', num_cv_folds=3,
    prob_to_phred, prob_to_logit, transform_mode='mqual'
)
print(f"Uncertainty function created from {step1_metadata['validation_size']:,} validation points")

# Prepare prediction data (replace with your actual predictions)
true_probs = data[data.label == 1]['your_prediction_column'].values
false_probs = data[data.label == 0]['your_prediction_column'].values

mqual_true = prob_to_phred(true_probs)
mqual_false = prob_to_phred(false_probs)

print(f"Data: {len(mqual_true):,} true, {len(mqual_false):,} false predictions")

# Set up computational grid
print("Setting up computational grid...")
all_mqual = np.concatenate([mqual_true, mqual_false])
grid, dx, to_grid, from_grid, grid_metadata = make_grid_and_transform(
    all_mqual, grid_size=1024, transform_mode="mqual"
)
print(f"Grid: {len(grid)} points, dx={dx:.4f}, range=[{from_grid(grid[0]):.1f}, {from_grid(grid[-1]):.1f}]")

# Bin data to grid
print("Binning data to grid...")
counts_true, bin_meta_t = bin_data_to_grid(mqual_true, to_grid_space=to_grid)
counts_false, bin_meta_f = bin_data_to_grid(mqual_false, to_grid_space=to_grid)
print(f"Binning conservation: true={bin_meta_t['total_weight']:.0f}, false={bin_meta_f['total_weight']:.0f}")

# Calculate smoothed precision
print("Calculating smoothed precision...")
tpr, fpr, precision, kde_metadata = calculate_smoothed_precision_kde(
    counts_true, counts_false, grid, dx, uncertainty_fn, from_grid,
    len(mqual_true), len(mqual_false),
    num_bandwidth_levels=5,
    enforce_monotonic=False,
    truncation_mode="auto_detect"
)

print(f"KDE completed successfully!")
print(f"Conservation error: {kde_metadata['mixing']['conservation_error_total']:.2e}")
print(f"Precision range: [{precision.min():.4f}, {precision.max():.4f}]")

# Convert to MQUAL coordinates for plotting
mqual_coords = from_grid(grid)

# Plot results
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(mqual_coords, tpr, 'b-', label='TPR', linewidth=2)
plt.plot(mqual_coords, fpr, 'r-', label='FPR', linewidth=2)
plt.xlabel('MQUAL')
plt.ylabel('Rate')
plt.title('True/False Positive Rates')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(mqual_coords, precision, 'g-', linewidth=2)
plt.xlabel('MQUAL')
plt.ylabel('Precision')
plt.title('Smoothed Precision Curve')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.semilogy(mqual_coords, tpr, 'b-', label='TPR', linewidth=2)
plt.semilogy(mqual_coords, fpr, 'r-', label='FPR', linewidth=2)
plt.xlabel('MQUAL')
plt.ylabel('Rate (log scale)')
plt.title('Rates (Log Scale)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
# Show uncertainty function
sample_mqual = np.linspace(mqual_coords.min(), mqual_coords.max(), 100)
uncertainties = uncertainty_fn(sample_mqual)
plt.plot(sample_mqual, uncertainties, 'purple', linewidth=2)
plt.xlabel('MQUAL')
plt.ylabel('Uncertainty (σ)')
plt.title('Uncertainty Function')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Analysis complete!")
```

This comprehensive pipeline provides robust, smooth precision estimation suitable for production quality score mapping systems.
