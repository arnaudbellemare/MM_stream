# Robust Rolling PCA (R²-PCA) Implementation

## Overview

This implementation provides **Robust Rolling PCA (R²-PCA)** for stable principal component analysis in crypto markets. R²-PCA addresses the instability issues in traditional rolling PCA by maintaining consistent eigenvector orientations across time windows.

## Key Features

### 1. Cosine-Similarity Alignment
- **Prevents sign flips** between consecutive time windows
- **Maintains consistent ordering** of principal components
- **Ensures stable economic interpretation** of PC1 and PC2

### 2. Rolling Window Analysis
- **Captures evolving market dynamics** through time
- **Smooth transitions** between market regimes
- **Stable loadings** for contrarian-strength analysis

### 3. Crypto Market Application
- **PC1 (Market Factor):** Represents common market movements
- **PC2 (Contrarian Strength):** Captures defensive/contrarian behavior
- **Stable orientations** enable reliable trend analysis

## Implementation Details

### Core Function: `compute_r2_pca()`

```python
def compute_r2_pca(window_data, prev_evecs, n_components=2):
    """
    Compute R²-PCA with cosine similarity alignment.
    
    Args:
        window_data: Standardized returns matrix (n_samples, n_assets)
        prev_evecs: Previous window eigenvectors or None for first window
        n_components: Number of principal components to extract
        
    Returns:
        aligned_evecs: Aligned eigenvectors with consistent orientation
    """
```

### Algorithm Steps

1. **Standard PCA:** Compute principal components on current window
2. **Cosine Similarity:** Calculate dot products with previous eigenvectors
3. **Alignment:** Match each new eigenvector to closest previous axis
4. **Sign Flipping:** Flip eigenvectors with negative dot products
5. **Consistency:** Maintain stable orientations across windows

### Configuration Parameters

```python
ALIGNMENT_METHOD = "cosine"  # Cosine similarity alignment
FLIP_THRESHOLD = 0.0         # Always flip if negative dot product
```

## Usage in Crypto Markets

### PC1 Interpretation (Market Factor)
- **High PC1:** Strong market leader, moves with overall crypto market
- **Low PC1:** Market laggard, weak correlation with market trends
- **Stable orientation:** Consistent interpretation across time windows

### PC2 Interpretation (Contrarian Strength)
- **High PC2:** Contrarian strength, performs well in down markets
- **Low PC2:** Persistent weakness, vulnerable to market stress
- **Defensive behavior:** Assets that buck market trends

### Benefits for Crypto Analysis

1. **Stable Trend Identification:** No sign flips disrupt trend analysis
2. **Consistent Economic Meaning:** PC1 and PC2 maintain interpretation
3. **Smooth Regime Transitions:** Gradual evolution of market structure
4. **Reliable Contrarian Signals:** Stable identification of defensive assets

## Integration with Existing Pipeline

The R²-PCA implementation integrates seamlessly with the existing quantitative research dashboard:

- **New Tab:** Dedicated R²-PCA analysis interface
- **Interactive Controls:** Asset selection, window size, timeframe
- **Visualization:** PC1 vs PC2 plots, time series, loadings
- **Interpretation:** Clear explanation of results and economic meaning

## Technical Implementation

### Data Requirements
- **Multiple Assets:** At least 2 crypto assets for PCA
- **Aligned Returns:** Common time periods across assets
- **Standardization:** Returns normalized to zero mean, unit variance
- **Sufficient History:** Window size + analysis period

### Performance Considerations
- **Computational Efficiency:** O(n²) complexity for alignment
- **Memory Usage:** Stores previous eigenvectors for alignment
- **Caching:** Results cached for repeated analysis
- **Scalability:** Handles multiple assets efficiently

## Example Results

### Market Regime Analysis
- **Bull Markets:** High PC1, mixed PC2 (leaders vs laggards)
- **Bear Markets:** Low PC1, high PC2 (contrarian strength emerges)
- **Corrections:** Mixed PC1, high PC2 (defensive assets outperform)

### Asset Classification
- **Market Leaders:** High PC1, low PC2 (BTC, ETH in bull markets)
- **Contrarian Assets:** Low PC1, high PC2 (defensive coins, stablecoins)
- **Laggards:** Low PC1, low PC2 (weak performers across regimes)

## Future Enhancements

1. **Dynamic Window Sizing:** Adaptive window sizes based on market volatility
2. **Multi-Asset Optimization:** Enhanced handling of large asset universes
3. **Regime Detection:** Automatic identification of market regimes
4. **Portfolio Construction:** Integration with portfolio optimization

## References

- **Academic Foundation:** Based on robust PCA literature
- **Crypto Applications:** Adapted for cryptocurrency market dynamics
- **Implementation:** Streamlit-based interactive dashboard
- **Visualization:** Plotly charts for comprehensive analysis

---

*This implementation provides a robust foundation for principal component analysis in crypto markets, enabling stable trend identification and contrarian strength analysis.*
