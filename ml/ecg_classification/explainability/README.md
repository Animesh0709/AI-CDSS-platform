# Phase 8.0: 1D ECG ResNet Explainability Module

## 1. Overview
This module provides waveform-level explainability for the trained `ECGResNet1D` multi-label 12-lead ECG abnormality classifier. It exposes spatio-temporal and lead-specific attributions to interpret model predictions without modifying or retraining the underlying network.

Target Superclasses:
- `NORM`: Normal ECG
- `MI`: Myocardial Infarction
- `STTC`: ST/T Change
- `CD`: Conduction Disturbance
- `HYP`: Hypertrophy

---

## 2. Methodology

### A. 1D Grad-CAM (`grad_cam.py`)
- **Target Layer**: The final residual stage (`model.layer4`), producing a feature map of shape `(Batch, 128, 32)` for an input of shape `(Batch, 12, 1000)`.
- **Target Objective**: Class-specific raw logit $y^c$ (before sigmoid probability activation), preserving gradient sensitivity without saturation:
  $$\alpha_k^c = \frac{1}{L} \sum_{t=1}^L \frac{\partial y^c}{\partial A_k(t)}$$
- **Feature Map Weighting**:
  $$L_{\text{Grad-CAM}}^c(t) = \text{ReLU}\left( \sum_k \alpha_k^c A_k(t) \right)$$
- **Upsampling**: Linear 1D interpolation from $L=32$ back to the original 1,000 temporal samples ($100\text{ Hz} \times 10\text{ s}$).
- **Normalization**: Per-record min-max scaling to $[0.0, 1.0]$.
- **Zero-Gradient Handling**: If gradients or weights are identically zero, the method outputs a true zero attribution map rather than noisy artifacts.

### B. Integrated Gradients (`integrated_gradients.py`)
- **Axiomatic Foundation**: Satisfies Completeness and Implementation Invariance (Sundararajan et al., 2017).
- **Attribution Formulation**:
  $$\text{Attr}_i^c(x) = (x_i - x_i') \times \int_{0}^1 \frac{\partial F_c(x' + \alpha(x - x'))}{\partial x_i} \, d\alpha$$
- **Baseline Options**:
  - `zero`: Zero electrical isoelectric line (standard default).
  - `mean`: Channel-wise mean vector computed strictly from training records (folds 1–8).
  - `random`: Gaussian perturbation baseline.
- **Numerical Approximation**: 50 interpolation steps using the trapezoidal rule or Riemann mid-point sum.
- **Completeness Verification**: Computes the numerical error between the sum of attributions and the score difference:
  $$\text{Error} = \left| \sum_{l=1}^{12} \sum_{t=1}^{1000} \text{Attr}_{l,t}^c - (F_c(x) - F_c(x')) \right|$$
- **Lead-Wise Importance**: Aggregates total absolute energy across the temporal axis:
  $$\text{Importance}_l^c = \sum_{t=1}^{1000} \left| \text{Attr}_{l,t}^c \right|$$

---

## 3. Visualization (`visualization.py`)
Produces multi-panel diagnostic figures:
1. **12-Lead ECG Signals**: Standard clinical lead ordering (`I, II, III, aVR, aVL, aVF, V1-V6`) with Grad-CAM salience color overlay.
2. **Lead-Wise Importance Ranking**: Horizontal bar chart indicating relative attribution percentage per lead.
3. **Temporal Attention Profile**: 1D Grad-CAM activation trace with high-saliency threshold cutoff.
4. **Lead-by-Time Heatmap**: 2D Diverging heatmap (`coolwarm`) showing positive (red / contributing) vs negative (blue / attenuating) feature attributions.

---

## 4. Validation & Sanity Checks (`validation.py`)
The module includes a systematic validation test harness:
1. **Dimensionality**: Verifies output tensor shapes for batched and unbatched inputs.
2. **Numerical Finiteness**: Confirms absence of `NaN` or `Inf` values.
3. **Input Immutability**: Guarantees input waveform tensors are never modified in place.
4. **Class Specificity**: Verifies that different target classes generate distinct attribution patterns.
5. **IG Completeness**: Confirms path integral convergence within $<5\%$ relative error.
6. **Model Randomization Check**: Confirms that weight scrambling destroys attribution patterns (Adebayo et al., 2018).
7. **Zero-Baseline Edge Case**: Verifies zero input against zero baseline produces zero attribution.
8. **Hardware Compatibility**: Validated on CPU and Apple Silicon MPS.

---

## 5. Methodological and Clinical Safeguards
- **Explanations are NOT Clinical Evidence**: Highlighted ECG segments indicate statistical correlations utilized by neural network filters, not verified anatomical lesions or confirmed infarct locations.
- **Non-Causal**: Attribution does not demonstrate physiological causation.
- **Raw Waveform Scope**: This module explains the 1D ResNet raw numerical classifier; it does not explain optical scanning or pixel-domain image artifacts.
