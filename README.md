# Physics-Informed Neural Network for Damped Harmonic Oscillator

## Overview & Motivation  
Classical solvers for ordinary differential equations (ODEs) rely on meshing, numerical integration, or eigenfunction expansions—and can struggle when data is noisy or parameters vary.  
**Physics-Informed Neural Networks (PINNs)** embed the governing physics directly into the loss function, offering a mesh-free, data-driven alternative.  
This repository demonstrates a simple yet robust PINN that accurately recovers the full family of linear damped-oscillator solutions (undamped, underdamped, critically damped, and overdamped) without architectural “bells and whistles.”

---

## 📥 Input  
- **Physical parameters**  
  - Mass: `m = 1.0`  
  - Spring constant: `k = 4.0`  
- **Damping regimes**  
  - Undamped (`c = 0.0`)  
  - Underdamped (`c = 0.1`)  
  - Critically damped (`c = 4.0`)  
  - Overdamped (`c = 8.0`)  
- **Initial conditions**  
  - Position: `x(0) = 1.0`  
  - Velocity: `x′(0) = 0.0`  
- **Time window**: \( t \in [0,5] \)

---

## 🔧 Pre-processing  
- **Latin-Hypercube sampling**  
  - Generates \(N_f = 20{,}000\) collocation points per epoch, uniformly covering the time domain.  
- **Initial‐condition constraints**  
  - Fixed \(N_{ic} = 50\) samples at \(t=0\).  
- Conversion of all inputs to `float32` tensors for TensorFlow compatibility.

---

## 🏗️ Model Architecture  
- **Feed-forward network**  
  - 3 hidden layers, 50 neurons each  
  - `tanh` activation  
  - Single scalar output \(x(t)\)  
- **Physics loss** computed via automatic differentiation:  
  \[
    \underbrace{m\,x''(t) + c\,x'(t) + k\,x(t)}_{\text{PDE residual}}
  \]  
- **Total loss** =  
  \[
    \text{MSE(initial conditions)} \;+\; \text{MSE(PDE residual)}.
  \]

---

## 🚀 Training & Evaluation  
- **Optimizer**: Adam, learning rate \(1\times10^{-3}\)  
- **Epochs**: 5 000 per damping regime  
- **Per-epoch sampling**: re-draw collocation points each iteration  
- **Monitoring**: `tqdm` progress bars + periodic loss logging  
- **Error metrics** (200-point uniform test grid):  
  - \(L^2\) norm  
  - \(L^\infty\) norm  

| Regime                     | L² Error      | L∞ Error      |
|----------------------------|---------------|---------------|
| **Undamped** (c = 0.0)     | 5.8 × 10⁻⁴    | 1.1 × 10⁻³    |
| **Underdamped** (c = 0.1)  | 5.5 × 10⁻⁴    | 9.8 × 10⁻⁴    |
| **Critically damped** (c=4)| 2.1 × 10⁻³    | 4.1 × 10⁻³    |
| **Overdamped** (c = 8.0)   | 3.8 × 10⁻³    | 7.5 × 10⁻³    |

---

## 🖼️ Output  
- Overlay plots of **PINN prediction** vs. **exact analytical solution** for each regime.  
- Tabulated error norms all comfortably below the \(10^{-2}\) threshold.  
- Interactive progress bars visualize training stability and sampling-induced fluctuations.

---

## ✨ Significance & Next Steps  
- **Proof-of-Concept**: A vanilla PINN can recover a family of second-order ODE solutions with high accuracy.  
- **Generality**: Framework readily extends to more complex linear/nonlinear ODEs or low-dimensional PDEs.  
- **Practical applications**: Hybrid data-driven + physics-based modeling in engineering, control, and scientific computing.  
- **Future work**:  
  - Hyperparameter sweeps (network size, collocation count, learning rate)  
  - Error-vs-time diagnostics  
  - Generalization tests (train on [0,5], evaluate on [5,8])  
  - Extension to PDEs (heat, wave, Navier–Stokes)

---
