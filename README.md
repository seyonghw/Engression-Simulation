# Engression Simulation Study
This project compares five Engression loss functions (power(β=0.5), power(β=1), exp, log1p, frac)
using synthetic pre-ANM and post-ANM models.

## Workflow
- `test.py`: Runs simulation (72 × 10 cases)
- `plot_results.py`: Boxplots of mean MSEs per (model, dims)
- `plot_loss_distributions.py`: Overall MSE distribution comparisons (4 losses)
- `outlier.ipynb`: Summarizes high-MSE (>1) cases

## Outputs
Results and figures are stored in the `result/` folder.
