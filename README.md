# Focal Loss for LightGBM in R
Implements the alpha-balanced focal loss, gradient, and hessian for use with LightGBM in R.

focal_metric(preds, dtrain, gamma = 1, alpha = 0.5): returns a length-3 list, with the metric *name*, *value*, and *higher_better = FALSE*.

focal_loss(preds, dtrain, gamma = 1, alpha = 0.5): returns a length-2 list, with the evaluated gradient *grad* and hessian *hess*.
