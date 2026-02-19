"""
Linear regression with multi-dimensional data.

regression/generative modelling focus


Model:
  F(X) = X @ W
  Cost: C = ||F(X) - Y||_2^2 + λ||W||_2^2

Shapes: X (n×k), W (k×p), Y (n×p)

Gradient: X^T @ 2E + λ·2W  where E = F(X) - Y

Bias: add a column of 1s to X and an extra row to W.
"""

import numpy as np


def F(X, W):
    """Predictions: F(X) = X @ W."""
    return np.matmul(X, W)


def cost(Y_est, Y, W, Lambda):
    """Cost and error. Returns (E, cost_value)."""
    E = Y_est - Y
    return E, np.linalg.norm(E, 2) + Lambda * np.linalg.norm(W, 2)


def gradient(E, X, W, Lambda):
    """Gradient: 2 * X^T @ E + λ * 2 * W."""
    return 2 * np.matmul(X.T, E) + Lambda * 2 * W


def fit(W, X, Y, alpha, Lambda, max_itr, verbose=True):
    """Gradient descent to minimize regularized MSE."""
    for i in range(max_itr):
        Y_est = F(X, W)
        E, c = cost(Y_est, Y, W, Lambda)
        Wg = gradient(E, X, W, Lambda)
        W = W - alpha * Wg
        if verbose and i % 100 == 0:
            print(c)
    return W


def main():
    n, k, p = 100, 8, 3
    np.random.seed(42)
    X = np.random.random([n, k])
    W = np.random.random([k, p])
    Y = np.random.random([n, p])
    max_itr = 1000
    alpha = 0.0001
    Lambda = 0.01

    # Add bias: concatenate column of 1s to X, extra row to W
    X = np.concatenate((X, np.ones((n, 1))), axis=1)
    W = np.concatenate((W, np.random.random((1, p))), axis=0)

    W = fit(W, X, Y, alpha, Lambda, max_itr)
    return W


if __name__ == "__main__":
    main()
