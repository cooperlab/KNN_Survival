# -*- coding: utf-8 -*-
"""
Cost function for the NCA

Modified from: https://github.com/RolT/NCA-python
"""
import numpy as np


def _get_P(AX, y, threshold = None, ):
    
    """
    Gets Pi - the probability that i is correctly classified
    
    Parameters:
    -----------
    AX - transformed matrix X, shape = [n_features, n_samples]
    y - labels, shape = [n_samples]
    
    Returns:
    --------
    Pi - 
    
    """
    
    # Expand dims of AX to [n_features, n_samples, n_samples], where
    # each "channel" in the third dimension is the difference between
    # one sample and all other samples
    normAX = AX[:, :, None] - AX[:, None, :]
    
    # Now get the euclidian distance (Fobenius norm) between
    # every patient and all others -> [n_samples, n_samples]
    normAX = np.linalg.norm(normAX, axis=0)

    denomSum = np.sum(np.exp(-normAX[:, :]), axis=0)
    Pij = np.exp(- normAX) / denomSum[:, None]
    if threshold is not None:
        Pij[Pij < threshold] = 0
        Pij[Pij > 1-threshold] = 1

    mask = (y != y[:, None])
    Pijmask = np.ma.masked_array(Pij, mask)
    P = np.array(np.sum(Pijmask, axis=1))
    mask = np.negative(mask)
    
    
    
    

def cost(A, X, y, threshold=None):
    """Compute the cost function and the gradient
    This is the objective function to be minimized
    Parameters:
    -----------
    A : array-like
        Projection matrix, shape = [dim, n_features] with dim <= n_features
    X : array-like
        Training data, shape = [n_features, n_samples]
    y : array-like
        Target values, shape = [n_samples]
    Returns:
    --------
    f : float
        The value of the objective function
    gradf : array-like
        The gradient of the objective function, shape = [dim * n_features]
    """

    (D, N) = np.shape(X)
    A = np.reshape(A, (np.int32(np.size(A) / np.size(X, axis=0)), np.size(X, axis=0))) #mohamed
    (d, aux) = np.shape(A)
    assert D == aux

    AX = np.dot(A, X)
    normAX = np.linalg.norm(AX[:, :, None] - AX[:, None, :], axis=0)

    denomSum = np.sum(np.exp(-normAX[:, :]), axis=0)
    Pij = np.exp(- normAX) / denomSum[:, None]
    if threshold is not None:
        Pij[Pij < threshold] = 0
        Pij[Pij > 1-threshold] = 1

    mask = (y != y[:, None])
    Pijmask = np.ma.masked_array(Pij, mask)
    P = np.array(np.sum(Pijmask, axis=1))
    mask = np.negative(mask)

    f = np.sum(P)

    Xi = X[:, :, None] - X[:, None, :]
    Xi = np.swapaxes(Xi, 0, 2)

    Xi = Pij[:, :, None] * Xi

    Xij = Xi[:, :, :, None] * Xi[:, :, None, :]

    gradf = np.sum(P[:, None, None] * np.sum(Xij[:], axis=1), axis=0)

    # To optimize (use mask ?)
    for i in range(N):
        aux = np.sum(Xij[i, mask[i]], axis=0)
        gradf -= aux

    gradf = 2 * np.dot(A, gradf)
    gradf = -np.reshape(gradf, A.shape) #mohamed
    f = np.size(X, 1) - f

    return [f, gradf]



def cost_g(A, X, y, threshold=None):
    """Compute the cost function and the gradient for the K-L divergence
    Parameters:
    -----------
    A : array-like
        Projection matrix, shape = [dim, n_features] with dim <= n_features
    X : array-like
        Training data, shape = [n_features, n_samples]
    y : array-like
        Target values, shape = [n_samples]
    Returns:
    --------
    g : float
        The value of the objective function
    gradg : array-like
        The gradient of the objective function, shape = [dim * n_features]
    """

    (D, N) = np.shape(X)
    A = np.reshape(A, (np.int32(np.size(A) / np.size(X, axis=0)), np.size(X, axis=0))) #mohamed
    (d, aux) = np.shape(A)
    assert D == aux

    AX = np.dot(A, X)
    normAX = np.linalg.norm(AX[:, :, None] - AX[:, None, :], axis=0)

    denomSum = np.sum(np.exp(-normAX[:, :]), axis=0)
    Pij = np.exp(- normAX) / denomSum[:, None]
    if threshold is not None:
        Pij[Pij < threshold] = 0
        Pij[Pij > 1-threshold] = 1

    mask = (y != y[:, None])
    Pijmask = np.ma.masked_array(Pij, mask)
    P = np.array(np.sum(Pijmask, axis=1))
    mask = np.negative(mask)

    g = np.sum(np.log(P))

    Xi = X[:, :, None] - X[:, None, :]
    Xi = np.swapaxes(Xi, 0, 2)

    Xi = Pij[:, :, None] * Xi

    Xij = Xi[:, :, :, None] * Xi[:, :, None, :]

    gradg = np.sum(np.sum(Xij[:], axis=1), axis=0)

    # To optimize (use mask ?)
    for i in range(N):
        aux = np.sum(Xij[i, mask[i]], axis=0) / P[i]
        gradg -= aux

    gradg = 2 * np.dot(A, gradg)
    gradg = -np.reshape(gradg, A.shape) #mohamed
    g = -g
    
    return [g, gradg]
