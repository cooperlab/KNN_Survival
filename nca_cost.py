# -*- coding: utf-8 -*-
"""
Cost function for the NCA

Modified from: https://github.com/RolT/NCA-python
"""
import numpy as np


def cost(A, X, y, SIGMA = 1, LAMBDA = 0):
    
    """
    Compute the cost function and the gradient
    This is the objective function to be minimized
    
    Parameters:
    -----------
    A : array-like
        Feature weights, shape = [n_features]
    X : array-like
        Training data, shape = [n_features, n_samples]
    y : array-like
        Target values, shape = [n_samples]
    SIGMA: float
        sigma of kernel that determines emphasis on close neighbors.
        [0 -> Inf], as SIGMA >> 0, only nearest neighbor is considered, 
        while as SIGMA >> Inf, all points have the same chance.
        Set SIGMA = 1 to strictly follow the original implementation
        of NCA.
    LAMBDA: float
        reguarization parameter
        
    Returns:
    --------
    f : float
        The value of the objective function
    gradf : array-like
        The gradient of the objective function wrt feature weights, 
        shape = [n_features]
    """

    # Setting things up
    #==========================================================================
    
    # fix and check dims
    (D, N) = np.shape(X)
    assert D == len(A)

    # avoid division by zero
    assert SIGMA > 0
    
    # diagonalize weights
    A_diag = np.zeros((D, D))
    np.fill_diagonal(A_diag, A)

    # transform feats according to current A
    AX = np.dot(A_diag, X)
    A_diag = None
    
    
    # Get Pi - the probability that i is correctly classified
    #==========================================================================
    
    # Expand dims of AX to [n_features, n_samples, n_samples], where
    # each "channel" in the third dimension is the difference between
    # one sample and all other samples
    normAX = AX[:, :, None] - AX[:, None, :]
    
    # Now get the squared euclidian distance between
    # every patient and all others -> [n_samples, n_samples]
    
    #normAX = np.sum(normAX ** 2, axis=0)
    normAX = np.linalg.norm(AX[:, :, None] - AX[:, None, :], axis=0)

    # Calculate Pij, the probability that j will be chosen 
    # as i's neighbor, for all i's
    def kappa(z):
        return np.exp(-z / SIGMA)
    denomSum = np.sum(kappa(normAX[:, :]), axis=0)
    Pij = kappa(normAX) / denomSum[:, None]

    # Calculate P, the probablity that i will be correctly 
    # classified, for all i
    mask = (y != y[:, None])
    Pijmask = np.ma.masked_array(Pij, mask)
    P = np.array(np.sum(Pijmask, axis=1))
    mask = np.negative(mask)
    
    
    # Get objective function
    #==========================================================================

    # Objective function to be maximized
    f = np.sum(P) - LAMBDA * np.sum(A ** 2)
    
    
    # Get gradient of f w.r.t feature weights 
    #==========================================================================
    
    # Expand dims of X to [n_samples, n_samples, n_features], where
    # each "channel" in the first dimension is the difference between
    # one sample and all other samples
    # then multiply by Pij
    Xi = X[:, :, None] - X[:, None, :]
    Xi = np.swapaxes(Xi, 0, 2)
    Xi = Pij[:, :, None] * Xi

    # Go through features and calculate gradient
    
    gradf = np.zeros(D)
    
    #l = 0
    for l in range(D):
        left = P * np.sum(Xi[:, :, l], axis=1)
        right = np.sum(Xi[:, :, l] * mask, axis=1)
        gradf[l] = 2 * ((np.sum(left - right) / SIGMA) - LAMBDA) * A[l]
        

    return [f, gradf]
