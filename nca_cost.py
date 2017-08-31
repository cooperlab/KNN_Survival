# -*- coding: utf-8 -*-
"""
Cost function for the NCA

Modified from: https://github.com/RolT/NCA-python
"""
import numpy as np


def _get_P(AX, y, SIGMA = 1):
    
    """
    Gets Pi - the probability that i is correctly classified
    
    Parameters:
    -----------
    AX - transformed matrix X, array-like, shape = [n_features, n_samples]
    y - labels, array-like, shape = [n_samples]
    SIGMA - [0 -> Inf], as SIGMA >> 0, only nearest neighbor is considered, 
             while as SIGMA >> Inf, all points have the same chance.
             Set SIGMA = 1 to strictly follow the original implementation
             of NCA.
    
    Returns:
    --------
    Pij - probability that j will be chosen as i's neighbor, for all i.
          array-like, shape = [n_samples, n_samples]
    
    P - probability that i will be correctly classified, for all i. 
        array-like, shape = [n_samples]
        
    mask - yi == yj fo all i. bool array, shape = [n_samples, n_samples]
    
    """
    
    # Expand dims of AX to [n_features, n_samples, n_samples], where
    # each "channel" in the third dimension is the difference between
    # one sample and all other samples
    normAX = AX[:, :, None] - AX[:, None, :]
    
    # Now get the squared euclidian distance between
    # every patient and all others -> [n_samples, n_samples]
    normAX = np.sum(normAX ** 2, axis=0)

    # Calculate Pij, the probability that j will be chosen 
    # as i's neighbor, for all i's
    def kappa(z):
        return np.exp(-z / SIGMA)
    denomSum = np.sum(kappa(normAX[:, :]), axis=0)
    Pij = kappa(normAX) / denomSum[:, None]

    # Calculate P, the probablity that i will be correctly classified, 
    # for all i
    mask = (y != y[:, None])
    Pijmask = np.ma.masked_array(Pij, mask)
    P = np.array(np.sum(Pijmask, axis=1))
    mask = np.negative(mask)
    
    return Pij, P, mask
    

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
        sigma of kernel that determines emphasis on close neighbors
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
    
    # Get objective function
    #==========================================================================
    
    # Get Pij, Pi and mask of label equality
    Pij, P, mask = _get_P(AX, y, SIGMA = SIGMA)

    # Get objective function to be maximized
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

#    if implementation == "matrix":
#        
#        #
#        # A more efficient but memory-hogging implementation
#        # suitable for small datasets
#        #
#        
#        Xij = Xi[:, :, :, None] * Xi[:, :, None, :]
#    
#        gradf = np.sum(P[:, None, None] * np.sum(Xij[:], axis=1), axis=0)
#    
#        # To optimize (use mask ?)
#        for i in range(N):
#            aux = np.sum(Xij[i, mask[i]], axis=0)
#            gradf -= aux
#            
#        # notice the negative sign (since the goal is to minimize objective)
#        gradf = -2 * np.dot(A, gradf / SIGMA)
#            
#    elif implementation == "loop":
        
    #
    # A less efficient but not memory-intensive implementation
    # suitable for large datasets
    #
    
    gradf = np.zeros(D)
    
    l = 0
    # for l in range(D)
    
    left = P * np.sum(Xi[:, :, l], axis=1)
    right = np.sum(Xi[:, :, l] * mask, axis=1)
    gradf[l] = np.sum(left - right) / SIGMA
    
    
    #right = np.sum(mask * Xi[:, :, l])
        
        
    
    
    

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
