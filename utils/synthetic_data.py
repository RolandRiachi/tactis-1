import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

from torch.distributions import MultivariateNormal

device = torch.device("cuda")

def generate_random_walk(n_dim=10, n_steps=100000, drift=None, cov=None, test_split=0.1):
    if drift is None:
        drift = torch.zeros((n_dim,))
    if cov is None:
        cov = torch.eye(n_dim)

    N = MultivariateNormal(drift, covariance_matrix=cov)
    X = N.sample((n_steps, )).cumsum(dim=0).T

    X_train = X[:, :int(n_steps*(1-test_split))]
    X_test = X[:, int(n_steps*(1-test_split)) :]
    
    return X_train, X_test

def generate_non_stationary_walk(n_dim=10, n_steps=100000, drifts=None, covs=None, test_split=0.1):
    if drifts is covs is None:
        return generate_random_walk(n_dim, n_steps, test_split)

    steps = []
    for n in range(n_steps):
        d = drifts[n] if drifts else torch.zeros(n_dim)
        c = covs[n] if covs else torch.eye(n_dim)

        sample = torch.tensor(np.random.multivariate_normal(mean=d, cov=c))
        steps.append(sample)

    X = torch.vstack(steps).cumsum(dim=0).T

    X_train = X[:, :int(n_steps*(1-test_split))]
    X_test = X[:, int(n_steps*(1-test_split)) :]

    return X_train, X_test

def generate_big_gauss(n_dim=10, n_steps=100000, drifts=None, covs=None, test_split=0.1):
    if drifts is covs is None:
        return generate_random_walk(n_dim, n_steps, test_split)
    elif drifts is None:
        drifts = torch.zeros(n_dim * n_steps)
    elif covs is None:
        covs = torch.eye(n_dim * n_steps)

    N = MultivariateNormal(drifts, covariance_matrix=covs)
    X = N.sample().view(n_dim, n_steps).cumsum(dim=1)

    X_train = X[:, :int(n_steps*(1-test_split))]
    X_test = X[:, int(n_steps*(1-test_split)) :]

    return X_train, X_test