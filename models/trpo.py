import numpy as np
import scipy
import torch

from scipy.sparse.linalg import cg

from models.pg import PolicyNetwork


class TRPO:
    def __init__(self, state_dim, action_dim, discrete, action_std=0.1, lr=1e-3):
        pass