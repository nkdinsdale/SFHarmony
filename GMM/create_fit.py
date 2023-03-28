import numpy as np
import sklearn.mixture
import torch
from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from gmm import GaussianMixture
import time

def get_params(feat):
    feat = feat.reshape(-1, 32, 1)
    feat = torch.from_numpy(feat)
    model = GaussianMixture(32, 3, 1, covariance_type="diag", init_params="random")
    model.fit(feat, n_iter=20)
    pi = model.pi.detach().numpy().reshape(-1, 3)
    var = model.var.detach().numpy().reshape(-1, 3)
    mu = model.mu.detach().numpy().reshape(-1, 3)
    return pi, var, mu

emb = np.load('../pretrained_model/source_1_X_1_test_embeddings.npy').squeeze()
print(emb.shape)
t0 = time.time()
pi, var, mu = get_params(emb)

t1 = time.time()
t = t1 - t0

print(pi.shape)
print(pi)
print(var.shape)
print(mu.shape)

np.save('normal_model_mednist_X1_training_X1_pi_3', pi)
np.save('normal_model_mednist_X1_training_X1_var_3', var)
np.save('normal_model_mednist_X1_training_X1_mu_3', mu)




