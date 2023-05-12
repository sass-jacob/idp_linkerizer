import torch
import warnings
import numpy as np
import random
import os
import json
import time 

from tqdm import tqdm
from torch import Tensor
from scipy.stats import norm
from typing import Dict, Optional, Tuple, Union
from botorch.acquisition import AcquisitionFunction, AnalyticAcquisitionFunction
from botorch.acquisition import UpperConfidenceBound, PosteriorMean
from botorch.acquisition.objective import PosteriorTransform
from botorch.optim import optimize_acqf_discrete
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.fit import fit_gpytorch_model
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.transforms import t_batch_mode_transform
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.kernels import RBFKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

class PosteriorSigma(AnalyticAcquisitionFunction):
    """
    Single-outcome Posterior Mean.
    """

    def __init__(
        self,
        model: Model,
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
    ) -> None:
        r"""Single-outcome Posterior Mean.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem. Note
                that if `maximize=False`, the posterior mean is negated. As a
                consequence `optimize_acqf(PosteriorMean(gp, maximize=False))`
                actually returns -1 * minimum of the posterior mean.
        """
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Evaluate the posterior mean and sigma on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A tuple of two `(b1 x ... bk)`-dim tensors of Posterior Mean and Sigma values
            at the given design points `X`.
        """
        _, sigma = self._mean_and_sigma(X)
        return sigma 
    
def fit_model(X, y):
    gp = SingleTaskGP(X, y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

  # Set mll to dtype and device of X
    mll = mll.to(X)
    optimizer = torch.optim.Adam(gp.parameters(), lr=1e-3)

    gp.train()
    epochs = 1000

    for epoch in tqdm(range(epochs), "Training GP model", disable=True):
        optimizer.zero_grad()
        output = gp(X)
        loss = -mll(output, y.squeeze())
        loss.backward()
        optimizer.step()
        if (epoch+1)%100 == 0:
              print(f"Epoch {epoch+1 :>3d}/{epochs} - Loss: {loss.item():>4.3f}, \
                    noise: {gp.likelihood.noise.item():>4.3f}")

    #fit_gpytorch_model(mll)
    return gp

def get_all_acqf_vals(acq_func, search_space_enc, max_batch_size = 2048):
    
    search_space_batched = search_space_enc.unsqueeze(-2)

    with torch.no_grad():
        acq_values = torch.cat([acq_func(X_) for X_ in search_space_batched.split(max_batch_size)], dim=-1)
        
    return acq_values.squeeze()


def get_distribution(search_space_enc, X_enc, y, isSampled):
       
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device)
    
    y_standarized = (y-y.mean())/y.std()
    gp = fit_model(X_enc.to(device), y_standarized.to(device))
    gp.eval()
    
    max_batch_size = 2048
    search_space_batched = search_space_enc[~isSampled].unsqueeze(-2)
    
    with torch.no_grad():
        mu_vals = torch.cat([PosteriorMean(gp)(X_) for X_ in search_space_batched.split(max_batch_size)])
        print("Done with mean")
        sigma_vals = torch.cat([PosteriorSigma(gp)(X_) for X_ in search_space_batched.split(max_batch_size)])
    print("Done calculating mu and sigma values")
    
    return mu_vals, sigma_vals

if __name__ == "__main__":
    
    project_dir = ".."
    X_train = []
    y_train = torch.empty((0, 1))
    
    file_list = ['UCB_loop_0_rmsd_values.txt']
    for file in file_list:
        file_path = project_dir+'/rmsd/UCB/'+file
        with open(file_path, 'r') as file:
            for line in file:
                X, y = line.split('\t')
                X_train.append(X)
                # -float(y) since we want to minimize RMSD (max. -RMSD)
                y_train = torch.cat([y_train, -torch.tensor([[float(y)]])], dim=0)
                    
    print(f"Train set size {len(y_train)}")
    
    with open(project_dir +'/saved_files/encoded_linker_dict.json', 'r') as file:
        search_space_dict = json.load(file)
    search_space = list(search_space_dict.keys())
    search_space_enc = torch.tensor(list(search_space_dict.values()))
    #search_space_enc = torch.tensor([encoding[x] for x in search_space])
    print(search_space_enc.shape)

    # True if elements in search_space is in X_train
    isSampled = torch.tensor(np.isin(search_space, X_train))
    print(isSampled.sum())

    X_train_enc = torch.tensor([search_space_dict[x] for x in X_train])
    
    acqf = 'UCB'
    
    print("Starting Evaluation")

    mu_vals, sigma_vals = mu_vals, sigma_vals = get_distribution(search_space_enc, X_train_enc, y_train, isSampled)
    print("mu", mu_vals.shape)
    print("sigma", sigma_vals.shape)
    print("mu", mu_vals)
    print("sigma", sigma_vals)
    
    filename = project_dir+'/analysis/'+f'/UCB_distribution_loop_{int(len(y_train)/100)}.jsonl'
    with open(filename, 'w') as outfile:
        json.dump(mu_vals.tolist(), outfile)
        outfile.write('\n')
        json.dump(sigma_vals.tolist(), outfile)
    
    
