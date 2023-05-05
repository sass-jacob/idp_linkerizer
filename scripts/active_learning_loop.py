import torch
import warnings
import numpy as np
import random
import os
import json
import time 

from tqdm import tqdm
from scipy.stats import norm
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf_discrete
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.transforms import t_batch_mode_transform
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.kernels import RBFKernel
from botorch.acquisition import AcquisitionFunction


warnings.filterwarnings("ignore")
#from botorch.models.transforms import InputTransform

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

class GreedyAcquisitionFunction(AcquisitionFunction):
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        posterior = self.model.posterior(X)
        mean = posterior.mean
        return mean

class EpsilonGreedyAcquisitionFunction(AcquisitionFunction):
    def __init__(self, model, epsilon):
        super().__init__(model=model)
        self.epsilon = epsilon
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        q_values = self.model.posterior(X).mean#.squeeze()
        if random.random() < self.epsilon:
            rand = torch.tensor([random.uniform(0, 1)], device=X.device)
            print(rand)
            return rand
        else:
            return q_values.argmax().unsqueeze(0)

def active_learning_loop_epsilon_greedy(search_space, search_space_enc, X_enc, y, isSampled, epsilon):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device)
    y_standarized = (y-y.mean())/y.std()
    gp = fit_model(X_enc.to(device), y_standarized.to(device))
    acq_func = EpsilonGreedyAcquisitionFunction(gp, epsilon)
    next_X_list = {}
    num_sample = 100  # TODO: change to 100 for linkers

    while (len(next_X_list) < num_sample):
        next_X = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,
            choices=search_space_enc[~isSampled]
        )

        next_X_expanded = next_X[0].expand(search_space_enc.shape[0], -1)

        pos = torch.all(torch.eq(search_space_enc, next_X_expanded), dim=1).nonzero()[0]
        next_X_list[search_space[pos]] = float(next_X[1])
        isSampled[pos] = True

    return next_X_list, isSampled


def active_learning_loop_greedy(search_space, search_space_enc, X_enc, y, isSampled):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device)
    y_standarized = (y-y.mean())/y.std()
    gp = fit_model(X_enc.to(device), y_standarized.to(device))
    acq_func = GreedyAcquisitionFunction(gp)
    next_X_list = {}
    num_sample = 100  # TODO: change to 100 for linkers

    while (len(next_X_list) < num_sample):
        next_X = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,
            choices=search_space_enc[~isSampled]
        )

        next_X_expanded = next_X[0].expand(search_space_enc.shape[0], -1)

        pos = torch.all(torch.eq(search_space_enc, next_X_expanded), dim=1).nonzero()[0]
        next_X_list[search_space[pos]] = float(next_X[1])
        isSampled[pos] = True

    return next_X_list, isSampled



def active_learning_loop_UCB(search_space, search_space_enc, X_enc, y, isSampled):
    """
    INPUT
    search_space_enc (N_linkers x 48 2d tensor, float): Total space of linkers
    X_enc (N_sampled x 48 2d tensor, float): Encoded sampled linkers
    y (N_sampled 1d tensor, float): Corresponding RMSD values
    isSampled (N_linkers 1d tensor, boolean): True if the linker has been sampled, False if not

    RETURN
    next_X_list (dict): key - encoded linkers
                        value - acquisition function value
    isSampled (N_linkers 1d tensor, boolean): True if the linker has been sampled, False if not
    """
    start_al = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device)
    
    y_standarized = (y-y.mean())/y.std()
    gp = fit_model(X_enc.to(device), y_standarized.to(device))
    
    gp.eval()
    beta = 10
    print("Beta: ", beta)
    acq_func = UpperConfidenceBound(gp, beta=beta)
    next_X_list = {}
    num_sample = 100 # TODO: change to 100 for linkers

    print(f"training surrogate model {time.time()-start_al}")
    while(len(next_X_list) < num_sample):
        next_X = optimize_acqf_discrete(
                  acq_function=acq_func,
                  q = 1,
                  choices = search_space_enc[~isSampled]
          )

        next_X_expanded = next_X[0].expand(search_space_enc.shape[0], -1)
        pos = torch.all(torch.eq(search_space_enc, next_X_expanded), dim=1).nonzero()[0]
        next_X_list[search_space[pos]] = float(next_X[1])
        isSampled[pos] = True
        print(next_X_list)
        
    return next_X_list, isSampled


if __name__ == "__main__":
    
    project_dir = ".."
    X_train = []
    y_train = torch.empty((0, 1))

    # Read all rmsd values in /UCB directory
    
    # TODO: Change to the directory that you are running (ex) UCB->greedy)
    # TODO: store the first_random_100_rmsds.txt in the directory (initial set of linkers)
    for root, dirs, files in os.walk(project_dir+'/rmsd/UCB'):
        for file_name in files:
            file_path = os.path.join(root, file_name)  # Get the full file path
            print("Input files", file_path)
            with open(file_path, 'r') as file:
                for line in file:
                    X, y = line.split('\t')
                    X_train.append(X)
                    # -float(y) since we want to minimize RMSD (max. -RMSD)
                    y_train = torch.cat([y_train, -torch.tensor([[float(y)]])], dim=0)
                    
    print(f"Train set size {len(y_train)}")

    start = time.time()
    # TODO: save encoded_linker_dict.json to the directory below. No need to change file name
    with open(project_dir +'/saved_files/encoded_linker_dict.json', 'r') as file:
        search_space_dict = json.load(file)
    search_space = list(search_space_dict.keys())
    search_space_enc = torch.tensor(list(search_space_dict.values()))
    #search_space_enc = torch.tensor([encoding[x] for x in search_space])
    print(search_space_enc.shape)

    # True if elements in search_space is in X_train
    isSampled = torch.tensor(np.isin(search_space, X_train))
    print(isSampled)

    X_train_enc = torch.tensor([search_space_dict[x] for x in X_train])
    
    # TODO: Change to the acqf that you are running (ex) UCB->greedy) 
    acqf = 'epsilongreedy' 
    e = 0.5

    print(f"Reading in files {time.time()-start}")

    if acqf == 'UCB':
        next_X_list, isSampled = active_learning_loop_UCB(search_space, search_space_enc, X_train_enc, y_train, isSampled)
        print(next_X_list)
        print(isSampled)

    if acqf == 'greedy':
        next_X_list, isSampled = active_learning_loop_greedy(search_space, search_space_enc, X_train_enc, y_train, isSampled)
        print(next_X_list)
        print(isSampled)

    if acqf == 'epsilongreedy':
        next_X_list, isSampled = active_learning_loop_epsilon_greedy(search_space, search_space_enc, X_train_enc, y_train, isSampled, e)
        print(next_X_list)
        print(isSampled)


    filename = project_dir +'/active_learning_results/'+acqf+f'/loop_{int(len(y_train)/100)}.jsonl'
    
    # saved as two json objects separated by '\n'
    with open(filename, 'w') as outfile:
        json.dump(next_X_list, outfile)
        outfile.write('\n')
        json.dump(isSampled.tolist(), outfile)
