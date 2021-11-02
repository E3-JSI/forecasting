import torch
import numpy as np
import sys
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer

sys.path.insert(0, './lib')
from gmm_linear_layer import create_gmm_linear_layer


class TorchNetwork:
    """
    Wrapper for PyTorch. The class is designed to behave as a sklearn model with fit and predict methods.
    """

    def __init__(self, parent):
        self.parent = parent
        self.ann = None

    def fit(self, x, y):
        loss_function = torch.nn.MSELoss()
        dimensions = [np.shape(x)[1], (np.shape(x)[1] + 1) // 2, 1]
        layers = (2 * len(dimensions) - 3) * [torch.nn.ReLU()]
        layers[::2] = [torch.nn.Linear(dimensions[k], dimensions[k + 1]) for k in range(len(dimensions) - 1)]

        if self.parent.gmm_layer:
            gmm_layer = create_gmm_linear_layer(x, np.shape(x)[1], np.shape(x)[1], self.parent.initial_imputer,
                                                n_gmm=self.parent.n_gmm,
                                                min_n_gmm=self.parent.min_n_gmm,
                                                max_n_gmm=self.parent.max_n_gmm,
                                                gmm_seed=self.parent.gmm_seed,
                                                verbose=self.parent.verbose)
            self.ann = torch.nn.Sequential(gmm_layer, *layers)
        else:
            self.ann = torch.nn.Sequential(*layers)

        opt = torch.optim.Adam(self.ann.parameters(), lr=self.parent.learning_rate)

        # partition 3k-tuples randomly into batches of 64 for nt-thread parallelization
        train_batches = torch.utils.data.DataLoader(
            [[x[i], y[i]] for i in range(len(x))], batch_size=self.parent.batch_size, shuffle=True,
            num_workers=self.parent.num_workers)

        for training_round in range(self.parent.training_rounds):
            print(f'Train epoch {training_round:3}', end="\t")

            loss = 0
            # randomly select a batch of 64 pairs x,y, each of length 3k,3l
            for x, y in train_batches:
                # Training pass; set the gradients to 0 before each loss calculation.
                opt.zero_grad()
                loss = loss_function(self.ann(x.float()), y.unsqueeze(1).float())

                # backpropagation: compute gradients
                loss.backward()

                # apply gradients to improve weights ann[k].weight.grad to minimize f
                opt.step()
            print(f'loss:{loss.item():12.3f}')

    def predict(self, x):
        return self.ann(torch.tensor(x).float()).detach().tolist()
