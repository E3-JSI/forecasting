import torch
from torch import nn
import numpy as np
from sklearn.mixture import GaussianMixture


def nr(x):
    """ Helper function for GMM Linear layer. Works for matrix x of shape (n_samples, n_features). """
    return 1 / (np.sqrt(2 * np.pi)) * torch.exp(-torch.square(x) / 2) + x / 2 * (1 + torch.erf(x / np.sqrt(2)))


def linear_relu_missing_values(W, b, x, p, mean, cov):
    """ Helper function for GMM Linear layer. It can take all samples at once, but it applies only one Gaussian. """
    m = torch.where(torch.isnan(x), mean, x)
    sigma = torch.where(torch.isnan(x), torch.abs(cov), torch.tensor(0.0))
    return p * nr((m @ W + b) / torch.sqrt(
        (torch.square(W) * torch.abs(sigma.view([sigma.shape[0], sigma.shape[1], 1]))).sum(axis=1)))


class GMMLinear(nn.Module):
    """ Layer for processing missing data from the paper Processing of missing data by neural networks. """

    def __init__(self, in_features, out_features, gmm_weights, gmm_means, gmm_covariances):
        """
        in_features and out_features are number of input and output features,
        other parameters are outputs of GaussianMixture.
        """
        super(GMMLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.gmm_weights = nn.Parameter(torch.tensor(np.log(gmm_weights)).float())
        self.gmm_means = nn.Parameter(torch.tensor(gmm_means).float())
        self.gmm_covariances = nn.Parameter(torch.tensor(np.abs(gmm_covariances)).float())
        self.n_gmm = len(gmm_weights)

        if not self.gmm_means.shape == (self.n_gmm, self.in_features):
            raise Exception('gmm_means does not match correct shape (n_components, n_features)')
        if not self.gmm_covariances.shape == (self.n_gmm, self.in_features):
            raise Exception("gmm_covariances does not match correct shape (n_components, n_features). \
                            GaussianMixture must be called with parameter covariance_type='diag'.")

        # weight matrix and bias of this layer
        self.W = nn.Parameter(torch.randn([self.in_features, self.out_features]))
        self.b = nn.Parameter(torch.randn([self.out_features]))

    def forward(self, x):
        indices_full = torch.logical_not(torch.isnan(x).any(axis=1))
        indices_missing = torch.isnan(x).any(axis=1)
        x_full = x[indices_full]
        x_missing = x[indices_missing]
        p = nn.functional.softmax(self.gmm_weights, dim=0)
        out_missing = linear_relu_missing_values(self.W, self.b, x_missing, p[0], self.gmm_means[0],
                                                 self.gmm_covariances[0])

        for i in range(1, self.n_gmm):
            out_missing += linear_relu_missing_values(self.W, self.b, x_missing, p[i], self.gmm_means[i],
                                                      self.gmm_covariances[i])

        out_full = nn.functional.relu(x_full @ self.W + self.b)

        out = torch.zeros(size=(x.shape[0], self.out_features))
        out[indices_full] = out_full
        out[indices_missing] = out_missing

        assert torch.logical_not(torch.any(torch.isnan(out)))
        return out


def create_gmm_linear_layer(X, in_features, out_features, initial_imputer, n_gmm, min_n_gmm=1, max_n_gmm=10,
                            verbose=True, gmm_seed=None):
    """
    Returns object of class GMMLinear. X is input data with missing values, which are imputed with initial_imputer
    and then used as input to GaussianMixture. If initial_imputer is None, then we assume X is already imputed data
    without missing values. n_gmm is number of components of GaussianMixture. If n_gmm is set to -1,
    then all values between min_n_gmm and max_n_gmm are checked and the one with the best BIC score is chosen.
    """
    if initial_imputer is not None:
        x_imputed = initial_imputer.fit_transform(X)
    else:
        x_imputed = X

    if n_gmm == -1:
        n_gmm = best_gmm_n_components(x_imputed, min_n_gmm, max_n_gmm, verbose)
        if verbose:
            print('Best n_components =', n_gmm)

    gmm = GaussianMixture(n_components=n_gmm, covariance_type='diag', random_state=gmm_seed).fit(x_imputed)
    return GMMLinear(in_features, out_features, gmm.weights_, gmm.means_, gmm.covariances_)


def best_gmm_n_components(X, n_min, n_max, verbose=True, gmm_seed=None):
    """ Returns best number of components for GaussianMixture (between n_min and n_max) based on BIC score. """
    min_n = -1
    min_bic = np.infty
    for n_components in range(n_min, n_max + 1):
        gmm = GaussianMixture(n_components=n_components, covariance_type='diag', random_state=gmm_seed).fit(X)
        bic = gmm.bic(X)
        if verbose:
            print(f'n_components={n_components},\tBIC={bic}')
        if min_bic > bic:
            min_bic = bic
            min_n = n_components
    return min_n


def build_multilayer_model(X, dimensions, initial_imputer, n_gmm, gmm_seed):
    """
    Returns neural network with first layer GMMLinear and the rest normal linear layers with ReLU activation.
    Number and dimensions of layers are defined with array dimensions.
    """
    gmm_layer = create_gmm_linear_layer(X, dimensions[0], dimensions[1], initial_imputer, n_gmm, min_n_gmm=1,
                                        max_n_gmm=10, verbose=True, gmm_seed=gmm_seed)

    layers_list = [gmm_layer]
    for i in range(1, len(dimensions) - 1):
        if i > 1:
            layers_list.append(nn.ReLU())
        layers_list.append(nn.Linear(dimensions[i], dimensions[i + 1]))

    return nn.Sequential(*layers_list)
