from src.parameters import Parameters
import inspect
from imports.general import *
from imports.ml import *
import itertools
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood

class GPSampler(object):
    """ Gaussian Process data sampler for bayesian optimization
    """

    def __init__(self, parameters: Parameters):
        self.d = parameters.d
        self.seed = parameters.seed
        np.random.seed(self.seed)
        self.ne_true = np.nan

        #Create grid to make GP on
        self.full_grid_n = 10
        self.input_grid = []
        for i in range(self.d):
            self.input_grid.append(list(np.linspace(-1., 1., self.full_grid_n)))
        if self.d>1:
            self.input = []
            for element in itertools.product(*self.input_grid):
                self.input.append(element)
            self.input = torch.tensor(self.input)
        else:
            self.input=torch.tensor(self.input_grid[0]).unsqueeze(-1)

        self.problem = parameters.problem


        self.theta = 1
        self.out_scale = 1
        self.sigma_sqr = np.exp(-2)
        self.mean_module = gpytorch.means.ZeroMean()
        self.kernel_module = gpytorch.kernels.RBFKernel(ard_num_dims=self.d).double()
        self.kernel_module.lengthscale = torch.tensor(self.theta).double()
        self.means = self.mean_module(self.input)
        self.covar = self.kernel_module(self.input)
        self.mvn = gpytorch.distributions.MultivariateNormal(self.means, self.covar)
        self.f = self.mvn.rsample().detach()
        self.y = self.f + (torch.randn(self.full_grid_n**self.d)*np.sqrt(self.sigma_sqr)).double()
        self.y = self.y.unsqueeze(-1)
        self.model = botorch.models.SingleTaskGP(self.input, self.y)
        self.model.mean_module = self.mean_module
        self.model.covar_module = self.kernel_module
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model.double()
        self.model.eval()
        self.likelihood.eval()
        #Find the max for the sake of BO
        output = self.likelihood(self.model(self.input))
        idx_max = output.loc.detach().argmax().item() #This is the index of the max y
        self.max = output.loc.detach()[idx_max]
        idx_min = output.loc.detach().argmin().item() #This is the index of the min y
        self.min = output.loc.detach()[idx_min]



        self.lbs = [-1 for i in range(self.d)]
        self.ubs = [1 for i in range(self.d)]
        self.X = self.sample_X(parameters.n_initial)
        self.y = self.get_y(self.X)

    def sample_X(self, n_samples: int = 1) -> None:
        X = np.random.uniform(low=self.lbs, high=self.ubs, size=(n_samples, self.d))
        return X

    def get_y(self, X: np.ndarray) -> None:
        return self.likelihood(self.model(torch.tensor(X).double())).mean.detach()

    def __str__(self):
        return str(self.problem)