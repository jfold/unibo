
from gpytorch.mlls import ExactMarginalLogLikelihood
import torch
import numpy as np
import botorch
import matplotlib.pyplot as plt
import pickle
import gpytorch

train_x = np.array([0.1, 0.4, 0.6, 0.9])
train_y = 4*train_x-4*train_x**2
print(np.max(train_y))
n=4

model = botorch.models.SingleTaskGP(torch.tensor(train_x).unsqueeze(-1), torch.tensor(train_y).unsqueeze(-1))
likelihood = ExactMarginalLogLikelihood(model.likelihood, model)
botorch.fit.fit_gpytorch_model(likelihood)

test_x = torch.linspace(0,1, 100).double()
test_y = test_x-test_x**2

# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = torch.linspace(0, 1, 51)
    posterior = model.posterior(test_x, observation_noise=True)
    mu_predictive = posterior.mean.cpu().detach().numpy().squeeze()
    sigma_predictive = (np.sqrt(posterior.variance.cpu().detach().numpy())).squeeze()

print(mu_predictive)
print(sigma_predictive)
with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    # Plot training data as black stars
    ax.plot(train_x, train_y, 'k*')
    # Plot predictive means as blue line
    ax.plot(test_x, mu_predictive, 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x, mu_predictive-sigma_predictive, mu_predictive+sigma_predictive, alpha=0.5)
    ax.set_ylim([0, 1.5])
    ax.set_xlim([0,1])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    plt.show()
torch.save(model.state_dict(), 'objects/BO_test_GP')
