from dataclasses import asdict
from botorch.posteriors.posterior import Posterior
from src.dataset import Dataset
from src.parameters import Parameters
from imports.general import *
from imports.ml import *


class BayesianNeuralNetwork(BatchedMultiOutputGPyTorchModel):
    """Bayesian Neural Network (BNN) surrogate class. """

    def __init__(self, parameters: Parameters, dataset: Dataset, name: str = "BNN"):
        super().__init__()
        # TODO: hyperparameter tuning: pruning?
        self.name = name
        self.d = parameters.d
        self.model = nn.Sequential(
            bnn.BayesLinear(
                prior_mu=0.1, prior_sigma=1.0, in_features=self.d, out_features=500
            ),
            nn.ReLU(),
            bnn.BayesLinear(
                prior_mu=0.0, prior_sigma=0.1, in_features=500, out_features=1,
            ),
        )
        self.mse_loss = nn.MSELoss()
        self.kl_loss = bnn.BKLLoss(reduction="mean", last_layer_only=False)
        self.kl_weight = 1.0
        self._set_dimensions(train_X=dataset.data.X, train_Y=dataset.data.y)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.1)
        self.fit(X_train=dataset.data.X, y_train=dataset.data.y)

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x, covar_x = self.predict(x)
        mean_x = torch.tensor(mean_x.squeeze())
        covar_x = torch.tensor(np.diag(covar_x.squeeze()))
        return MultivariateNormal(mean_x, covar_x)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, n_epochs: int = 500):
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        self.loss = []
        for step in range(n_epochs):
            pre = self.model(X_train)
            mse = self.mse_loss(pre, y_train)
            kl = self.kl_loss(self.model)
            elbo = mse + self.kl_weight * kl
            self.loss.append(elbo)
            self.optimizer.zero_grad()
            elbo.backward()
            self.optimizer.step()

        self.loss.append(elbo)
        # fig = plt.figure()
        # plt.plot(np.log(self.loss))
        # plt.show()
        # raise ValueError()

    def predict(
        self, X_test: np.ndarray, stabilizer: float = 1e-8, n_ensemble: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates mean (prediction) and variance (uncertainty)"""
        X_test = torch.tensor(X_test, dtype=torch.float32)
        predictions = np.full((n_ensemble, X_test.shape[0]), np.nan)
        for n in range(n_ensemble):
            predictions[n, :] = self.model(X_test).cpu().detach().numpy().squeeze()
        mu_predictive = np.nanmean(predictions, axis=0)
        sigma_predictive = np.nanstd(predictions, axis=0) + stabilizer

        mu_predictive = (
            mu_predictive[:, np.newaxis] if mu_predictive.ndim == 1 else mu_predictive
        )
        sigma_predictive = (
            sigma_predictive[:, np.newaxis]
            if sigma_predictive.ndim == 1
            else sigma_predictive
        )

        return mu_predictive, sigma_predictive
