from dataclasses import asdict
from botorch.posteriors.posterior import Posterior
from src.dataset import Dataset
from src.parameters import Parameters
from imports.general import *
from imports.ml import *


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, 50)
        self.hidden_fc = nn.Linear(50, 10)
        self.output_fc = nn.Linear(10, output_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        h_1 = F.relu(self.input_fc(x))
        h_2 = F.relu(self.hidden_fc(h_1))
        y_pred = self.output_fc(h_2)
        return y_pred


class DeepEnsemble(BatchedMultiOutputGPyTorchModel):
    """DeepEnsemble (DE) surrogate class. """

    def __init__(self, parameters: Parameters, dataset: Dataset, name: str = "DE"):
        super().__init__()
        self.name = name
        self.seed = parameters.seed
        self.d = parameters.d
        self.change_std = parameters.change_std
        self.std_change = parameters.std_change
        self.n_networks = 10
        self.mse_loss = nn.MSELoss()
        # nn.GaussianNLLLoss
        self._set_dimensions(train_X=dataset.data.X_train, train_Y=dataset.data.y_train)
        self.fit(X_train=dataset.data.X_train, y_train=dataset.data.y_train)

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x, covar_x = self.predict(x)
        mean_x = torch.tensor(mean_x.squeeze())
        covar_x = torch.tensor(np.diag(covar_x.squeeze()))
        return MultivariateNormal(mean_x, covar_x)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_epochs: int = 1500,
        rand_portion: float = 1.0,
    ):
        n_samples = X_train.shape[0]
        self.models = []
        sigmas = []
        preds = []
        for n in range(self.n_networks):
            torch.manual_seed(self.seed + 2022 + n)
            idxs = np.random.choice(
                range(n_samples), size=int(rand_portion * n_samples), replace=False
            )
            X_train_torch = torch.tensor(X_train[idxs, :], dtype=torch.float32)
            y_train_torch = torch.tensor(y_train[idxs, :], dtype=torch.float32)
            model = MLP(self.d)
            self.optimizer = torch.optim.Adam(
                model.parameters(), lr=1e-2, weight_decay=1e-3
            )
            loss = []
            for _ in range(n_epochs):
                pre = model(X_train_torch)
                mse = self.mse_loss(pre, y_train_torch)
                loss.append(mse)
                self.optimizer.zero_grad()
                mse.backward()
                self.optimizer.step()
            self.models.append(model)

            # Compute observation_noise
            pre = pre.cpu().detach().numpy()
            preds.append(np.mean(pre))
            sigmas.append(np.mean((pre - y_train_torch.cpu().detach().numpy()) ** 2))

        self.observation_noise = np.sqrt(
            np.mean(sigmas) + np.mean(preds - np.mean(preds))
        )
        # plt.figure()
        # plt.plot(loss)
        # plt.show()
        # raise ValueError()

    def predict(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray = None,
        stabilizer: float = 1e-8,
        observation_noise: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates mean (prediction) and variance (uncertainty). If y_test is parsed, the test loss is printed."""
        X_test = torch.tensor(X_test, dtype=torch.float32)
        predictions = np.full((self.n_networks, X_test.shape[0]), np.nan)
        for n in range(self.n_networks):
            predictions[n, :] = self.models[n](X_test).cpu().detach().numpy().squeeze()

        mu_predictive = np.nanmean(predictions, axis=0)
        sigma_predictive = np.nanstd(predictions, axis=0) + stabilizer

        if observation_noise:
            sigma_predictive += self.observation_noise

        mu_predictive = (
            mu_predictive[:, np.newaxis] if mu_predictive.ndim == 1 else mu_predictive
        )
        sigma_predictive = (
            sigma_predictive[:, np.newaxis]
            if sigma_predictive.ndim == 1
            else sigma_predictive
        )
        if y_test is not None:
            test_loss = self.mse_loss(
                torch.tensor(mu_predictive, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.float32),
            )
            print(test_loss)
        if self.change_std:
            sigma_predictive *= self.std_change
        return mu_predictive, sigma_predictive
