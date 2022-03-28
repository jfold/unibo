from numpy import ndarray
from demo import *


class RandomForest(object):
    """Random forest surrogate class. """

    def __init__(self):
        self.set_hyperparameter_space()

    def set_hyperparameter_space(self):
        self.rf_params_grid = {
            "n_estimators": [10, 100, 1000],
            "max_depth": [5, 10, 20],
            # "max_samples": [int(self.n_initial / 2), int(self.n_initial)],
            "max_features": ["auto", "sqrt"],
        }

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        grid_search = GridSearchCV(
            estimator=RandomForestRegressor(),
            param_grid=self.rf_params_grid,
            cv=5,
            n_jobs=-1,
            verbose=0,
        ).fit(X_train, y_train.squeeze())
        self.model = grid_search.best_estimator_

    def predict(
        self, X_test: np.ndarray, stabilizer: float = 1e-8
    ) -> Tuple[np.ndarray, np.ndarray]:
        mu_predictive = self.model.predict(X_test)
        sigma_predictive = self.calculate_y_std(X_test) + stabilizer
        return mu_predictive, sigma_predictive

    def calculate_y_std(self, X: np.ndarray) -> np.ndarray:
        predictions = self.tree_predictions(X)
        sigma_predictive = np.nanstd(predictions, axis=0)
        return sigma_predictive

    def tree_predictions(self, X: np.ndarray) -> np.ndarray:
        predictions = np.full((len(self.model.estimators_), X.shape[0]), np.nan)
        for i_e, estimator in enumerate(self.model.estimators_):
            predictions[i_e, :] = estimator.predict(X)
        return predictions


class BayesianNeuralNetwork(object):
    """Bayesian Neural Network (BNN) surrogate class. """

    def __init__(self, d: int = 1):
        self.d = d
        self.model = nn.Sequential(
            bnn.BayesLinear(
                prior_mu=0.0,
                prior_sigma=1.0 / self.d,
                in_features=self.d,
                out_features=50,
            ),
            nn.ReLU(),
            bnn.BayesLinear(
                prior_mu=0.0, prior_sigma=1.0 / 50, in_features=50, out_features=10,
            ),
            nn.ReLU(),
            bnn.BayesLinear(
                prior_mu=0.0, prior_sigma=1.0 / 10, in_features=10, out_features=1,
            ),
        )
        self.mse_loss = nn.MSELoss()
        self.kl_loss = bnn.BKLLoss(reduction="mean", last_layer_only=True)
        self.kl_weight = 1.0
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, n_epochs: int = 3000):
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).view((len(y_train), 1))
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
        fig = plt.figure()
        plt.plot(np.log(self.loss))
        plt.ylabel("Log-loss")
        plt.show()

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

        return mu_predictive.squeeze(), sigma_predictive.squeeze()


class Demo(object):
    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def make_dataset(
        self,
        n: int = 100,
        n_train: float = 0.1,
        n_val: float = 0.2,
        sigma_data: float = 1e-1,
    ) -> None:
        np.random.seed(self.seed)
        self.sigma_data = sigma_data
        self.n = n
        self.x = np.linspace(0, 1, self.n)
        self.f = np.sin(2 * self.x)
        self.y = self.f + np.random.normal(0, self.sigma_data, size=self.f.shape)
        (
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test,
            train_idxs,
            self.test_idxs,
        ) = (
            train_test_split(
                self.x,
                self.y,
                np.arange(n),
                shuffle=True,
                train_size=n_train,
                random_state=self.seed,
            ),
        )[
            0
        ]
        (
            self.x_train,
            self.x_val,
            self.y_train,
            self.y_val,
            self.train_idxs,
            self.val_idxs,
        ) = (
            train_test_split(
                self.x_train,
                self.y_train,
                train_idxs,
                shuffle=True,
                test_size=n_val,
                random_state=self.seed,
            ),
        )[
            0
        ]

    def plot_predictive(
        self,
        y_pred: np.ndarray,
        sigma_a: np.ndarray,
        sigma_e: np.ndarray,
        sigma_p: np.ndarray,
        save: bool = False,
    ):
        alpha_e = 0.2 if np.mean(sigma_a) > np.mean(sigma_e) else 0.1

        fig = plt.figure(figsize=(10, 5))
        plt.plot(self.x, self.f, color="black", label="f")
        plt.plot(
            self.x_train, self.y_train, ".", color="black", label="train", alpha=0.6
        )
        plt.plot(self.x_test, self.y_test, ".", color="green", label="test", alpha=0.1)
        plt.plot(self.x, y_pred, color="blue", label="Prediction")
        plt.fill_between(
            self.x, self.f, y_pred, color="red", alpha=0.5, label="Bias",
        )
        # plt.fill_between(
        #     self.x,
        #     y_pred - sigma_a,
        #     y_pred + sigma_a,
        #     facecolor="none",
        #     alpha=alpha_a,
        #     hatch="X",
        #     linewidth=0.0,
        #     edgecolor="blue",
        # )
        plt.plot(
            self.x,
            y_pred - sigma_a,
            "--",
            color="blue",
            label=r"$\sigma_a$" + " (noise)",
        )
        plt.plot(
            self.x, y_pred + sigma_a, "--", color="blue",
        )
        plt.fill_between(
            self.x,
            y_pred - sigma_e,
            y_pred + sigma_e,
            color="blue",
            alpha=alpha_e,
            label=r"$\sigma_e$" + " (variance)",
        )
        plt.fill_between(
            self.x,
            y_pred - sigma_p,
            y_pred + sigma_p,
            color="blue",
            alpha=0.05,
            label=r"$\sigma_p$",
        )

        plt.legend(bbox_to_anchor=(1.1, 1.05))
        plt.ylim([0.0, 1.2])
        # plt.xlim([0.2,1.0])
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        if save:
            fig.savefig(f"visualizations/figures/predictive-example.pdf")
        plt.show()

    def plot_calibration(self, summary: Dict, save: bool = False):
        fig = plt.figure(figsize=(10, 5))
        plt.plot(summary["y_p_array"], summary["y_p_array"], "--")
        plt.plot(
            summary["y_p_array"],
            summary["y_calibration"],
            "--o",
            label=r"$\mathcal{C}_y$ " + f"(error={summary['y_calibration_mse']:.2E})",
            color="blue",
        )
        plt.plot(
            summary["f_p_array"],
            summary["f_calibration"],
            "--o",
            label=r"$\mathcal{C}_f$ " + f"(error={summary['f_calibration_mse']:.2E})",
            color="black",
        )
        plt.xlabel("Expected Confidence Level")
        plt.ylabel("Observed Confidence Level")
        plt.legend(bbox_to_anchor=(1.1, 1.05))
        if save:
            fig.savefig(f"visualizations/figures/calibration-example.pdf")
        plt.show()

    def fit_gp(
        self, sigma_initial: float = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        np.random.seed(self.seed)
        rbf_kernel = 1.0 * RBF(length_scale_bounds=(1e-5, 1e3))
        if sigma_initial is not None:
            noise_kernel = WhiteKernel(
                noise_level=self.sigma_data ** 2, noise_level_bounds="fixed"
            )
        else:
            noise_kernel = WhiteKernel(noise_level_bounds=(1e-10, 1e1))
        kernel = rbf_kernel + noise_kernel
        model = GaussianProcessRegressor(kernel=kernel).fit(
            self.x_train[:, np.newaxis], self.y_train
        )
        y_pred, sigma_p = model.predict(self.x[:, np.newaxis], return_std=True)
        if sigma_initial is not None:
            sigma_a = np.full(sigma_p.shape, sigma_initial)
        else:
            sigma_a = np.full(sigma_p.shape, np.sqrt(np.exp(model.kernel_.theta[-1])))
        sigma_e = np.sqrt(sigma_p ** 2 - sigma_a ** 2)
        bias = np.mean((self.f - y_pred) ** 2)
        # print(model.kernel_)
        # print(model.kernel_.theta)
        # print(np.exp(model.kernel_.theta[-1]))
        # print(np.mean(sigma_a))
        # print(np.mean(sigma_p))
        return sigma_p, sigma_a, sigma_e, bias, y_pred

    def fit_rf(
        self, sigma_a: float = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        np.random.seed(self.seed)
        model = RandomForest()
        model.fit(self.x_train[:, np.newaxis], self.y_train)
        y_pred, sigma_p = model.predict(self.x[:, np.newaxis])
        # if sigma_a is not None:
        #     sigma_a = None
        sigma_e = sigma_p ** 2 - sigma_a ** 2
        bias = np.mean((self.f - y_pred) ** 2)
        return sigma_p, sigma_a, sigma_e, bias, y_pred

    def fit_bnn(
        self, sigma_a: float = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        np.random.seed(self.seed)
        model = BayesianNeuralNetwork()
        model.fit(self.x_train[:, np.newaxis], self.y_train)
        y_pred, sigma_p = model.predict(self.x[:, np.newaxis])
        # if sigma_a is not None:
        #     sigma_a = None
        sigma_e = sigma_p ** 2 - sigma_a ** 2
        bias = np.mean((self.f - y_pred) ** 2)
        return sigma_p, sigma_a, sigma_e, bias, y_pred

