from imports.general import *
from imports.ml import *
from src.parameters import Parameters


class VerificationData(object):
    """Linear sum activation data generation class."""

    def __init__(self, parameters: Parameters = Parameters):
        self.snr = parameters.snr
        self.K = parameters.K
        self.d = parameters.d
        self.seed = parameters.seed
        self.n_test = parameters.n_test
        np.random.seed(self.seed)
        self.lbs = [-1 for b in range(self.d)]
        self.ubs = [1 for b in range(self.d)]
        self.init_model()
        self.X = self.sample_X(parameters.n_initial)
        self.y = self.get_y(self.X)

    def init_model(self) -> None:
        self.sigma_0 = 5
        self.W = np.random.normal(
            loc=0, scale=np.sqrt(self.sigma_0 / (self.d)), size=(self.d, self.K)
        )
        self.w = np.random.normal(loc=0, scale=1, size=(self.K, 1))

    def get_y(self, X: np.array) -> np.ndarray:
        """Generates y according to
        \fb &= \phi(\Xb)\textbf{w} = \sum_{k=1}^K w_k \phi_k(\Xb\textbf{W} )  \label{eq:sim_f}\\
        \yb &= \fb + \epsilon  \,\,,\,\, \epsilon \sim \mathcal{N}(0,\sigma_n^2) \label{eq:sim_y}
        """
        n_samples = X.shape[0]
        self.XW = X @ self.W
        self.phi = np.sin(self.XW)
        self.f = self.phi @ self.w
        self.signal_var = np.var(self.f)
        self.noise_var = self.signal_var / self.snr
        noise_samples = np.random.normal(
            loc=0, scale=np.sqrt(self.noise_var), size=(n_samples, 1)
        )
        y = self.f + noise_samples
        self.ne_true = -norm.entropy(loc=0, scale=np.sqrt(self.noise_var))
        return y

    def sample_X(self, n_samples: int = 1) -> np.ndarray:
        X = np.random.uniform(low=self.lbs, high=self.ubs, size=(n_samples, self.d))
        return X

