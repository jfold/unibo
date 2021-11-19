from imports.general import *
from imports.ml import *
from src.parameters import Defaults, Parameters


class VerificationData(object):
    """Linear sum activation data generation class."""

    def __init__(self, parameters: Parameters = Defaults()):
        self.snr = parameters.snr
        self.K = parameters.K
        self.d = parameters.d
        self.seed = parameters.seed
        self.dtype = parameters.dtype
        self.n_test = parameters.n_test
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        self.init_model()
        self.X = self.sample_X(parameters.n_initial)
        self.y = self.sample_y(self.X)

    def init_model(self) -> None:
        self.sigma_0 = 5
        self.W = tf.cast(
            tfp.distributions.Normal(
                loc=0, scale=np.sqrt(self.sigma_0 / (self.d))
            ).sample(sample_shape=(self.d, self.K), seed=self.seed),
            dtype=self.dtype,
        )
        self.w = tf.cast(
            tfp.distributions.Normal(loc=0, scale=1).sample(
                sample_shape=(self.K, 1), seed=self.seed
            ),
            dtype=self.dtype,
        )

    def sample_y(self, X: np.array) -> np.array:
        """Generates y according to
        \fb &= \phi(\Xb)\textbf{w} = \sum_{k=1}^K w_k \phi_k(\Xb\textbf{W} )  \label{eq:sim_f}\\
        \yb &= \fb + \epsilon  \,\,,\,\, \epsilon \sim \mathcal{N}(0,\sigma_n^2) \label{eq:sim_y}
        """
        n_samples = X.shape[0]
        self.XW = tf.linalg.matmul(X, self.W)
        self.phi = tf.math.sin(self.XW)
        self.f = tf.linalg.matmul(self.phi, self.w)
        self.signal_var = np.var(self.f)
        self.noise_var = self.signal_var / self.snr
        self.noise_dist = tfp.distributions.Normal(
            loc=0, scale=tf.math.sqrt(self.noise_var)
        )
        noise_samples = tf.cast(
            self.noise_dist.sample(sample_shape=(n_samples, 1), seed=self.seed),
            dtype=self.dtype,
        )
        y = self.f + noise_samples
        self.ne_true = -self.noise_dist.entropy()
        return y.numpy()

    def sample_X(self, n_samples: int = 1) -> np.array:
        X = tf.cast(
            tfp.distributions.Uniform(low=-1, high=1).sample(
                sample_shape=(n_samples, self.d), seed=self.seed
            ),
            dtype=self.dtype,
        )
        return X.numpy()


