from imports.general import *
from imports.ml import *


class VerificationData(object):
    """Linear sum activation data generation class."""

    def __init__(self, snr: float = 1.0, K: int = 1, seed: int = 0):
        self.snr = snr
        self.K = K
        self.seed = seed
        self.summary.update({"K": self.K, "SNR": self.snr})
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        self.init_model()

    def init_model(self):
        self.sigma_0 = 5
        self.W = tf.cast(
            tfp.distributions.Normal(
                loc=0, scale=np.sqrt(self.sigma_0 / (self.D))
            ).sample(sample_shape=(self.D, self.K), seed=self.seed),
            dtype=self.dtype,
        )
        self.w = tf.cast(
            tfp.distributions.Normal(loc=0, scale=1).sample(
                sample_shape=(self.K, 1), seed=self.seed
            ),
            dtype=self.dtype,
        )

    def generate_data(self):
        """Method generates data according to
        \fb &= \phi(\Xb)\textbf{w} = \sum_{k=1}^K w_k \phi_k(\Xb\textbf{W} )  \label{eq:sim_f}\\
        \yb &= \fb + \epsilon  \,\,,\,\, \epsilon \sim \mathcal{N}(0,\sigma_n^2) \label{eq:sim_y}
        """
        self.noise_dist = tfp.distributions.Normal

        self.X = tf.cast(
            tfp.distributions.Uniform(low=-1, high=1).sample(
                sample_shape=(self.N, self.D), seed=self.seed
            ),
            dtype=self.dtype,
        )
        self.XW = tf.linalg.matmul(self.X, self.W)
        self.phi = tf.math.sin(self.XW)
        self.f = tf.linalg.matmul(self.phi, self.w)
        self.signal_var = np.var(self.f)
        self.noise_var = self.signal_var / self.snr
        self.noise_dist = self.tfp.distributions.Normal(
            loc=0, scale=tf.math.sqrt(self.noise_var)
        )
        noise_samples = tf.cast(
            self.noise_dist.sample(sample_shape=(self.N, 1), seed=self.seed),
            dtype=self.dtype,
        )
        self.y = self.f + noise_samples
        self.ne_true = -self.noise_dist.entropy()

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.indices_train,
            self.indices_test,
        ) = train_test_split(
            self.X.numpy(),
            self.y.numpy(),
            np.arange(self.n_train + self.n_test),
            n_train=self.n_train,
            random_state=self.seed,
        )

        self.summary.update({"true_nentropy": self.ne_true})
        self.X_train_tf = tf.convert_to_tensor(self.X_train, dtype=self.dtype)
        self.y_train_tf = tf.convert_to_tensor(self.y_train, dtype=self.dtype)
        self.X_test_tf = tf.convert_to_tensor(self.X_test, dtype=self.dtype)
        self.y_test_tf = tf.convert_to_tensor(self.y_test, dtype=self.dtype)
