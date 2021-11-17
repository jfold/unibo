from helpers.other import *
from helpers.ml import *


class VerificationData(object):
    """Linear sum activation data generation class. Must be inherited by CalibrationBase class"""

    def __init__(self, noise_dist: str = "Normal", snr: float = 1.0, K: int = 1):
        self.noise_dist = noise_dist
        self.snr = snr
        self.K = K
        self.summary.update(
            {"K": self.K, "SNR": self.snr, "noise_dist_name": self.noise_dist_name}
        )

    def generate_data(self):
        """Method generates data according to
        \fb &= \phi(\Xb)\textbf{w} = \sum_{k=1}^K w_k \phi_k(\Xb\textbf{W} )  \label{eq:sim_f}\\
        \yb &= \fb + \epsilon  \,\,,\,\, \epsilon \sim \mathcal{N}(0,\sigma_n^2) \label{eq:sim_y}
        """
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        self.noise_dist = getattr(tfp.distributions, self.noise_dist_name)
        self.sigma_0 = 5

        self.X = tf.cast(
            tfp.distributions.Uniform(low=-1, high=1).sample(
                sample_shape=(self.N, self.D), seed=self.seed
            ),
            dtype=self.dtype,
        )
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
        self.XW = tf.linalg.matmul(self.X, self.W)
        self.phi = tf.math.sin(self.XW)
        self.f = tf.linalg.matmul(self.phi, self.w)
        self.signal_var = np.var(self.f)
        self.noise_var = self.signal_var / self.snr
        if self.noise_dist_name == "Normal":
            self.noise_dist = self.noise_dist(loc=0, scale=tf.math.sqrt(self.noise_var))
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
            np.arange(self.N),
            n_train=self.n_train,
            random_state=self.seed,
        )

        self.N_train, self.N_test = self.X_train.shape[0], self.X_test.shape[0]
        self.summary.update({"true_nentropy": self.ne_true})

        self.X_train_tf = tf.convert_to_tensor(self.X_train, dtype=self.dtype)
        self.y_train_tf = tf.convert_to_tensor(self.y_train, dtype=self.dtype)
        self.X_test_tf = tf.convert_to_tensor(self.X_test, dtype=self.dtype)
        self.y_test_tf = tf.convert_to_tensor(self.y_test, dtype=self.dtype)
