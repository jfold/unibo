from attr import has
from src.parameters import Parameters
from .evalset import test_funcs
import inspect
from imports.general import *
from imports.ml import *


class Benchmark(object):
    """ Benchmark dataset for bayesian optimization
    ##### http://infinity77.net/global_optimization/test_functions.html#multidimensional-test-functions-index
    """

    def __init__(self, parameters: Parameters):
        self.d = parameters.d
        self.seed = parameters.seed
        self.noisify = parameters.noisify
        self.snr = parameters.snr
        self.n_test = parameters.n_test
        np.random.seed(self.seed)
        self.benchmarks = test_funcs
        all_problems = inspect.getmembers(self.benchmarks)
        if parameters.problem not in [a for a, b in all_problems]:
            raise NameError(f"Could not find problem: {parameters.problem}")
        self.benchmark_tags = {}
        for name, obj in inspect.getmembers(self.benchmarks):
            if inspect.isclass(obj):
                try:
                    self.benchmark_tags.update({name: obj(dim=self.d).classifiers})
                except:
                    pass

        if parameters.problem not in self.benchmark_tags:
            raise ValueError(
                f"Problem {parameters.problem} does not support dimensionality {self.d}"
            )

        self.problem = getattr(self.benchmarks, parameters.problem)(dim=self.d)
        self.x_lbs = np.array([b[0] for b in self.problem.bounds])
        self.x_ubs = np.array([b[1] for b in self.problem.bounds])
        self.X_mean = None
        self.f_mean = None
        self.y_mean = None
        self.f_min_idx = np.nan
        self.y_min_idx = np.nan
        self.ne_true = None
        self.y_max = np.maximum(np.abs(self.problem.fmax), np.abs(self.problem.fmin))
        self.f_max = self.problem.fmax
        self.f_min = self.problem.fmin
        self.y_min = self.f_min
        self.f_min_loc = np.array([self.problem.min_loc])
        self.y_min_loc = self.f_min_loc
        self.sample_testset_and_compute_data_stats()
        self.X_train, self.y_train, self.f_train = self.sample_data(
            parameters.n_initial
        )

    def sample_testset_and_compute_data_stats(self, n_samples: int = 3000) -> None:

        self.sample_data(n_samples=n_samples, first_time=True)
        self.X_test, self.y_test, self.f_test = self.sample_data(n_samples=self.n_test)

    def sample_data(
        self, n_samples: int = 1, first_time: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # sample X
        X = np.random.uniform(low=self.x_lbs, high=self.x_ubs, size=(n_samples, self.d))

        # query f
        f = []
        for x in X:
            f.append(self.problem.evaluate(x))
        f = np.array(f)

        if first_time:
            self.X_mean = np.mean(X, axis=0)
            self.X_std = np.std(X, axis=0)
            self.f_mean = np.mean(f)
            self.f_std = np.std(f)
            self.signal_std = np.std(f)
            self.noise_std = np.sqrt(self.signal_std ** 2 / self.snr)

        noise = np.random.normal(loc=0, scale=self.noise_std, size=f.shape)
        y = f + noise

        if first_time:
            self.ne_true = -norm.entropy(loc=0, scale=self.noise_std)
            self.y_mean = np.mean(y)
            self.y_std = np.std(y)

        if (
            self.X_mean is not None
            and self.f_mean is not None
            and self.y_mean is not None
        ):
            X = (X - self.X_mean) / self.X_std
            f = (f - self.f_mean) / self.f_std
            y = (y - self.y_mean) / self.y_std

        return X, y[:, np.newaxis], f[:, np.newaxis]

    # def sample_X(self, n_samples: int = 1) -> np.ndarray:
    #     X = np.random.uniform(low=self.x_lbs, high=self.x_ubs, size=(n_samples, self.d))
    #     if self.X_mean is not None:
    #         # Standardize
    #         X = (X - self.X_mean) / self.X_std

    #     return X

    # def get_y(self, X: np.ndarray, add_noise: bool = True) -> np.ndarray:
    #     f = []
    #     for x in X:
    #         f.append(self.problem.evaluate(x))
    #     f = np.array(f)

    #     if self.f_mean is not None:
    #         f = (f - self.f_mean) / self.f_std

    #     if self.noisify and add_noise:
    #         noise = np.random.normal(loc=0, scale=self.noise_std, size=f.shape)
    #         y = f + noise

    #         if self.y_mean is not None:
    #             y = y / self.noise_std
    #         return y[:, np.newaxis]

    #     return f

    def __str__(self):
        return str(self.problem)

    def problems_with_tags(self, tags: list) -> list:
        """Finds all problems containing one of tag in input
        """
        assert type(tags[0]) is str
        problems = []
        for i_t, tag in enumerate(tags):
            for p, t in self.benchmark_tags.items():
                if tag in t:
                    problems.append(p)
        problems = list(set(problems))
        return problems

    def problems_with_all_tags(self, tags: list) -> list:
        """Finds all problems containing all tags in input
        """
        assert type(tags[0]) is str
        problems = []
        for p, t in self.benchmark_tags.items():
            if set(tags) <= set(t):
                problems.append(p)
        problems = list(set(problems))
        return problems

    def problems_only_with_all_tags(self, tags: list) -> list:
        """Finds all problems containing all tags and only all in input
        """
        assert type(tags[0]) is str
        problems = []
        for p, t in self.benchmark_tags.items():
            if set(tags) == set(t):
                problems.append(p)
        problems = list(set(problems))
        return problems

