from attr import has
from src.parameters import Parameters
from .evalset import test_funcs
import inspect
from imports.general import *
from imports.ml import *

#Updated on 19/01 by Mikkel to have 1 additional dataset - we need to have test, pool and valid set (init set is sampled from pool set).

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
        self.n_validation = parameters.n_validation
        self.n_initial = parameters.n_initial
        self.maximize = parameters.maximization
        self.n_pool = parameters.n_pool
        self.real_world = False
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

        self.sample_initial_dataset()

    def sample_initial_dataset(self) -> None:
        self.X_pool, self.y_pool, self.f_pool = self.sample_data(
            n_samples=self.n_pool, first_time=True, test_set=False
        )

        self.X_test, self.y_test, self.f_test = self.sample_data(
            n_samples=self.n_test, first_time=True, test_set=True
        )

        init_indexes = np.random.permutation(len(self.X_pool))[:self.n_initial]
        self.X_train = self.X_pool[init_indexes]
        self.y_train = self.y_pool[init_indexes]
        self.f_train = self.f_pool[init_indexes]


        self.X_pool = np.delete(self.X_pool, init_indexes, axis=0)
        self.y_pool = np.delete(self.y_pool, init_indexes, axis=0)
        self.f_pool = np.delete(self.f_pool, init_indexes, axis=0)

        self.X_val, self.y_val, self.f_val = self.sample_data(
            n_samples=self.n_validation
        )

    def compute_set_scaling_properties(self, X: np.ndarray, f: np.ndarray) -> None:
        self.X_mean_pool_scaling = np.mean(X, axis=0)
        self.X_std_pool_scaling = np.std(X, axis=0)
        self.f_mean_pool_scaling = np.mean(f)
        self.f_std_pool_scaling = np.std(f)
        self.signal_std_pool = np.std(f)
        self.noise_std = np.sqrt(self.signal_std_pool ** 2 / self.snr)
        self.ne_true = -norm.entropy(loc=0, scale=self.noise_std)
        self.y_mean_pool_scaling = self.f_mean_pool_scaling
        self.y_std_pool_scaling = np.sqrt(self.f_std_pool_scaling ** 2 + self.noise_std ** 2)

    def standardize(
        self, X: np.ndarray, y: np.ndarray, f: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = (X - self.X_mean_pool_scaling) / self.X_std_pool_scaling  # (f - self.X_mean) / np.max(np.abs(X))  #
        f = (f - self.f_mean_pool_scaling) / self.f_std_pool_scaling  # (f - self.f_mean) / np.max(np.abs(f))  #
        y = (y - self.y_mean_pool_scaling) / self.y_std_pool_scaling  # (f - self.f_mean) / np.max(np.abs(f))  #
        return X, y, f

    def sample_data(
        self, n_samples: int = 1, first_time: bool = False, test_set: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # sample X
        X = np.random.uniform(low=self.x_lbs, high=self.x_ubs, size=(n_samples, self.d))

        # query f
        f = []
        for x in X:
            f.append(self.problem.evaluate(x))
        f = np.array(f)

        if first_time and not test_set:
            self.compute_set_scaling_properties(X, f)

        noise = np.random.normal(loc=0, scale=self.noise_std, size=f.shape)
        y = f + noise

        X, y, f = self.standardize(X, y, f)

        

        if first_time and test_set:
            self.y_min_idx_test = np.argmin(y)
            self.y_min_loc_test = X[self.y_min_idx_test, :]
            self.y_min_test = y[self.y_min_idx_test]
            self.y_max_test = np.max(y)
            self.f_min_idx_test = np.argmin(f)
            self.f_min_loc_test = X[self.f_min_idx_test, :]
            self.f_min_test = f[self.f_min_idx_test]
            self.f_max_test = np.max(f)
            self.X_mean_test = np.mean(X, axis=0)
        elif first_time and not test_set:
            self.y_min_idx_pool = np.argmin(y)
            self.y_min_loc_pool = X[self.y_min_idx_pool, :]
            self.y_min_pool = y[self.y_min_idx_pool]
            self.y_max_pool = np.max(y)
            self.f_min_idx_pool = np.argmin(f)
            self.f_min_loc_pool = X[self.f_min_idx_pool, :]
            self.f_min_pool = f[self.f_min_idx_pool]
            self.f_max_pool = np.max(f)
            self.X_mean_pool = np.mean(X, axis=0)
            self.y_mean_pool = np.mean(y, axis=0)
            self.X_std_pool = np.std(X, axis=0)
            self.y_std_pool = np.std(y, axis=0)
        return X, y[:, np.newaxis], f[:, np.newaxis]

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

