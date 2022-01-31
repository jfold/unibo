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
        np.random.seed(self.seed)
        self.ne_true = np.nan
        self.benchmarks = test_funcs
        all_problems = inspect.getmembers(self.benchmarks)
        if parameters.problem not in [a for a, b in all_problems]:
            raise ValueError(f"Could not find problem: {parameters.problem}")
        self.benchmark_tags = {}
        for name, obj in inspect.getmembers(self.benchmarks):
            if inspect.isclass(obj):
                try:
                    self.benchmark_tags.update({name: obj(dim=self.d).classifiers})
                except:
                    pass
        if parameters.problem in self.benchmark_tags:
            self.problem = getattr(self.benchmarks, parameters.problem)(dim=self.d)
            self.y_max = np.maximum(
                np.abs(self.problem.fmax), np.abs(self.problem.fmin)
            )
            self.f_max = self.problem.fmax / self.y_max
            self.f_min = self.problem.fmin / self.y_max
            self.lbs = [b[0] for b in self.problem.bounds]
            self.ubs = [b[1] for b in self.problem.bounds]

            self.signal_var = self.get_signal_var()
            self.noise_var = self.signal_var / self.snr
            self.X = self.sample_X(parameters.n_initial)
            self.y = self.get_y(self.X)
        else:
            raise ValueError(
                f"Problem {parameters.problem} does not support dimensionality {self.d}"
            )

    def get_signal_var(self, n_samples: int = 3000) -> float:
        X = self.sample_X(n_samples=n_samples)
        y = self.get_y(X, add_noise=False)
        return np.var(y)

    def sample_X(self, n_samples: int = 1) -> np.ndarray:
        X = np.random.uniform(low=self.lbs, high=self.ubs, size=(n_samples, self.d))
        return X

    def get_y(self, X: np.ndarray, add_noise: bool = True) -> np.ndarray:
        y_new = []
        for x in X:
            y_new.append(self.problem.evaluate(x) / self.y_max)
        y_arr = np.array(y_new)
        if self.noisify and add_noise:
            noise_samples = np.random.normal(
                loc=0, scale=np.sqrt(self.noise_var), size=y_arr.shape
            )
            y_arr += noise_samples
        return y_arr[:, np.newaxis]

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

