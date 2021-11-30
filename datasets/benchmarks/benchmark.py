from src.parameters import Parameters
from .evalset import test_funcs
import inspect
from imports.general import *
from imports.ml import *


class Benchmark(object):
    """ 
    ##### http://infinity77.net/global_optimization/test_functions.html#multidimensional-test-functions-index
    """

    def __init__(self, parameters: Parameters):
        self.benchmarks = test_funcs
        self.problem = getattr(self.benchmarks, parameters.problem)(dim=parameters.d)
        self.benchmark_tags = {}
        for name, obj in inspect.getmembers(self.benchmarks):
            if inspect.isclass(obj):
                try:
                    self.benchmark_tags.update({name: obj(dim=2).classifiers})
                except:
                    pass
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        self.X_dist = tfp.distributions.Uniform(
            low=[b[0] for b in self.problem.bounds],
            high=[b[1] for b in self.problem.bounds],
        )
        self.X = self.sample_X(parameters.n_initial)
        self.y = self.sample_y(self.X)

    def sample_X(self, n_samples: int = 1) -> None:
        X = tf.cast(
            self.X_dist.sample(sample_shape=(n_samples, self.d), seed=self.seed),
            dtype=self.dtype,
        )
        return X.numpy()

    def sample_y(self, X: np.ndarray) -> None:
        y_new = []
        for x in X:
            y_new.append(self.problem.evalulate(x))
        return np.array(y_new)

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

