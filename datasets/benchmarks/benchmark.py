from ..benchmarks.evalset import test_funcs
import inspect


class Benchmark(object):
    """ 
    ##### http://infinity77.net/global_optimization/test_functions.html#multidimensional-test-functions-index
    """

    def __init__(self, subproblem: str, dim: int):
        self.benchmarks = test_funcs
        self.problem = getattr(self.benchmarks, subproblem)(dim=dim)
        count = 0
        self.benchmark_tags = {}
        for name, obj in inspect.getmembers(self.benchmarks):
            if inspect.isclass(obj):
                try:
                    self.benchmark_tags.update({name: obj(dim=2).classifiers})
                except:
                    pass

    def __str__(self):
        return str(self.problem)

    def problems_with_tags(self, tags: list) -> list:
        """Finds all problems containing one of tag in input

        Args:
            tags (list): [list of tags can be: ]

        Returns:
            list (str): [list of problems]
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

        Args:
            tags (list): [list of tags can be: ]

        Returns:
            list (str): [list of problems]
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

        Args:
            tags (list): [list of tags can be: ]

        Returns:
            list (str): [list of problems]
        """
        assert type(tags[0]) is str
        problems = []
        for p, t in self.benchmark_tags.items():
            if set(tags) == set(t):
                problems.append(p)
        problems = list(set(problems))
        return problems

