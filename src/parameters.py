import json
import string
from imports.general import *
from imports.ml import *
from dataclasses import dataclass, asdict


@dataclass
class Parameters:
    surrogate: str = "GP"  # surrogate function name
    acquisition: str = "EI"  # acquisition function name
    recal_mode: str = "cv"
    data_name: str = "Benchmark"  # dataclass name
    seed: bool = 0  # random seed
    d: int = 1  # number of input dimensions
    n_test: int = 5000  # number of test samples for calibration analysis
    n_initial: int = 10  # number of starting points
    n_validation: int = 100  # number of iid samples for recalibration
    n_evals: int = 100  # number of BO iterations
    n_pool : int = 5000
    rf_cv_splits: int = 2  # number of CV splits for random forest hyperparamtuning
    vanilla: bool = False  # simplest implementation (used for test)
    plot_it: bool = False  # whether to plot during BO loop
    save_it: bool = True  # whether to save progress
    bo: bool = False  # performing bo to sample X or merely randomly sample X
    noisify: bool = True
    test: bool = True
    beta: float = 1.0 #beta value if acquisition function is UCB. Experimenting with different values seem to indicate that beta = 1 is best, but this is probably largely dependant on optim. problem. 
    recalibrate: bool = False
    analyze_all_epochs: bool = True
    extensive_metrics: bool = True
    maximization: bool = False
    fully_bayes: bool = False  # if fully bayes in BO rutine (marginalize hyperparams)
    xi: float = 0.0  # exploration parameter for BO
    problem: str = ""  # e.g. "Alpine01" # subproblem name, overwrites problem_idx
    problem_idx: int = 0
    prob_acq: bool = False  # if acqusition function should sample like a prob dist. If False, argmax is used.
    std_change: float = 1.0  # how to manipulate predictive std
    snr: float = 100.0
    sigma_data: float = None  # follows from problem
    sigma_noise: float = None  # computed as function of SNR and sigma_data
    n_calibration_bins: int = 20
    K: int = 1  # number of terms in sum for VerificationData
    b_train: int = 64 # Batch size while training NN on MNIST
    hidden_size: int = 100 # hidden layer number of neurons for NN on MNIST
    savepth: str = os.getcwd() + "/results/"
    experiment: str = ""  # folder name
    n_seeds_per_job: int = 1 #Select how many jobs to run for this particular seed. Set via input params only.
    save_scratch: bool = False #If want results saved on scratch directory.

    def __init__(self, kwargs: Dict = {}, mkdir: bool = False) -> None:
        self.update(kwargs)
        if self.surrogate == "RS" and (self.recalibrate or self.bo):
            sys.exit(0)
        self.acquisition = "RS" if self.surrogate == "RS" else self.acquisition

        if self.problem == "" and self.data_name.lower() == "benchmark":
            problem = self.find_benchmark_problem_i()
            kwargs["problem"] = problem
            kwargs['savepth'] = "./results_synth_data/"

        elif self.data_name.lower() == "mnist":
            kwargs["problem"] = "mnist"
            kwargs["d"] = 5
            kwargs['savepth'] = "./results_real_data/results_mnist/"
        elif self.data_name.lower() == "fashionmnist":
            kwargs["problem"] = "fashionmnist"
            kwargs["d"] = 5
            kwargs['savepth'] = "./results_real_data/results_FashionMNIST/"
        elif self.data_name.lower() == "fashionmnist_cnn":
            kwargs["problem"] = "fashionmnist_cnn"
            kwargs["d"] = 5
            kwargs['savepth'] = "./results_real_data/results_FashionMNIST_CNN/"
        elif self.data_name.lower() == "mnist_cnn":
            kwargs["problem"] = "mnist_cnn"
            kwargs["d"] = 5
            kwargs['savepth'] = "./results_real_data/results_MNIST_CNN/"
        elif self.data_name.lower() == "news":
            kwargs["problem"] = "news"
            kwargs["d"] = 5
            kwargs['savepth'] = "./results_real_data/results_News/"
        elif self.data_name.lower() == "svm_wine":
            kwargs["problem"] = "svm_wine"
            kwargs["d"] = 2
            kwargs['savepth'] = "./results_real_data/results_SVM/"
        if self.save_scratch:
            kwargs['savepth'] = kwargs['savepth'].replace(".", "/work3/mikkjo/unibo_results")
        self.update(kwargs)

        if mkdir and not os.path.isdir(self.savepth):
            os.mkdir(self.savepth)
        if self.test:
            folder_name = f"test{self.experiment}"
        else:
            folder_name = (
                f"{self.experiment}--{datetime.now().strftime('%d%m%y-%H%M%S')}"
                + "--"
                + f"seed-{self.seed}"
                + "--"
                + "".join(random.choice(string.ascii_lowercase) for x in range(6))
            )
        setattr(self, "savepth", f"{self.savepth}{folder_name}/")
        if mkdir and not os.path.isdir(self.savepth):
            os.mkdir(self.savepth)
            self.save()

    def update(self, kwargs, save=False) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Parameter {key} not found")
        if save:
            self.save()

    def find_benchmark_problem_i(self) -> str:
        with open(f"datasets/benchmarks/unibo-problems.json") as json_file:
            problems = json.load(json_file)
        return problems[str(self.d)][self.problem_idx]

    def save(self) -> None:
        json_dump = json.dumps(asdict(self))
        with open(self.savepth + "parameters.json", "w") as f:
            f.write(json_dump)
