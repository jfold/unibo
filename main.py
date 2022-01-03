from imports.general import *
from src.experiment import Experiment
from src.parameters import Parameters

# PLAN:
# -1) Save with less decimals
# 0) Unittest validation of input/output
# 1) Best practice in hyperparameter tuning
# 2) Define experimental grid from Dewancker et al. 2016
# 3) Sharpness: how fast does the mean (+ variance) of negative entropy converge?


def run():
    start = time.time()
    try:
        args = sys.argv[-1].split("|")
    except:
        args = []
    print("------------------------------------")
    print("RUNNING EXPERIMENT...")
    kwargs = {}
    parameters = Parameters(mkdir=False)
    for arg in args:
        try:
            var = arg.split("=")
            print(getattr(parameters, var))
            print(type(getattr(parameters, var)))
            if isinstance(type(getattr(parameters, var)), int):
                val = int(arg.split("=")[1])
            elif isinstance(type(getattr(parameters, var)), bool):
                val = arg.split("=")[1].lower() == "true"
            elif isinstance(type(getattr(parameters, var)), float):
                val = float(arg.split("=")[1])
            elif isinstance(type(getattr(parameters, var)), str):
                val = arg.split("=")[1]
            else:
                var = None
                print("COULD NOT FIND VARIABLE:", var)
            kwargs.update({var: val})
        except:
            if "main.py" not in args:
                print("Trouble with " + arg)
    parameters = Parameters(mkdir=True)
    parameters.update(kwargs, save=True)
    experiment = Experiment(parameters)
    experiment.run()
    print("FINISHED EXPERIMENT")
    print("------------------------------------")


if __name__ == "__main__":
    run()
