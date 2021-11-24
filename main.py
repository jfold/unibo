from imports.general import *
from src.experiment import Experiment
from src.parameters import Parameters


def run():
    start = time.time()
    try:
        args = sys.argv[-1].split("|")
    except:
        args = []
    print("------------------------------------")
    print("RUNNING EXPERIMENT...")
    kwargs = {}
    parameters = Parameters()
    for arg in args:
        try:
            var = arg.split("=")[0]

            if isinstance(type(getattr(parameters, var)), int):
                val = int(arg.split("=")[1])
            elif isinstance(type(getattr(parameters, var)), bool):
                val = arg.split("=")[1].lower() == "true"
            elif isinstance(type(getattr(parameters, var)), float):
                val = float(arg.split("=")[1])
            elif isinstance(type(getattr(parameters, var)), str):
                val = arg.split("=")[1]
            else:
                print("COULD NOT FIND VARIABLE:", var)
            kwargs.update({var: val})
        except:
            if "main.py" not in args:
                print("Trouble with " + arg)
    parameters.update(kwargs)
    parameters.save()
    experiment = Experiment(parameters)
    experiment.demo()
    print("FINISHED EXPERIMENT")
    print("------------------------------------")


if __name__ == "__main__":
    run()
