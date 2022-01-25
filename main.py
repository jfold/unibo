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
    print("Arguments:", args)
    print("RUNNING EXPERIMENT...")
    kwargs = {}
    parameters_temp = Parameters(mkdir=False)
    if args[0] != "main.py":
        for arg in args:
            var = arg.split("=")[0]
            val = arg.split("=")[1]
            par_val = getattr(parameters_temp, var)

            if isinstance(par_val, bool):
                val = val.lower() == "true"
            elif isinstance(par_val, int):
                val = int(val)
            elif isinstance(par_val, float):
                val = float(val)
            elif isinstance(par_val, str):
                pass
            else:
                var = None
                print("COULD NOT FIND VARIABLE:", var)
            kwargs.update({var: val})

    parameters = Parameters(kwargs, mkdir=True)
    print("Running with:", parameters)
    experiment = Experiment(parameters)
    experiment.run()
    print("FINISHED EXPERIMENT")
    print("------------------------------------")


if __name__ == "__main__":
    run()
