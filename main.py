from .imports.general import *
from .src.experiment import Experiment
from .src.parameters import Parameters


def run():
    start = time.time()
    try:
        args = sys.argv[-1].split("|")
    except:
        args = []
    print("------------------------------------")
    print("RUNNING EXPERIMENT...")
    kwargs = {"savepth": os.getcwd() + "/results/"}
    parameters = Parameters(**kwargs)
    for arg in args:
        try:
            var = arg.split("=")[0]

            if type(getattr(parameters, var)) is int:
                val = int(arg.split("=")[1])
            elif type(getattr(parameters, var)) is bool:
                val = arg.split("=")[1].lower() == "true"
            elif type(getattr(parameters, var)) is float:
                val = float(arg.split("=")[1])
            elif type(getattr(parameters, var)) is str:
                val = arg.split("=")[1]
            else:
                print("COULD NOT FIND VARIABLE:", var)
            kwargs.update({var: val})
        except:
            if "main.py" not in args:
                print("Trouble with " + arg)
    parameters = Parameters(**kwargs)
    experiment = Experiment(parameters)
    experiment.run()
    print("FINISHED EXPERIMENT")
    print("------------------------------------")


if __name__ == "__main__":
    run()
