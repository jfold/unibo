import sys
import matplotlib
import matplotlib.pyplot as plt

if "tensorflow" not in sys.modules:
    import tensorflow as tf
if "tensorflow_probability" not in sys.modules:
    import tensorflow_probability as tfp
from scipy.stats import norm, uniform, kstest, entropy
from sklearn.model_selection import train_test_split


matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["axes.grid"] = True
matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["figure.figsize"] = (10, 6)
matplotlib.rcParams["savefig.bbox"] = "tight"
