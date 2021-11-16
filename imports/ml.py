import sys

if "tensorflow" not in sys.modules:
    import tensorflow as tf
if "tensorflow_probability" not in sys.modules:
    import tensorflow_probability as tfp
from scipy.stats import norm, uniform, kstest, entropy
from sklearn.model_selection import train_test_split
