__author__ = 'AlexLlamas'
import numpy as np
import pandas as pd
from Plots import *


def data1(numSamples, noise):
    u = np.random.uniform(-10, 10, numSamples).reshape(numSamples, 1)  # uniform
    x = np.linspace(-10, 10, numSamples).reshape(numSamples, 1)        # incremental
    y1 = np.random.normal(np.sin(x) * 10, noise).reshape(numSamples, 1)     # sinusoidal of x
    y2 = np.random.normal((x-x), noise).reshape(numSamples, 1)         # constant in cero
    y3 = np.random.normal((x-u), noise).reshape(numSamples, 1)         # x - u expected uniform
    df = pd.DataFrame(np.concatenate((u, x, y1, y2, y3), axis=1),
                      columns=['u', 'x', 'y1', 'y2', 'y3'])
    return df
