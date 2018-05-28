import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from pymc3.gp.util import plot_gp_dist


import random
np.random.seed(123)
random.seed(123)

import pymc3 as pm
import theano.tensor as tt


def parse_data(weather_data, interval_length=120, time_offset=400):
    temp = weather_data['Last Minute Average Temperature'][time_offset:time_offset+interval_length]

    # Need to figure out how to use the actual times.
    #time = weather_data['Date and Time'][time_offset:time_offset+interval_length]

    time = np.array([i for i in range(0, interval_length)])

    return time, temp


def subsample_data(x, y, subsample_size=40, cap=40):
    if cap != 1:
        x, y = x[0:-cap], y[0:-cap] # Do not sample from the last cap values

    x_sub, y_sub = zip(*random.sample(list(zip(x, y)), subsample_size))
    return np.array(x_sub), np.array(y_sub)

def subsample_np_data(x, y, subsample_size=10, start=0, from_end=0):
    x, y = x.flatten(), y.flatten()
    sub_s = np.random.choice(range(start, x.shape[0] - from_end), subsample_size)
    return x[sub_s][:, None], y[sub_s], np.sort(sub_s).tolist()

def plot_data_with_sub(x, y, x_sub, y_sub):
    plt.plot(x, y)
    plt.scatter(x_sub, y_sub)

def re_scale(x, scale=10):
    return x / float(scale)


if __name__ == "__main__":

    np.random.seed(1)

    # Load data

    # weather_data = pd.read_csv("data/Weather_Data_2017.csv")

    # x0, y0 = parse_data(weather_data)
    # x0 = re_scale(x0)
    # x_sub, y_sub = subsample_data(x0, y0, 40)

    # plot_data_with_sub(x0, y0, x_sub, y_sub)

    # plt.show()



  

