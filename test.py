import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from DataLoader import DataLoader
from EMR_fitting import EMR_fitting

def test_DataLoader():
    dataloader = DataLoader("APT_479")
    # print(dataloader.seq_dict)
    # dataloader.get_WASSR_Zspec()
    dataloader.read_Zspec(1.5)

def test_w1(): 
    w1 = 267.522 * 1.5
    f = w1 * w1 * math.pi
    print(f)

def test_dict():
    result = {"a":1, "b":2, "c":3}
    print(list(result.values()))

def test_magnitude():
    N_line = []
    ratio_line = []
    w1 = 267.522 * 1.5
    # print("w1/2pi:", w1/(2*math.pi))
    dataloader = DataLoader("APT_479")
    offset = DataLoader.Zspec_offset.wide_14
    fitting = EMR_fitting(1.5, 1, 0.1)
    Rrfm_line = fitting.cal_Rrf(offset*128, 10e-6, "G")
    for i, x in enumerate(offset):
        x *= 128
        val = w1/(2*math.pi*x)
        N_line.append(val*val)
        ratio_line.append(Rrfm_line[i] / (val*val))
        print(x/128, Rrfm_line[i], val*val, Rrfm_line[i] / (val*val))
    plt.scatter(offset, Rrfm_line, color="red")
    plt.show()
    plt.scatter(offset, N_line, color="blue")
    plt.show()
    plt.scatter(offset, ratio_line, color="green")
    plt.show()
    
test_DataLoader()       






