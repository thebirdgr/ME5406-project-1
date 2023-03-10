import matplotlib.pyplot as plt
import csv
import time 
import numpy as np 
from datetime import datetime
import os
import re
import math
import pandas as pd

directory = './csv/10-decay/'
files_list = []


for filename in os.scandir(directory):
    if filename.is_file():
        if (re.findall("10", filename.path)):
            files_list.append(filename)


x_r_list = []
y_r_list = []
labels_r = []

for file in files_list:
    x_r = []
    y_r = []
    r = file.name.split("-")
    labels_r.append(r[0])
    
    with open(file.path,'r') as csvfile:
        lines = csv.reader(csvfile, delimiter=',')
        r_label = ''
        for row in lines:
            if row:
                # if(r[0] == "sarsa"):
                #     print(math.copysign(1, float(row[1])))
                x_r.append(float(row[0]))
                y_r.append(float(row[1]))
    
    x_r_list.append(x_r)
    y_r_list.append(y_r)
    

for i in range(len(labels_r)):
    plt.plot(x_r_list[i], y_r_list[i], label = labels_r[i].upper())


plt.rcParams["figure.figsize"] = (30,20)
ax = plt.gca()
N = 20

xmin, xmax = ax.get_xlim()
custom_ticks = np.linspace(xmin, xmax, N, dtype=int)
ax.set_xticks(custom_ticks)
ax.set_xticklabels(custom_ticks)

ymin, ymax = ax.get_ylim()
custom_ticks = np.linspace(ymin, ymax, N, dtype=int)
ax.set_yticks(custom_ticks)
ax.set_yticklabels(custom_ticks)

leg = plt.legend(loc='upper left')
title = "10x10 Cumulative Rewards Comparison With Decay"
plt.title(title, fontsize = 30)
plt.xlabel("Number of Episodes", fontsize = 20)
plt.ylabel("Cumulative Rewards", fontsize = 20)
plt.savefig("./Graphs/10-decay-convergence.png")
plt.show()