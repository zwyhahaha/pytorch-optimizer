import numpy as np
import pandas as pd
import argparse
import csv
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

parser = argparse.ArgumentParser(description='Plotting for hypergradient descent PyTorch tests', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir', help='directory to read the csv files written by train.py', default='results', type=str)
parser.add_argument('--plotDir', help='directory to save the plots', default='plots', type=str)
opt = parser.parse_args()

os.makedirs(opt.plotDir, exist_ok=True)



files = [
    'results/mlp/+1e-02/osmm.csv',
    'results/mlp/+1e-03/sgdn.csv',
    'results/mlp/adam_+1e-05_+1e-02_False.csv',
    'results/mlp/adam_+1e-04_+1e-07.csv',
    'results/mlp/adam.csv',
    # 'results/mlp/osmm_+1e-02_+1e-01.csv',
    # 'results/mlp/osmm_+1e-02_+1e-03.csv',
]

files = [
    'results/logreg/+1e-02/osmm.csv',
    'results/logreg/osmm_+5e-02_+1e-01_False.csv',
    'results/logreg/+1e-03_+1e-07/adam.csv',
    'results/logreg/adam_+1e-04_+1e-01_False.csv',
]

data = {}
data_epoch = {}
selected = []
for fname in files:
    # Extract the part of the path after 'logreg/'
    name = '/'.join(fname.split('/')[2:4])
    data[name] = pd.read_csv(fname)
    data_epoch[name] = data[name][pd.notna(data[name].LossEpoch)]
    selected.append(name)
fig = plt.figure(figsize=(5, 12))
ax = fig.add_subplot(311)

for name in selected:
    plt.plot(data_epoch[name].Epoch, data_epoch[name].LossEpoch, label=name)
plt.yscale('log')
plt.ylabel('Training loss')
plt.tick_params(labeltop=False, labelbottom=False, bottom=False, top=False, labelright=False)
plt.grid()
inset_axes(ax, width="50%", height="35%", loc=1)
for name in selected:
    plt.plot(data[name].Iteration, data[name].Loss,label=name)
# plt.yticks(np.arange(-0.01, 0.051, 0.01))
plt.xlabel('Iteration')
plt.ylabel('Training loss')
plt.xscale('log')
# plt.xlim([0,9000])
plt.grid()

ax = fig.add_subplot(312)
for name in selected:
    plt.plot(data_epoch[name].Epoch, data_epoch[name].Beta, label=name)
plt.ylabel('Beta')
plt.tick_params(labeltop=False, labelbottom=False, bottom=False, top=False, labelright=False)
plt.grid()
inset_axes(ax, width="50%", height="35%", loc=1)
for name in selected:
    plt.plot(data[name].Iteration, data[name].Beta,label=name)
plt.xlabel('Iteration')
plt.ylabel('Beta')
plt.xscale('log')
# plt.xlim([0,9000])
plt.grid()

plt.legend()
plt.savefig('{}/{}.pdf'.format('plots', 'logreg_test'), bbox_inches='tight')