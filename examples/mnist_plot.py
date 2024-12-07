import matplotlib.pyplot as plt
import numpy as np
import re
import os

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def read_data(fdir,filename):
    train_losses = []
    test_losses = []
    test_accuracies = []
    
    with open(os.path.join(fdir,filename), 'r') as f:
        for line in f:
            if line.startswith("Train Epoch:"):
                loss = float(re.search(r'Loss: (\d+\.\d+)', line).group(1))
                train_losses.append(loss)
            elif line.startswith("Test set:"):
                loss = float(re.search(r'Average loss: (\d+\.\d+)', line).group(1))
                accuracy = float(re.search(r'Accuracy: \d+/\d+ \((\d+)%\)', line).group(1))
                test_losses.append(loss)
                test_accuracies.append(accuracy)
    
    return train_losses, test_losses, test_accuracies

# Read data from all files
fdir = '/home/shanshu/pytorch-optimizer/cvg_plot/mnist_batch1000'
files = os.listdir(fdir)
files = ['Adam_lr0.01.txt','OSGM_lr1.0.txt','OSMM_lr1.0.txt',\
         'NAG_lr0.01.txt',]
files = ['Adam_lr0.01.txt','OSGM_lr0.1.txt','OSMM_lr0.1.txt',\
         'NAG_lr0.05.txt',]
labels = [f.split('_')[0] for f in files]

all_data = [read_data(fdir,file) for file in files]

# Plotting
fig, ax1 = plt.subplots()

# Training Loss
for i, (train_losses, _, _) in enumerate(all_data):
    smoothed_losses = moving_average(train_losses, 1)
    epochs = np.arange(1, len(smoothed_losses) + 1)
    ax1.plot(epochs, smoothed_losses, label=labels[i])
    # epochs = np.arange(1, len(train_losses) + 1)
    # ax1.plot(epochs[::1], train_losses[::1], label=labels[i])
ax1.set_yscale('log')
ax1.set_xlabel('Iterations', fontsize=14)
ax1.set_ylabel('Training Loss', fontsize=14)
ax1.set_title('MNIST Dataset', fontsize=16)
ax1.legend(fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)

# Adjust layout and save
plt.tight_layout()
plt.savefig('cvg_plot/mnist_batch1000.png', dpi=300, bbox_inches='tight')
