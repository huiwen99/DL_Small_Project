from dataset import *
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_class_distribution():
    """
    Plots bar graphs of the class distributions of train and test set
    over the 3 classes: normal, infected (covid) and infected (non-covid).

    There are 5 bar graphs plotted:
    - Number of images per class for each dataset (side by side)
    - Percentage of per class for train set
    - Percentage of each class for test set
    - Percentage of normal vs infected for train set
    - Percentage of normal vs infected for test set
    """

    # Plot number of images per class for each dataset
    train = Lung_Train_Dataset_3CC()
    test = Lung_Test_Dataset_3CC()

    labels = train.dataset_numbers.keys()
    train_num = train.dataset_numbers.values()
    test_num = test.dataset_numbers.values()
    
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, train_num, width, label='Train')
    ax.bar(x + width / 2, test_num, width, label='Test')

    ax.set_title('No. of images per class')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.show()

    # Calculating percentage of each class in each dataset
    total = sum(train_num)
    train_num = [v / total for v in train_num]
    total = sum(test_num)
    test_num = [v / total for v in test_num]

    # Plot percentage of each class for Train set
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.bar(labels, train_num, color='C0', label='Train')
    ax1.legend()
    ax1.set_ylim([0,1])
    ax1.set_title("Percentage per class (Train)")

    # Plot percentage of each class for Test set
    ax2.bar(labels, test_num, color='C1', label='Test')
    ax2.legend()
    ax2.set_ylim([0,1])
    ax2.set_title("Percentage per class (Test)")
    
    # Recalculating percentage for normal vs infected in each dataset
    labels = ['normal', 'infected']

    train_num = [train_num[0], train_num[1] + train_num[2]]
    test_num = [test_num[0], test_num[1] + test_num[2]]
    
    # Plot percentage of normal vs infected for Train set
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.bar(labels, train_num, color='C0', label='Train')
    ax1.legend()
    ax1.set_ylim([0,1])
    ax1.set_title("Percentage per class (Train)")

    # Plot percentage of normal vs infected for Test set
    ax2.bar(labels, test_num, color='C1', label='Test')
    ax2.legend()
    ax2.set_ylim([0,1])
    ax2.set_title("Percentage per class (Test)")
        
    plt.show()
    
def plot_curves(train_arrs, test_arrs, plot_name):
    """
    Plots training and testing learning curves over successive epochs
    """
    plt.plot(train_arrs, label="Train")
    plt.plot(test_arrs, label="Test")
    plt.title(plot_name)
    plt.legend()
    plt.show()