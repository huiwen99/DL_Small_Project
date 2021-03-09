from dataset import *
import matplotlib.pyplot as plt
import numpy as np

def plot_class_distribution():
    """
    Plots bar graphs of the class distributions of train and test set
    over the 3 classes: normal, infected (covid) and infected (non-covid).

    There are 3 bar graphs plotted:
    - Number of images per class for each dataset (side by side)
    - Percentage of each class for train set
    - Percentage of each class for test set
    """

    # Plot number of images per class for each dataset
    train = Lung_Train_Dataset()
    test = Lung_Test_Dataset()

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
    train_num = {v / total for v in train_num}
    total = sum(test_num)
    test_num = {v / total for v in test_num}

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
    plt.show()