from torch.utils.data import DataLoader
from dataset import *

def load_train(batch_size):
    """
    Returns a DataLoader from train dataset
    """

    ld_train = Lung_Train_Dataset()
    train_loader = DataLoader(ld_train, batch_size=batch_size, shuffle=True)
    return train_loader

def load_train(batch_size):
    """
    Returns a DataLoader from test dataset
    """

    ld_test = Lung_Test_Dataset()
    test_loader = DataLoader(ld_test, batch_size=batch_size, shuffle=True)
    return test_loader

def load_val(batch_size):
    """
    Returns a DataLoader from validation dataset
    """

    ld_val = Lung_Val_Dataset()
    val_loader = DataLoader(ld_val, batch_size=batch_size, shuffle=True)
    return val_loader


