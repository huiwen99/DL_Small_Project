from torch.utils.data import DataLoader

def load_train(ld_train, batch_size):
    """
    Returns a DataLoader from train dataset
    """

    train_loader = DataLoader(ld_train, batch_size=batch_size, shuffle=True)
    return train_loader

def load_test(ld_test, batch_size):
    """
    Returns a DataLoader from test dataset
    """

    test_loader = DataLoader(ld_test, batch_size=batch_size, shuffle=True)
    return test_loader

def load_val(ld_val, batch_size):
    """
    Returns a DataLoader from validation dataset
    """
    
    val_loader = DataLoader(ld_val, batch_size=batch_size, shuffle=False)
    return val_loader
