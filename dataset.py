# Matplotlib
import matplotlib.pyplot as plt
# Numpy
import numpy as np
# Pillow
from PIL import Image
# Torch
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class Lung_Dataset(Dataset):
    """
    Generic Dataset class. Used as parent class for each subgroup (train, test, val).
    """

    def __init__(self):
        """
        Constructor for generic Dataset class - simply assembles important parameters in attributes.
        """

        # All images are of size 150 x 150
        self.img_size = (150,150)

        # 3 classes: normal, infected (covid) and infected (non-covid)
        self.classes = {0: 'normal', 1:'covid', 2:'non-covid'}

        # Dataset should belong to only one group: train, test or val
        self.groups = None

        # Number of images for each class in the dataset
        self.dataset_numbers = {'normal': 0,\
                                'covid': 0,\
                                'non-covid': 0}

        # Path to images in the dataset
        self.dataset_paths = {'normal': None,\
                              'covid': None,\
                              'non-covid': None}


    def describe(self):
        """
        Descriptor function.
        Will print details about the dataset when called.
        """

        msg = "This is the {} dataset of the Lung Dataset".format(self.groups)
        msg += "used for the Small Project in the  50.039 Deep Learning class. \n"
        msg += "It contains a total of {} images, ".format(sum(self.dataset_numbers.values()))
        msg += "of size {} by {}.\n".format(self.img_size[0], self.img_size[1])
        msg += "The images are stored in the following locations "
        msg += "and each one contains the following number of images:\n"
        for key, val in self.dataset_paths.items():
            msg += " - {}, in folder {}: {} images.\n".format(key, val, self.dataset_numbers[key])
        print(msg)

    def open_img(self, class_val, index_val):
        """
        Opens image with specified parameters.

        Parameters:
            - class_val should be set to 'normal' or 'covid' or 'non-covid'.
            - index_val should be an integer with values between 0 and the maximal number of images in dataset.

        Returns loaded image as normalized Numpy array.
        """

        # Asserts checking for consistency in passed parameters
        err_msg = "Error - class_val variable should be set to 'normal', 'covid' or 'non-covid'."
        assert class_val in self.classes.values(), err_msg

        max_val = self.dataset_numbers['{}'.format(class_val)]
        err_msg = "Error - index_val variable should be an integer between 0 and the maximal number of images."
        err_msg += "\n(In {}, you have {} images.)".format(class_val, max_val)
        assert isinstance(index_val, int), err_msg
        assert index_val >= 0 and index_val <= max_val, err_msg

        # Open file as before
        path_to_file = '{}/{}.jpg'.format(self.dataset_paths['{}'.format(class_val)], index_val)
        with open(path_to_file, 'rb') as f:
            im = np.asarray(Image.open(f)) / 255
        f.close()
        return im

    def show_img(self, class_val, index_val):
        """
        Opens, then displays image with specified parameters

        Parameters:
            - class_val should be set to 'normal' or 'covid' or 'non-covid'.
            - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        """

        # open image
        im = self.open_img(class_val,index_val)

        # display
        plt.imshow(im)

    def __len__(self):
        """
        Length special method, returns the number of images in dataset.
        """

        # Length function
        return sum(self.dataset_numbers.values())

    def __getitem__(self, index):
        """
        Getitem special method.

        Expects an integer value index, between 0 and len(self) - 1.

        Returns the image and its label as a one hot vector, both
        in torch tensor format in dataset.
        """

        # Get item special method
        first_val = int(list(self.dataset_numbers.values())[0])
        second_val = int(list(self.dataset_numbers.values())[1]) + first_val
        if index < first_val:
            class_val = 'normal'
            label = torch.Tensor([1, 0, 0])
        elif index < second_val:
            class_val = 'covid'
            index -= first_val
            label = torch.Tensor([0, 1, 0])
        else:
            class_val = 'non-covid'
            index -= second_val
            label = torch.Tensor([0, 0, 1])
        im = self.open_img(self.groups, class_val, index)
        im = transforms.functional.to_tensor(np.array(im)).float()
        return im, label


class Lung_Train_Dataset(Lung_Dataset):
    """
       Specific Dataset class for training dataset.
    """

    def __init__(self):
        super().__init__()

        # Dataset consists of training images
        self.groups = 'train'

        # Number of images for each class in the dataset
        self.dataset_numbers = {'normal': 1341, \
                                'covid': 1345, \
                                'non-covid': 2530}

        # Path to images in the dataset
        self.dataset_paths = {'normal': './dataset/train/normal/', \
                              'covid': './dataset/train/infected/covid', \
                              'non-covid': './dataset/train/infected/non-covid'}


    def data_augmentation(self):
        """TODO"""
        pass

class Lung_Test_Dataset(Lung_Dataset):
    """
       Specific Dataset class for testing dataset.
    """

    def __init__(self):
        super().__init__()

        # Dataset consists of testing images
        self.groups = 'test'

        # Number of images for each class in the dataset
        self.dataset_numbers = {'normal': 234, \
                                'covid': 138, \
                                'non-covid': 242}

        # Path to images in the dataset
        self.dataset_paths = {'normal': './dataset/test/normal/', \
                              'covid': './dataset/test/infected/covid', \
                              'non-covid': './dataset/test/infected/non-covid'}

class Lung_Val_Dataset(Lung_Dataset):
    """
       Specific Dataset class for validation dataset.
    """

    def __init__(self):
        super().__init__()

        # Dataset consists of validation images
        self.groups = 'val'

        # Number of images for each class in the dataset
        self.dataset_numbers = {'normal': 8, \
                                'covid': 8, \
                                'non-covid': 8}

        # Path to images in the dataset
        self.dataset_paths = {'normal': './dataset/val/normal/', \
                              'covid': './dataset/val/infected/covid', \
                              'non-covid': './dataset/val/infected/non-covid'}

