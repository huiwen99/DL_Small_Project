import argparse
from model import *
from dataset import *
import torch
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

# arguments to command line
parser = argparse.ArgumentParser(description="Evaluate model")
parser.add_argument("classifier", type=str, help="set type of classifier")
parser.add_argument("--checkpoint", default=[None, None], nargs='+', help="checkpoint to load model")

# get arugments
args = parser.parse_args()
model_name = args.classifier
checkpoint = args.checkpoint

# set cpu
device = torch.device("cpu")

# set batch size
batch_size = 1

# load dataset and define model
if model_name == "two_binary_classifiers":
    ld_val = DataLoader(Lung_Val_Dataset_3CC(), batch_size=batch_size, shuffle=False)
    model = TwoBinaryClassifiers().to(device)
    if checkpoint[0]:
        model.bc1 = load_model("binary_classifier_1", checkpoint[0])
    if checkpoint[1]:
        model.bc2 = load_model("binary_classifier_2", checkpoint[1])
else:
    if model_name == "binary_classifier_1":
        ld_val =  DataLoader(Lung_Val_Dataset_BC1(), batch_size=batch_size, shuffle=False)
        model = Normal_VS_Infected().to(device)
    elif model_name == "binary_classifier_2":
        ld_val = DataLoader(Lung_Val_Dataset_BC2(), batch_size=batch_size, shuffle=False)
        model = NonCovid_VS_Covid().to(device)

    if checkpoint[0]:
        model = load_model(model_name, checkpoint[0])

# evaluate on test model
display_performance(model, device, ld_val)
