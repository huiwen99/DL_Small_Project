import argparse
from model import *
from dataloader import *
from dataset import *
from utils import plot_curves
import torch
import torch.optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

# arguments to command line
parser = argparse.ArgumentParser(description="Train model")
parser.add_argument("classifier", type=str, help="set type of classifier")
parser.add_argument("--epochs", type=int, default=10, help="set epochs")
parser.add_argument("--batch", type=int, default=16, help="set batch size")
parser.add_argument("--lr", type=float, default=0.001, help="set learning rate")
parser.add_argument("--beta1", type=float, default=0.9, help="set the first momentum term")
parser.add_argument("--beta2", type=float, default=0.999, help="set the second momentum term")
parser.add_argument("--weight_decay", type=float, default=1e-5, help="set weight decay")
parser.add_argument("--gamma", type=float, default=0.7, help="set learning rate gamma")
parser.add_argument("--step_size", type=int, default=1, help="set scheduler step size")
parser.add_argument("--cuda", type=bool, default=True, help="enable cuda training")
parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint to load model")
parser.add_argument("--save_dir", type=str, default=None, help="file path to save the model")

# get arugments
args = parser.parse_args()
model_name = args.classifier
batch_size = args.batch
epochs = args.epochs
learning_rate = args.lr
beta1 = args.beta1
beta2 = args.beta2
weight_decay = args.weight_decay
gamma = args.gamma
step_size = args.step_size
cuda = args.cuda
save_dir = args.save_dir
checkpoint = args.checkpoint

# set cpu / gpu
use_cuda = cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# load dataset and define model
if model_name == "binary_classifier_1":
    train_dset = Lung_Train_Dataset_BC1()
    ld_train = load_train(train_dset, batch_size)
    ld_test = load_test(Lung_Test_Dataset_BC1(), batch_size)
    model = Normal_VS_Infected().to(device)

    # weights for loss function
    normal_class = train_dset.dataset_numbers['normal']
    infected_class = train_dset.dataset_numbers['non-covid']+train_dset.dataset_numbers['covid']
    weights = torch.tensor([1./normal_class, 1./infected_class]).to(device)
    
elif model_name == "binary_classifier_2":
    train_dset = Lung_Train_Dataset_BC2()
    ld_train = load_train(train_dset, batch_size)
    ld_test = load_test(Lung_Test_Dataset_BC2(), batch_size)
    model = NonCovid_VS_Covid().to(device)

    # weights for loss function
    noncovid_class = train_dset.dataset_numbers['non-covid']
    covid_class = train_dset.dataset_numbers['covid']
    weights = torch.tensor([1./noncovid_class, 1./covid_class]).to(device)

# else:
#     ld_train = load_train(Lung_Train_Dataset_3CC(), batch_size)
#     ld_test = load_test(Lung_Test_Dataset_3CC(), batch_size)
#     model = ThreeClassesClassifier().to(device)

# initialize model and optimizer
if checkpoint:
    load_model(model_name, checkpoint)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

# train the model
train_losses = []
train_accs = []
train_f1s = []
test_losses = []
test_accs = []
test_f1s = []
print("Training...")
for epoch in range(1, epochs + 1):
    train_loss, train_acc, train_f1, test_loss, test_acc, test_f1 = train(model, device, ld_train, ld_test, optimizer, epoch, weights)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    train_f1s.append(train_f1)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    test_f1s.append(test_f1)

    scheduler.step()
    
    # save best checkpoint based on test F1 score
    if epoch == 1:
        best_test_score = test_f1
    
    if save_dir:
        if test_f1 < best_test_score:
            torch.save(model.state_dict(), save_dir)   
            
# plot learning curves
plot_curves(train_losses, test_losses, "Loss curves")
plot_curves(train_accs, test_accs, "Accuracy curves")
plot_curves(train_f1s, test_f1s, "F1 curves")
