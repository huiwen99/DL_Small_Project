import argparse
from model import *
from dataloader import *
import torch
import torch.optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

# arguments to command line
parser = argparse.ArgumentParser(description="Train model")
parser.add_argument("--epochs", type=int, default=10, help="set epochs")
parser.add_argument("--batch", type=int, default=4, help="set batch size")
parser.add_argument("--lr", type=float,default=0.001, help="set learning rate")
parser.add_argument("--gamma", type=float, default=0.7, help="set learning rate gamma")
parser.add_argument("--cuda", type=bool, default=True, help="enable cuda training")
parser.add_argument("--save_dir", type=str, default=None, help="file path to save the model at")

# get arugments
args = parser.parse_args()
batch_size = args.batch
epochs = args.epochs
learning_rate = args.lr
gamma = args.gamma
cuda = args.cuda
save_dir = args.save_dir

# set cpu / gpu
use_cuda = cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# load datasets
ld_train = load_train(batch_size)
ld_val = load_val(1)
ld_test = load_test(batch_size)

# initialize model and optimizer
model = Net().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

# save checkpoint every 5 epochs
checkpoint_every = 5

# train the model
train_losses = []
train_accs = []
test_losses = []
test_accs = []
print("Training...")
for epoch in range(1, epochs + 1):
    train_loss, train_acc, test_loss, test_acc = train(model, device, ld_train, ld_test, optimizer, epoch)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    scheduler.step()

    if epoch%checkpoint_every==0:
        # save model if applicable
        if save_dir!=None:
            torch.save(model.state_dict(), save_dir)

# plot learning curves
plt.plot(train_losses, label="Train")
plt.plot(test_losses, label="Test")
plt.title("Loss curves")
plt.legend()
plt.show()

plt.plot(train_accs, label="Train")
plt.plot(test_accs, label="Test")
plt.title("Accuracy curves")
plt.legend()
plt.show()


# evaluate on test model
display_performance(model, device, ld_val)

# save model if applicable
if save_dir!=None:
    torch.save(model.state_dict(), save_dir)
