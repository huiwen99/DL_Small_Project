import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import datetime
import matplotlib.pyplot as plt

class Net(nn.Module):
    """
    The proposed deep learning model that classifies X-ray images of patients
    and help doctors with the diagnosis of COVID/non-COVID pneuomnia.
    """
    def __init__(self):
        super(Net, self).__init__()
        # Conv2D: 1 input channel, 8 output channels, 3 by 3 kernel, stride of 1.
        self.conv1 = nn.Conv2d(1, 4, 3, 1)
        self.fc1 = nn.Linear(87616, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim = 1)
        return output

def train(model, device, train_loader, test_loader, optimizer, epoch):
    """
    Trains the model on training data
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # convert one-hot to numerical categories
        target = torch.argmax(target, dim=1).long()
        optimizer.zero_grad()
        output = model(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    train_loss, train_acc = evaluate(model, device, train_loader)
    print('Train Epoch: {} @ {} \nTrain Loss: {:.4f} - Train Accuracy: {:.1f}%'.format(
        epoch, datetime.datetime.now().time(), train_loss, train_acc))

    test_loss, test_acc = evaluate(model, device, test_loader)
    print("Test Loss: {:.4f} - Test Accuracy: {:.1f}%".format(test_loss, test_acc))
    return train_loss, train_acc, test_loss, test_acc


def evaluate(model, device, data_loader):
    """
    Evaluates the model and returns loss and accuracy
    """
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            target = torch.argmax(target, dim=1).long()
            output = model(data)
            criterion = nn.CrossEntropyLoss()
            loss += criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(data_loader)
    acc = 100. * correct / len(data_loader.dataset)
    model.train()

    return loss, acc

def display_performance(model, device, data_loader):
    """
    Displays subplot containing each image in the dataset, its ground  truth label and predicted labels.
    Also displays overall accuracy.
    """
    model.eval()
    fig = plt.figure(figsize=(20,40))
    cols = 4
    rows = 6
    # loss = 0
    correct = 0
    idx = 1
    classes = {0: 'normal', 1:'covid', 2:'non-covid'}
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            target = torch.argmax(target, dim=1).long()
            output = model(data)
            # criterion = nn.CrossEntropyLoss()
            # loss += criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            fig.add_subplot(rows, cols, idx)
            idx+=1
            target = target.item()
            pred = pred.item()
            plt.title("Ground truth label: {}\nPredicted label: {}".format(classes[target], classes[pred]))
            img = data[0].cpu().numpy().reshape(150,150,1)
            plt.imshow(img)


    # loss /= len(data_loader)
    acc = 100. * correct / len(data_loader.dataset)
    plt.suptitle("Validation set pictures with predicted and ground truth labels\nAverage performance {}/{} = {:.1f}%".format(
        correct,
        len(data_loader.dataset),
        acc),
        fontsize=30
    )
    plt.show()
    model.train()



def load_model(path):
    """
    Load model from file path
    """

    model = Net()
    model.load_state_dict(torch.load(path))
    return model
