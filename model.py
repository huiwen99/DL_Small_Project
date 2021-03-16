import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import datetime
import matplotlib.pyplot as plt

class TwoBinaryClassifiers(nn.Module):
    """
    The proposed deep learning model that classifies X-ray images of patients
    and help doctors with the diagnosis of COVID/non-COVID pneuomnia.
    Combines Normal_VS_Infected model and NonCovid_VS_Covid model.
    """
    def __init__(self):
        super(TwoBinaryClassifiers, self).__init__()
        self.bc1 = Normal_VS_Infected()
        self.bc2 = Covid_VS_NonCovid()
        
    def forward(self, x):
        # send the input to BC #1
        label = int(self.bc1(x).argmax(dim=1))
        
        # send the input to BC #2 if the output of BC #1 is predicted as infected
        if label == 1:
            label = int(self.bc2(x).argmax(dim=1)) + 1
        
        # one-hot encoding to 3-tensor
        output = torch.zeros((1, 3))
        output[0][label] = 1
        
        return output

class Normal_VS_Infected(nn.Module):
    """
    Binary classifier #1 that classifies X-ray images of patients into Normal or
    Infected classes.
    """
    def __init__(self):
        super(Normal_VS_Infected, self).__init__()
        # Conv2D: 1 input channel, 8 output channels, 3 by 3 kernel
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        # self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        # self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(75*75*8,128)
        self.classifier = nn.Linear(128, 2)

        #  Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight)
            elif isinstance(m, nn.Linear): # need to intialize m.weight?
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.classifier(x)
        output = F.log_softmax(x, dim = 1)
        return output
    
class Covid_VS_NonCovid(nn.Module):
    """
    Binary classifier #2 that classifies X-ray images of patients into COVID or
    non-Covid classes.
    """
    def __init__(self):
        super(Covid_VS_NonCovid, self).__init__()
        # Conv2D: 1 input channel, 8 output channels, 3 by 3 kernel
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        # self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        # self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(75 * 75 * 8, 128)
        self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.classifier(x)
        output = F.log_softmax(x, dim=1)
        return output
    
class ThreeClassesClassifier(nn.Module):
    """
    The baseline deep learning model that classifies X-ray images of patients
    and help doctors with the diagnosis of COVID/non-COVID pneuomnia.
    """
    def __init__(self):
        super(ThreeClassesClassifier, self).__init__()
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

    train_loss, train_acc, train_recall = evaluate(model, device, train_loader)
    print('Train Epoch: {} @ {} \nTrain Loss: {:.4f} - Train Accuracy: {:.1f}% - Train Recall: {:.1f}%'.format(
        epoch, datetime.datetime.now().time(), train_loss, train_acc, train_recall))

    test_loss, test_acc, test_recall = evaluate(model, device, test_loader)
    print("Test Loss: {:.4f} - Test Accuracy: {:.1f}% - Test Recall: {:.1f}%".format(test_loss, test_acc, test_recall))
    
    return train_loss, train_acc, train_recall, test_loss, test_acc, test_recall

def evaluate(model, device, data_loader):
    """
    Evaluates the model and returns loss, accuracy and recall for the positive class ([0,1])
    """
    model.eval()
    loss = 0
    correct = 0
    correct_true = 0
    target_true = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            target = torch.argmax(target, dim=1).long()
            output = model(data)
            criterion = nn.CrossEntropyLoss()
            loss += criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            correct_true += torch.sum(pred.squeeze(1) * target).item()
            target_true += torch.sum(target == 1).item()

    loss /= len(data_loader)
    acc = 100. * correct / len(data_loader.dataset)
    recall = 100. * correct_true / target_true
    return loss, acc, recall

def display_performance(model, device, data_loader):
    """
    Displays subplot containing each image in the dataset, its ground truth label and predicted labels.
    Also displays overall accuracy.
    """
    model.eval()
    fig = plt.figure(figsize=(20,40))
    cols = 4
    rows = 6
    correct = 0
    idx = 1
    classes = {0: 'normal', 1:'covid', 2:'non-covid'}
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            target = torch.argmax(target, dim=1).long()
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            fig.add_subplot(rows, cols, idx)
            idx+=1
            target = target.item()
            pred = pred.item()
            plt.title("Ground truth label: {}\nPredicted label: {}".format(classes[target], classes[pred]))
            img = data[0].cpu().squeeze(0)
            plt.imshow(img)

    acc = 100. * correct / len(data_loader.dataset)
    plt.suptitle("Validation set pictures with predicted and ground truth labels\nAverage accuracy {}/{} = {:.1f}%".format(
        correct,
        len(data_loader.dataset),
        acc),
        fontsize=30
    )
    plt.show()
                
def load_model(model_name, model_path):
    """
    Load model from file path
    """
    if model_name == "binary_classifier_1":
        model = Normal_VS_Infected()
    elif model_name == "binary_classifier_2":
        model = Covid_VS_NonCovid()
    else:
        model = ThreeClassesClassifier()
        
    model.load_state_dict(torch.load(model_path))
    return model

