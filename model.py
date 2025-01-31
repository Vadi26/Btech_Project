import torch
import torch.nn as nn
import torch.nn.functional as F

class FraudDetectionNet(nn.Module):
    def __init__(self, input_dim: int = 10, num_classes: int = 2):
        super(FraudDetectionNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)  # Binary classification (fraud or not)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))  # Output is a probability
        return x

class IOTNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(IOTNet, self).__init__()
        self.fc1 = nn.Linear(21, 128)  # Assuming 12 input features, adjust as per your dataset
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)  # Output layer for 24 classes

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class Net(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 14x14 -> 14x14
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 128 * 7 * 7)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_card_detection(net, trainloader, optimizer, epochs, device: str):
    # Train the network on the training set
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
    net.train()
    net.to(device)
    total_loss = 0
    for _ in range(epochs):
        for data, labels in trainloader:
            data, labels = data.to(device), labels.to(device).unsqueeze(1).float()
            optimizer.zero_grad()
            predictions = net(data)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    avg_loss = total_loss / (len(trainloader) * epochs)
    return avg_loss

def test_card_detection(net, testloader, device: str):
    # Validate the network on the entire test set
    criterion = nn.BCELoss()
    net.eval()
    net.to(device)
    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, labels in testloader:
            data, labels = data.to(device), labels.to(device).unsqueeze(1).float()
            predictions = net(data)
            total_loss += criterion(predictions, labels).item()
            predicted = (predictions >= 0.5).float()  # Threshold at 0.5
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = correct / total_samples
    return total_loss, accuracy

def train(net, trainloader, optimizer, epochs, device: str):
    # Train the network on the training set
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

def test(net, testloader, device: str):
    # Validate the network on the entire test set
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    net.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy

def train_iot(net, trainloader, optimizer, epochs, device: str):
    # Train the network on the training set
    criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
    net.train()
    net.to(device)
    
    for _ in range(epochs):
        total_loss = 0
        for data, labels in trainloader:
            data, labels = data.to(device), labels.to(device)
            labels = labels.long()
            optimizer.zero_grad()
            outputs = net(data)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    avg_loss = total_loss / (len(trainloader) * epochs)
    return avg_loss

def test_iot(net, testloader, device: str):
    # Validate the network on the test set
    criterion = nn.CrossEntropyLoss()
    net.eval()
    net.to(device)
    
    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, labels in testloader:
            data, labels = data.to(device), labels.to(device)
            outputs = net(data)
            loss = criterion(outputs, labels.long())
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)  # Get class with highest probability
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = correct / total_samples
    return total_loss, accuracy




def models_to_parameters(model):
    from flwr.common.parameter import ndarrays_to_parameters

    ndarrays = [val.cpu().numpy() for _, val in model.state_dict().items()]
    parameters = ndarrays_to_parameters(ndarrays)
    print("Extracted model parameters!")
    return parameters