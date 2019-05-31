import torch
from CNN import ConvNet
import numpy as np
from torch import nn
from data_set import DataSet
from torch.utils import data as torch_data


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def train_net(train_x, train_y, test_x, test_y):
    [_, _, n_freq, n_times] = train_x.shape
    n_classes = len(np.unique(train_y))

    train_data_set = DataSet(train_x, train_y)
    train_loader = torch_data.DataLoader(train_data_set, batch_size=32, shuffle=True)

    test_data_set = DataSet(test_x, test_y)
    test_x = test_data_set.all_x.to(device)
    test_y = test_data_set.all_y.to(device)
    # test_loader = torch_data.DataLoader(test_data_set)

    model = ConvNet(n_freq, n_times, n_classes).double().to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    num_epochs = 10

    total_step = len(train_loader)
    train_loss = []
    train_acc = []

    test_loss = []
    test_acc = []

    best_acc = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = []
        epoch_acc = []
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            # Run the forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            epoch_acc.append(correct / total)

            if i % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))
        train_loss.append(np.mean(epoch_loss))
        train_acc.append(np.mean(epoch_acc))
        model.eval()
        with torch.no_grad():
            outputs = model(test_x)
            loss = criterion(outputs, test_y)
            test_loss.append(loss.item())
            total = test_y.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == test_y).sum().item()
            acc = correct / total
            test_acc.append(acc)
            if acc > best_acc:
                print(acc)
                torch.save(model.state_dict(), 'best_model.pth')
                best_acc = acc


data = np.load('Data.npz')
train_x = data['train_x']
train_y = data['train_y']
test_x = data['test_x']
test_y = data['test_y']


train_net(train_x, train_y, test_x, test_y)
