import torch
from CNN import ConvNet
import numpy as np
from torch import nn
from data_set import DataSet
from torch.utils import data as torch_data


def train_net(train_x, train_y, test_x, test_y):
    [n_train, n_freq, n_times] = train_x.shape
    new_n_freq = int(np.ceil((n_freq - 6) / 4)) * 4 + 6
    new_n_times = int(np.ceil((n_times - 6) / 4)) * 4 + 6

    n_classes = len(np.unique(train_y))

    zero_train = np.zeros([n_train, new_n_freq, new_n_times])
    zero_test = np.zeros([test_x.shape[0], new_n_freq, new_n_times])
    zero_train[:,:n_freq, :n_times] = train_x
    zero_test[:, :n_freq, :n_times] = test_x

    train_x = zero_train[:, np.newaxis, :, :]
    test_x = zero_test[:, np.newaxis, :, :]

    train_data_set = DataSet(train_x, train_y.astype(float))
    train_loader = torch_data.DataLoader(train_data_set, batch_size=16, shuffle=True)

    test_data_set = DataSet(test_x, test_y.astype(float))
    test_loader = torch_data.DataLoader(test_data_set)

    model = ConvNet(new_n_freq, new_n_times, n_classes).double()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100

    total_step = len(train_x)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            # Run the forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if i % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))
        # model.eval()
        # with torch.no_grad():
        #     outputs = model(images)
        #     loss = criterion(outputs, labels)
        #     loss_list.append(loss.item())



data = np.load('Data_spec.npz')
train_net(data['x_train'], data['y_train'], data['x_test'], data['y_test'])
