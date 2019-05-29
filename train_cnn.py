import torch
from CNN import ConvNet
import numpy as np
from torch import nn
from data_set import DataSet
from torch.utils import data as torch_data


def train_net(train_x, train_y, test_x, test_y):
    [_, _, n_freq, n_times] = train_x.shape
    n_classes = len(np.unique(train_y))

    train_data_set = DataSet(train_x, train_y)
    train_loader = torch_data.DataLoader(train_data_set, batch_size=32, shuffle=True)

    test_data_set = DataSet(test_x, test_y)
    # test_loader = torch_data.DataLoader(test_data_set)

    model = ConvNet(n_freq, n_times, n_classes).double()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    num_epochs = 100

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
            outputs = model(test_data_set.all_x)
            loss = criterion(outputs, test_data_set.all_y)
            test_loss.append(loss.item())
            total = test_data_set.all_y.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == test_data_set.all_y).sum().item()
            acc = correct / total
            test_acc.append(acc)
            if acc > best_acc:
                model.save_state_dict('best_model.pt')
                best_acc = acc


def normalize_data(train_x, train_y, test_x, test_y):
    m = np.mean(train_x, axis=0)
    s = np.std(train_x, axis=0)
    train_x = np.divide(np.subtract(train_x, m), s)
    test_x = np.divide(np.subtract(test_x, m), s)

    [n_train, n_freq, n_times] = train_x.shape

    new_n_freq = int(np.ceil((n_freq - 6) / 4)) * 4 + 6
    new_n_times = int(np.ceil((n_times - 6) / 4)) * 4 + 6

    n_classes = len(np.unique(train_y))

    zero_train = np.zeros([n_train, new_n_freq, new_n_times])
    zero_test = np.zeros([test_x.shape[0], new_n_freq, new_n_times])
    zero_train[:, :n_freq, :n_times] = train_x
    zero_test[:, :n_freq, :n_times] = test_x

    train_x = zero_train[:, np.newaxis, :, :]
    test_x = zero_test[:, np.newaxis, :, :]

    indices = np.random.choice(len(test_y), 400)
    test_x = test_x[indices, :, :]
    test_y = test_y[indices]

    np.savez('Data', x_train=train_x, y_train=train_y, x_test=test_x, y_test=test_y)
    return train_x, train_y, test_x, test_y


data = np.load('Data_spec.npz')
train_x = data['x_train']
train_y = data['y_train']
test_x = data['x_test']
test_y = data['y_test']

# normalize_data(train_x, train_y, test_x, test_y)

train_net(train_x, train_y, test_x, test_y)
