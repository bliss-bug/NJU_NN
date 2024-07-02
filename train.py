from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from ResidualMLP import ResidualMLP
from data import MNISTDataset, DataLoader
import autograd as ag
import optim
import nn


def train_one_epoch(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss, num, acc = 0.0, 0, 0

    for x, y in tqdm(dataloader):
        logits = model(x)
        loss = loss_fn(logits, y)
        acc += np.sum(logits.numpy().argmax(axis=1) == y.numpy())
        total_loss += loss * x.shape[0]
        num += x.shape[0]

        optimizer.reset_grad()
        loss.backward()
        optimizer.step()

    return (total_loss / num).numpy().item(), (acc / num).item()



def test(dataloader, model, loss_fn):
    model.eval()
    total_loss, num, acc = 0.0, 0, 0

    for x, y in tqdm(dataloader):
        logits = model(x)
        loss = loss_fn(logits, y)
        acc += np.sum(logits.numpy().argmax(axis=1) == y.numpy())
        total_loss += loss * x.shape[0]
        num += x.shape[0]

    return (total_loss / num).numpy().item(), (acc / num).item()



if __name__ == '__main__':
    batch_size = 100
    lr = 1e-3
    epochs = 50
    num_classes = 10

    model = ResidualMLP(dim=784, hidden_dim=128, num_classes=num_classes)

    train_set = MNISTDataset("MNIST/train-images-idx3-ubyte.gz", "MNIST/train-labels-idx1-ubyte.gz")
    test_set = MNISTDataset("MNIST/t10k-images-idx3-ubyte.gz", "MNIST/t10k-labels-idx1-ubyte.gz")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9 ,weight_decay=1e-4)
    scheduler = optim.StepDecay(optimizer, 5, 0.9)

    x = list(range(1,epochs+1))
    losses, accs = [], []

    for i in range(epochs):
        train_loss, train_acc = train_one_epoch(train_loader, model, loss_fn, optimizer)
        losses.append(train_loss)
        accs.append(train_acc)
        scheduler.step()
        print('epoch {}: train_loss = {:.4f} | train_acc = {:.4f}'.format(i+1, train_loss, train_acc))

    plt.figure(1)
    plt.plot(x, losses)
    plt.xlabel('epochs')
    plt.ylabel('train loss')
    plt.savefig('loss.jpg')

    plt.figure(2)
    plt.plot(x, accs)
    plt.xlabel('epochs')
    plt.ylabel('train acc')
    plt.savefig('acc.jpg')

    test_loss, test_acc = test(test_loader, model, loss_fn)
    print('Test: test_loss = {:.4f} | test_acc = {:.4f}'.format(test_loss, test_acc))