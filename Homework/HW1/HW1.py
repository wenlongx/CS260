#Don't change batch size
batch_size = 64

import itertools

from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms


def load_data():
    ## USE THIS SNIPPET TO GET BINARY TRAIN/TEST DATA

    train_data = datasets.MNIST('data/',
                                train=True,
                                download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]))
    test_data = datasets.MNIST('data/',
                               train=False,
                               download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))
    subset_train_indices = ((train_data.train_labels == 0) +
                            (train_data.train_labels == 1)).nonzero().view(-1)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=False,
        sampler=SubsetRandomSampler(subset_train_indices))


    subset_test_indices = ((test_data.test_labels == 0) +
                           (test_data.test_labels == 1)).nonzero().view(-1)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        sampler=SubsetRandomSampler(subset_test_indices))

    return train_loader, test_loader

class LogisticRegression(nn.Module):
    def __init__(self, n_feature, n_class):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_feature, n_class)

    def forward(self, x):
        h = self.linear(x)
        return h

    def predict(self, x):
        h = self.linear(x)
        pred = torch.sigmoid(h)
        return torch.where(pred > 0.5,
                           torch.ones(pred.shape),
                           -torch.ones(pred.shape))

class LinearSVM(nn.Module):
    def __init__(self, n_feature, n_class):
        super(LinearSVM, self).__init__()
        self.linear = nn.Linear(n_feature, n_class, bias=True)

    def forward(self, x):
        h = self.linear(x)
        return h

    def predict(self, x):
        h = self.linear(x)
        return torch.sign(h)

def logistic_loss(pred, labels):
    return torch.mean(torch.log(1 + torch.exp(-torch.mul(pred, labels))))

def hinge_loss(pred, labels):
    ywx = torch.mul(pred, labels)
    return torch.mean(torch.max(torch.zeros(ywx.shape), torch.ones(ywx.shape) - ywx))

if __name__ == "__main__":

    train_loader, test_loader = load_data()

    # Training the Model
    # Notice that newest Pytorch merge tensor and variable, so the additional Variable wrapping is no longer required.

    D_in, D_out = 784, 1

    models = [
        ("LogisticRegression", logistic_loss),
        ("LinearSVM", hinge_loss)
    ]

    model_name_to_class = {
        "LinearSVM": LinearSVM,
        "LogisticRegression": LogisticRegression
    }

    num_epochs = 10
    momentum = 0.9
    learning_rates = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

    results = {}

    for lr in learning_rates:
        results[lr] = {}
        for use_momentum in [False, True]:
            results[lr][use_momentum] = {}
            if use_momentum:
                print(f"Learning Rate: {lr}\tMomentum:\t{momentum}")
            else:
                print(f"Learning Rate: {lr}")
            for model_name, loss_fn in models:
                results[lr][use_momentum][model_name] = {}
                results[lr][use_momentum][model_name]["train"] = []
                print(model_name)

                model = model_name_to_class[model_name](D_in, D_out)

                if use_momentum:
                    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
                else:
                    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

                for epoch in range(num_epochs):
                    total_loss = 0.
                    num_batches = 0.
                    for i, (images, labels) in enumerate(train_loader):
                        images = Variable(images.view(-1, 28*28))
                        #Convert labels from 0,1 to -1,1
                        labels = Variable(2*(labels.float()-0.5))

                        ## TODO

                        y_pred = model(images).flatten()

                        batch_loss = loss_fn(y_pred, labels)
                        total_loss += batch_loss

                        optimizer.zero_grad()
                        batch_loss.backward()
                        optimizer.step()

                        num_batches += 1.

                    avg_loss = total_loss / num_batches

                    results[lr][use_momentum][model_name]["train"].append(avg_loss)

                    # Print your results every epoch
                    print(f"Epoch: \t{epoch}\tLoss:\t{avg_loss.item()}")

                # Test the Model
                correct = 0.
                total = 0.
                for images, labels in test_loader:
                    images = Variable(images.view(-1, 28*28))
                    #Convert labels from 0,1 to -1,1
                    labels = Variable(2*(labels.float()-0.5)).long()

                    ## Put your prediction code here
                    prediction = model.predict(images)

                    correct += (prediction.view(-1).long() == labels).sum()
                    total += images.shape[0]

                results[lr][use_momentum][model_name]["test"] = (100 * (correct.float() / total))
                print('Accuracy of the model on the test images: %f %%' % (100 * (correct.float() / total)))

    # # UNCOMMENT THE LINES BELOW TO GENERATE THE PLOTS USED IN THE REPORT
    #
    # import matplotlib.pyplot as plt
    # import numpy as np
    #
    # for model_name, loss_fn in models:
    #     for use_momentum in [False, True]:
    #         plt.figure(figsize=(10,10))
    #         for lr in learning_rates:
    #             acc = results[lr][use_momentum][model_name]["test"]
    #             plt.plot(np.linspace(1, num_epochs, num_epochs), results[lr][use_momentum][model_name]["train"],
    #                      label=f"Learning Rate = {lr}, Test Accuracy = {acc:.4f}%")
    #
    #         if use_momentum:
    #             plt.title(f"{model_name} using SGD with Momentum")
    #         else:
    #             plt.title(f"{model_name} using SGD")
    #         plt.xlabel("Epoch")
    #         plt.ylabel("Training Loss")
    #         plt.legend()
    #         plt.show()
