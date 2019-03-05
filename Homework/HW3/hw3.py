import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import torch.utils.data as td
import random,time


class MLP(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()

        self.input_dim = np.prod(input_size)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        self.fc1 = nn.Linear(self.input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 32)
        self.fc7 = nn.Linear(32, num_classes)

    def forward(self, x):
        out = Variable(x.view(-1, self.input_dim))
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        out = self.relu(out)
        out = self.fc6(out)
        out = self.relu(out)
        out = self.fc7(out)
        return out

    def predict(self, x):
        out = self.forward(x)
        return torch.argmax(out, dim=1)

class CNN(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNN, self).__init__()

        """
        4 convolutional layers and 3 fully- connected layers, with ReLu activation function. The input dimension of 1st fully-connected layer must be 4096.
        """
        n_channels, x_dim, y_dim = input_dim

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(n_channels, 16, kernel_size=(5, 5), padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)

        self.fc1 = nn.Linear(128 * 32 * 32, 4096)
        self.fc2 = nn.Linear(4096, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        out = x

        out = self.conv1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.relu(out)

        out = out.view(-1, 128 * 32 * 32)

        out = self.fc1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.relu(out)

        out = self.fc3(out)

        return out

    def predict(self, x):
        out = self.forward(x)
        return torch.argmax(out, dim=1)


def cifar_loaders(batch_size, shuffle_test=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    train = datasets.CIFAR10('./', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]))
    test = datasets.CIFAR10('./', train=False,
        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
        shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
        shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader


if __name__ == "__main__":

    # Load Data
    batch_size = 64
    test_batch_size = 64

    train_loader, _ = cifar_loaders(batch_size)
    _, test_loader = cifar_loaders(test_batch_size)


    # Build Model
    num_epochs = 1

    input_dim = (3, 32, 32)
    num_classes = 10

    models = [
        (MLP(np.prod(input_dim), num_classes), "MLP"),
        (CNN(input_dim, num_classes), "CNN")
    ]

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    results = {}

    for model, model_name in models:
        print("====================================================================================")
        print(f"\t\t\t {model_name}")
        print("====================================================================================")

        results[model_name] = {
            "train_loss": [],
            "test_acc": 0
        }

        for epoch in range(num_epochs):
            total_loss = 0.
            num_batches = 0.
            for i, (images, labels) in enumerate(train_loader):

                # images = torch.Size([64, 3, 32, 32])
                # labels = torch.Size([64, ])

                labels = Variable(labels)
                y_pred = model(images)

                batch_loss = loss_fn(y_pred, labels)
                total_loss += batch_loss

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                num_batches += 1.

            avg_loss = total_loss / num_batches

            # Print your results every epoch
            print(f"Epoch: \t{epoch}\tLoss:\t{avg_loss.item()}")
            results[model_name]["train_loss"].append(avg_loss)

        # Test the Model
        correct = 0.
        total = 0.
        for images, labels in test_loader:

            ## Put your prediction code here
            prediction = model.predict(images)

            correct += (prediction.view(-1).long() == labels).sum()
            total += images.shape[0]

        print('Accuracy of the model on the test images: %f %%' % (100 * (correct.float() / total)))
        results[model_name]["test_acc"] = 100 * (correct.float() / total)

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
