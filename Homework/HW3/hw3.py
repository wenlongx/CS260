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

import sys
import gc


class MLP(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()

        self.input_dim = np.prod(input_size)

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        self.fc_out = nn.Linear(32, num_classes)

    def forward(self, x):
        out = x.view(-1, self.input_dim)
        out = self.fc(out)
        out = self.fc_out(out)
        return out

    def predict(self, x):
        out = self.forward(x)
        return torch.argmax(out, dim=1)

class MLP_NoRelu(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP_NoRelu, self).__init__()

        self.input_dim = np.prod(input_size)

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 32),
        )

        self.fc_out = nn.Linear(32, num_classes)

    def forward(self, x):
        out = x.view(-1, self.input_dim)
        out = self.fc(out)
        out = self.fc_out(out)
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
        self.n_channels, self.x_dim, self.y_dim = input_dim

        self.conv_layers = nn.Sequential(
            # 3 x 32 x 32
            nn.Conv2d(self.n_channels, 64, kernel_size=(3, 3), padding=1, stride=1),
            nn.ReLU(),
            # 64 x 32 x 32
            nn.Conv2d(64, 64, kernel_size=(2, 2), padding=0, stride=2),
            nn.ReLU(),
            # 64 x 16 x 16
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1, stride=1),
            nn.ReLU(),
            # 64 x 16 x 16
            nn.Conv2d(64, 64, kernel_size=(2, 2), padding=0, stride=2),
            nn.ReLU(),
            # 64 x 8 x 8
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 10)
        )


    def forward(self, x):
        out = x
        out = self.conv_layers(out)
        out = out.view(-1, 64 * 8 * 8)
        out = self.fc_layers(out)
        return out

    def predict(self, x):
        out = self.forward(x)
        return torch.argmax(out, dim=1)


class CNN_NoRelu(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNN_NoRelu, self).__init__()

        """
        4 convolutional layers and 3 fully- connected layers, with ReLu activation function. The input dimension of 1st fully-connected layer must be 4096.
        """
        self.n_channels, self.x_dim, self.y_dim = input_dim

        self.conv_layers = nn.Sequential(
            # 3 x 32 x 32
            nn.Conv2d(self.n_channels, 64, kernel_size=(3, 3), padding=1, stride=1),
            # 64 x 32 x 32
            nn.Conv2d(64, 64, kernel_size=(2, 2), padding=0, stride=2),
            # 64 x 16 x 16
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1, stride=1),
            # 64 x 16 x 16
            nn.Conv2d(64, 64, kernel_size=(2, 2), padding=0, stride=2),
            # 64 x 8 x 8
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 8 * 8, 1024),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.Dropout(p=0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        out = x
        out = self.conv_layers(out)
        out = out.view(-1, 64 * 8 * 8)
        out = self.fc_layers(out)
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

    # Build Model
    start_epoch = 0
    num_epochs = 100

    input_dim = (3, 32, 32)
    num_classes = 10

    if len(sys.argv) < 2 or (sys.argv[1] != "MLP" and sys.argv[1] != "MLP_NoRelu" and sys.argv[1] != "CNN" and sys.argv[1] != "CNN_NoRelu"):
        print("Please call this script with the argument \"MLP\", \"MLP_NoRelu\" or \"CNN\", to tell it which model to run. Defaulting to CNN.")
        model = CNN(input_dim, num_classes)
        model_name = "CNN"
    elif sys.argv[1] == "MLP":
        model = MLP(np.prod(input_dim), num_classes)
        model_name = "MLP"
    elif sys.argv[1] == "MLP_NoRelu":
        model = MLP_NoRelu(np.prod(input_dim), num_classes)
        model_name = "MLP_NoRelu"
    elif sys.argv[1] == "CNN":
        model = CNN(input_dim, num_classes)
        model_name = "CNN"
    elif sys.argv[1] == "CNN_NoRelu":
        model = CNN_NoRelu(input_dim, num_classes)
        model_name = "CNN_NoRelu"
    else:
        print("Please call this script with the argument \"MLP\", \"MLP_NoRelu\" or \"CNN\", to tell it which model to run. Defaulting to CNN.")

    if len(sys.argv) >= 5 and sys.argv[2] == "--restart":
        if int(sys.argv[3]) > 0:
            model.load_state_dict(torch.load(f"results/{model_name}_e{sys.argv[3]}.ckpt"))
        start_epoch = int(sys.argv[3])
        if len(sys.argv) >= 5:
            num_epochs = int(sys.argv[4])


    # Load Data
    batch_size = 64
    test_batch_size = 64

    train_loader, _ = cifar_loaders(batch_size)
    _, test_loader = cifar_loaders(test_batch_size)

    print("====================================================================================")
    print(f"\t\t\t\t {model_name}")
    print("====================================================================================")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    results = {
        "train_loss": [],
        "test_acc": 0
    }

    for epoch in range(start_epoch+1, num_epochs+1):
        total_loss = 0.
        num_batches = 0.
        start_time = time.time()
        for i, (images, labels) in enumerate(train_loader):

            # images = torch.Size([64, 3, 32, 32])
            # labels = torch.Size([64, ])

            y_pred = model(images)

            batch_loss = loss_fn(y_pred, labels)
            total_loss += batch_loss.detach().item()

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            num_batches += 1.

            del batch_loss, y_pred, images, labels

        avg_loss = total_loss / num_batches

        end_time = time.time()
        elapsed = end_time - start_time

        # Print your results every epoch
        print(f"Epoch: \t{epoch}\tTime: \t{elapsed:.6f}\tLoss:\t{avg_loss}")
        results["train_loss"].append(avg_loss)

        # test acc every 10 epochs
        if epoch % 10 == 0:
            model.eval()
            correct = 0.
            total = 0.
            for images, labels in test_loader:
                prediction = model.predict(images)

                correct += (prediction.view(-1).long() == labels).sum()
                total += images.shape[0]

            print('Accuracy of the model for epoch %f on the test images: %f %%' % (epoch, 100 * (correct.float() / total)))

            torch.save(model.state_dict(), f"results/{model_name}_e{epoch}.ckpt")

            model.train()
        gc.collect()

    # Test the Model
    model.eval()
    correct = 0.
    total = 0.
    for images, labels in test_loader:

        ## Put your prediction code here
        prediction = model.predict(images)

        correct += (prediction.view(-1).long() == labels).sum()
        total += images.shape[0]

    print('Accuracy of the model on the test images: %f %%' % (100 * (correct.float() / total)))
    results["test_acc"] = 100 * (correct.float() / total)
    test_acc = results["test_acc"]

    print(results["train_loss"])
    print(np.array(results["train_loss"]))

    # save model and loss
    torch.save(model.state_dict(), f"results/{model_name}_{test_acc:.0f}.ckpt")
    np.savetxt(f"results/{model_name}_{test_acc:.0f}_train_loss.txt", np.array(results["train_loss"]))

