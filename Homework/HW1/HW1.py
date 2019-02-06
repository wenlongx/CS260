#Don't change batch size
batch_size = 64

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
        h = torch.sigmoid(self.linear(x))
        return h

class LinearSVM():
    def __init__(self, n_feature, n_class):
        super(LinearSVM, self).__init__()
        self.linear = nn.Linear(n_feature, n_class)

    def forward(self, x):
        h = self.linear(x)
        return h

class LogRegLoss(torch.autograd.Function):
    @staticmethod
    def forward():


if __name__ == "__main__":

    train_loader, test_loader = load_data()

    # Training the Model
    # Notice that newest Pytorch merge tensor and variable, so the additional Variable wrapping is no longer required.

    D_in, D_out = 784, 1

    model = LogisticRegression(D_in, D_out)
    loss_fn = torch.nn.MSELoss(reduction='sum')

    num_epochs = 500


    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.view(-1, 28*28))
            #Convert labels from 0,1 to -1,1
            labels = Variable(2*(labels.float()-0.5))

            ## TODO

            y_pred = model(images)

            batch_loss = loss_fn(y_pred, labels)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()


        # Print your results every epoch

    exit()

    # Test the Model
    correct = 0.
    total = 0.
    for images, labels in test_loader:
        images = Variable(images.view(-1, 28*28))

        ## Put your prediction code here

        correct += (prediction.view(-1).long() == labels).sum()
        total += images.shape[0]
    print('Accuracy of the model on the test images: %f %%' % (100 * (correct.float() / total)))
