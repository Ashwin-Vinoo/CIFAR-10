# Created by Fuxin Lee. Modified by Ashwin Vinoo
# Date 3/4/2019

# Importing all the necessary modules
from __future__ import print_function
from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import os
from matplotlib import pyplot as plot

# --------------------------------- Hyper parameters ---------------------------------
# Whether we want to load the trained model
load_trained_model = False
# The name of the trained model to be loaded
trained_model_name = 'trained_network_adagrad.pth'
# The optimizer to use for training
optimizer_to_use = 'ADAGRAD'
# The primary plot title to display
plot_title = 'Title'
# This is the mini_batch size to be used
BATCH_SIZE = 32
# we train for this many epochs
MAX_EPOCH = 2
# ------------------------------------------------------------------------------------


# The neural network class which inherits from torch.nn.Module class
class Net(nn.Module):

    # The class constructor
    def __init__(self):
        # We call init of super class
        super(Net, self).__init__()
        # We have 4 2D convolution layers with padding of one
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        # One 2X2 max pool layer
        self.pool = nn.MaxPool2d(2, 2)
        # Two linear transform layers
        self.fc1 = nn.Linear(4096, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 10)
        # The batch normalization layers
        self.batch_norm_1 = nn.BatchNorm1d(4096)
        self.batch_norm_2 = nn.BatchNorm1d(512)

    # This function specifies the working of the forward pass
    def forward(self, x):
        # ReLU activation unit after convolution layer one
        x = F.relu(self.conv1(x))
        # ReLU activation unit after convolution layer two
        x = F.relu(self.conv2(x))
        # 2x2 Max pooling is performed
        x = self.pool(x)
        # ReLU activation unit after convolution layer three
        x = F.relu(self.conv3(x))
        # ReLU activation unit after convolution layer four
        x = F.relu(self.conv4(x))
        # 2x2 Max pooling is performed
        x = self.pool(x)
        # reshaping: -1 specifies that we don't care about number of rows but number of columns is as provided
        x = x.view(-1, self.num_flat_features(x))
        # Applies 2D batch normalization
        x = self.batch_norm_1(x)
        # linear transform layer one
        x = self.fc1(x)
        # ReLU activation unit
        x = F.leaky_relu(x)
        # Applies 2D batch normalization
        x = self.batch_norm_2(x)
        # Linear transform layer two
        x = self.fc3(x)
        # ReLU activation unit
        x = F.leaky_relu(x)
        # Applies 2D batch normalization
        x = self.batch_norm_2(x)
        # Linear transform layer four
        x = self.fc2(x)
        return x

    # This function returns the number of flat features
    @staticmethod
    def num_flat_features(x):
        # all dimensions except the batch dimension of the tensor
        size = x.size()[1:]
        # We initalize the number of features to be 1
        num_features = 1
        # Iterating through all the possible dimensions of the tensor
        for s in size:
            # We multiply the number of features by that amount and make it the new value of number of features
            num_features *= s
        # Returns the number of features within the tensor x
        return num_features


# This is a function to evaluate the neural network
def eval_net(data_loader):
    # initializing the number of correct predictions, total_loss and total predictions to be zero
    correct = 0
    total = 0
    total_loss = 0
    # Making the network to be in evaluation mode so that losses won't be accumulated, thereby saving memory
    net.eval()
    # The criterion used is the cross entropy loss
    criterion = nn.CrossEntropyLoss(reduction='sum')
    # Iterating through the data is data_loader
    for data in data_loader:
        # Splitting into images and labels
        images, labels = data
        # Converting to nn variable. The cuda() creates another Variable that isn’t a leaf node in the computation graph
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
        # The output obtained from the neural network
        outputs = net(images)
        # Returns the maximum value of each row of the input tensor in the given dimension dim and its index
        _, predicted = torch.max(outputs.data, 1)
        # Obtains the total number of images that are to be classified
        total += labels.size(0)
        # Correct predictions is when predicted output matches the label
        correct += (predicted == labels.data).sum()
        # The loss is evaluated based on the cross entropy loss criteria
        loss = criterion(outputs, labels)
        # Added to the total_loss
        total_loss += loss.data
    # Converting the network back into training mode
    net.train()
    # Returns the average loss and prediction accuracy
    return total_loss / total, int(correct) / total

# If this file is the main one called for execution
if __name__ == "__main__":

    # The first tuple (0.5, 0.5, 0.5) is the mean for all three channels and the second (0.5, 0.5, 0.5) is the SD
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Obtains the transformed training dataset CIFAR10
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # Combines a dataset and sampler to provide single or multi-process iterators over the dataset (2 sub-processes)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    # Obtains the transformed test dataset CIFAR10
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # Combines a dataset and sampler to provide single or multi-process iterators over the dataset (2 sub-processes)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Tuple of classes in which image may be classified
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print('Building model...')
    # Build the neural network model
    net = Net().cuda()
    # Converting the network back into training mode
    net.train()
    # The loss function which is used as the training criterion
    criterion = nn.CrossEntropyLoss()

    # Choosing the different types of optimizers available
    if optimizer_to_use == 'ADAM':
        # The optimizer to use during training is Adam
        optimizer = optim.Adam(net.parameters(), lr=0.01)
    elif optimizer_to_use == 'RMSPROP':
        # The optimizer to use during training is RMSprop
        optimizer = optim.RMSprop(net.parameters(), lr=0.01, momentum=0.9)
    elif optimizer_to_use == 'ADAGRAD':
        # The optimizer to use during training is Adagrad
        optimizer = optim.Adagrad(net.parameters(), lr=0.01)
    else:
        # The optimizer to use during training is stochastic gradient descent
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    # Gets the directory of this python file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # Check if we wanted to load the trained model
    if load_trained_model:
        # Loads the pretrained dictionary
        pretrained_dict = torch.load(dir_path + '\\' + trained_model_name)
        # The dictionary for the current neural network
        current_net_dict = net.state_dict()
        # Iterating through the key-value pairs in the dictionary
        for key, value in pretrained_dict.items():
            # Copies the values from the pretrained dictionary to the current model
            current_net_dict[key] = pretrained_dict[key]
        # Loads the data in the current network dictionary
        net.load_state_dict(current_net_dict)

    # These variables are used for plotting purposes
    x_epoch = np.linspace(1, MAX_EPOCH, MAX_EPOCH)
    y_training_loss = np.zeros(MAX_EPOCH)
    y_training_accuracy = np.zeros(MAX_EPOCH)
    y_testing_loss = np.zeros(MAX_EPOCH)
    y_testing_accuracy = np.zeros(MAX_EPOCH)

    print('Start training...')
    # loop over the dataset multiple times till we cover the maximum number of epochs
    for epoch in range(MAX_EPOCH):
        # The running loss is initialized to zero
        running_loss = 0.0
        # Enumerating through the training dataset
        for i, data in enumerate(trainloader, 0):
            # get the inputs and corresponding labels
            inputs, labels = data
            # wrap them in a variable
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            # obtain the outputs of the neural network
            outputs = net(inputs)
            # Calculate the loss by cross entropy loss between outputs and labels
            loss = criterion(outputs, labels)
            # Perform backward propagation to calculate the gradients at every layer
            loss.backward()
            # Optimizer changes the weights at every weight according to stochastic gradient descent
            optimizer.step()
            # print statistics
            running_loss += loss.data
            # print every 500 mini-batches
            if i % 500 == 499:
                print('    Step: %5d avg_batch_loss: %.5f' % (i + 1, running_loss / 500))
                # Resetting the running loss
                running_loss = 0.0
        print('    Finish training this EPOCH, start evaluating...')
        # Now that we finished training on an epoch, we evaluate the network on training dataset
        train_loss, train_acc = eval_net(trainloader)
        # We evaluate the network on the test dataset as well
        test_loss, test_acc = eval_net(testloader)
        # Store the training loss in a variable
        y_training_loss[epoch] = train_loss
        # Store the training accuracy in a variable
        y_training_accuracy[epoch] = train_acc
        # Store the test loss in a variable
        y_testing_loss[epoch] = test_loss
        # Store the test accuracy in a variable
        y_testing_accuracy[epoch] = test_acc
        print('EPOCH: %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f' %
              (epoch+1, train_loss, train_acc, test_loss, test_acc))
    print('Finished Training')
    print('Saving model...')
    # Save the neural network model to the specified path
    torch.save(net.state_dict(), dir_path + '\\' + 'trained_network.pth')
    print('Displaying Plots...')

    # Specify that we require two subplots along one row
    figure, (axis_1, axis_2) = plot.subplots(1, 2)
    # Specifying the primary plot title
    figure.suptitle(plot_title)
    # Plotting the training loss and accuracy
    axis_1.plot(x_epoch, y_training_loss, x_epoch, y_training_accuracy)
    # Adding in the legend
    axis_1.legend(('Training Loss', 'Training Accuracy'))
    # Specifying the plot title
    axis_1.set_title('Training Performance over Epochs')
    # Specifying the label for the x-axis
    axis_1.set_xlabel('Epochs')
    # Specifying the label for the y-axis
    axis_1.set_ylabel('Training Loss/Accuracy')
    # Setting the tick frequency on the x-axis
    axis_1.xaxis.set_ticks(np.arange(1, MAX_EPOCH+1, 1))
    # Plotting the testing loss and accuracy
    axis_2.plot(x_epoch, y_testing_loss, x_epoch, y_testing_accuracy)
    # Adding in the legend
    axis_2.legend(('Testing Loss', 'Testing Accuracy'))
    # Specifying the plot title
    axis_2.set_title('Testing Performance over Epochs')
    # Specifying the label for the x-axis
    axis_2.set_xlabel('Epochs')
    # Specifying the label for the y-axis
    axis_2.set_ylabel('Testing Loss/Accuracy')
    # Setting the tick frequency on the x-axis
    axis_2.xaxis.set_ticks(np.arange(1, MAX_EPOCH+1, 1))
    # Display the plot
    plot.show()

# ------------------------------------ END OF PROGRAM ---------------------------------------------
