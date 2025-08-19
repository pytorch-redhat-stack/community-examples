#import libraries

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

#define the transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

#download and load MNIST dataset
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

#define a neural network

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Input layer (28x28 = 784 pixels)
        self.fc2 = nn.Linear(128, 64)  # Hidden layer
        self.fc3 = nn.Linear(64, 10)  # Output layer (10 digits)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = torch.relu(self.fc1(x))  # Apply ReLU activation function
        x = torch.relu(self.fc2(x))  # Apply ReLU activation function
        x = self.fc3(x)  # Output layer (no activation function)
        return x


#intialize model,loss function and optimizer

model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#train the model

num_epochs = 5
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass (compute gradients)
        optimizer.step()  # Update the weights

        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 mini-batches
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0

print("Finished Training")

#evaluate the model

correct = 0
total = 0
with torch.no_grad():  # We don't need gradients for evaluation
    for inputs, labels in testloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)  # Get the predicted labels
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test dataset: {100 * correct / total:.2f}%")


#visualize the model

# Get a batch of images and labels from the testloader
dataiter = iter(testloader)
images, labels = next(dataiter)  # Use next() instead of .next()

# Get predictions
outputs = model(images)
_, predicted = torch.max(outputs, 1)

# Plot the images and predictions
fig = plt.figure(figsize=(10, 5))
for i in range(10):
    ax = fig.add_subplot(2, 5, i + 1)
    ax.imshow(images[i].numpy().squeeze(), cmap='gray')
    ax.set_title(f"Pred: {predicted[i].item()}")
    ax.axis('off')

plt.show()


