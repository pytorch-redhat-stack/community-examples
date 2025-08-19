import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the model architecture (as you have already done)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Input layer
        self.fc2 = nn.Linear(128, 64)       # Hidden layer
        self.fc3 = nn.Linear(64, 10)        # Output layer

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = torch.relu(self.fc1(x))  # Apply ReLU activation
        x = torch.relu(self.fc2(x))  # Apply ReLU activation
        x = self.fc3(x)  # Output layer
        return x

# Define transformations and data loaders
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to the range [-1, 1]
])

trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Initialize the model, criterion, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in trainloader:
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}")

# After training the model, save the model's state_dict (weights)
torch.save(model.state_dict(), 'fashion_mnist_model.pth')
print("Model saved successfully!")

# Load the model's state_dict
model = SimpleNN()  # Initialize the model again
model.load_state_dict(torch.load('fashion_mnist_model.pth'))  # Load the saved weights
model.eval()  # Set the model to evaluation mode

# Evaluate the model
correct = 0
total = 0
with torch.no_grad():  # No need to compute gradients for evaluation
    for inputs, labels in testloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)  # Get the predicted class
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test dataset: {100 * correct / total:.2f}%")

# Visualize some test images with predictions
import matplotlib.pyplot as plt

dataiter = iter(testloader)
images, labels = next(dataiter)  # Use next() instead of .next()

outputs = model(images)
_, predicted = torch.max(outputs, 1)

fig = plt.figure(figsize=(10, 5))
for i in range(10):
    ax = fig.add_subplot(2, 5, i + 1)
    ax.imshow(images[i].numpy().squeeze(), cmap='gray')
    ax.set_title(f"Pred: {predicted[i].item()}")
    ax.axis('off')

plt.show()
