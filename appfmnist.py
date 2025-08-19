import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import streamlit as st
from PIL import Image
import numpy as np

# Define the model architecture (same as in the previous steps)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = torch.relu(self.fc1(x))  # Apply ReLU activation
        x = torch.relu(self.fc2(x))  # Apply ReLU activation
        x = self.fc3(x)  # Output layer
        return x

# Load the trained model
model = SimpleNN()
model.load_state_dict(torch.load('fashion_mnist_model.pth'))  # Load your trained model weights
model.eval()

# Define transformation for input image
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize image to 28x28 (if not already)
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale if needed
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Function to predict the class of the image
def predict_image(image):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Mapping Fashion MNIST labels to class names
classes = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Streamlit app logic
st.title("Fashion MNIST Prediction App")
st.write("Upload a Fashion MNIST image to predict its class.")

# Image upload widget
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict the class of the image
    prediction = predict_image(image)
    st.write(f"Predicted Class: {classes[prediction]}")
