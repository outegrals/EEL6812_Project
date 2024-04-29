import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
from torch.utils.data import DataLoader
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models

# Data augmentation and normalization
transform = transforms.Compose([
    transforms.Grayscale(),  # Ensure grayscale
    transforms.Resize((128, 256)),  # Resize to a standard size for CNN
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize for stable training
])

# Load training data
train_dataset = datasets.ImageFolder('samples', transform=transform)

# Split into training and validation sets (80/20)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_data, val_data = torch.utils.data.random_split(train_dataset, [train_size, val_size])

batch_size = 16
training_loss = [0]
training_acc = [0]

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Define a simple CNN model
class StripClassifier(nn.Module):
    def __init__(self):
        super(StripClassifier, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # Grayscale, 16 filters
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 16 * 32, 128)  # Dense layer with a flat input
        self.fc2 = nn.Linear(128, 2)  # Output layer with 2 classes (positive and negative)
        # Activation
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        
        x = torch.flatten(x, 1)  # Flatten the output
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # No softmax (handled in loss function)
        return x

# Initialize model, loss function, and optimizer
model = StripClassifier()
criterion = nn.CrossEntropyLoss()  # Cross-entropy for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# Training loop
num_epochs = 10  # Training epochs
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Zero the gradients
        
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        
        running_loss += loss.item()  # Track loss
    
    
    # Validation
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)  # Forward pass
            _, predicted = torch.max(outputs, 1)  # Get predictions
            total += labels.size(0)  # Track total samples
            correct += (predicted == labels).sum().item()  # Track correct predictions
    
    accuracy = 100 * correct / total  # Calculate accuracy
    
    # track training loss and accuracy
    training_loss.append(running_loss / len(train_loader))
    training_acc.append(accuracy)
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Accuracy: {accuracy}%")
    
    
# plot training loss
plt.plot(training_loss, label = 'training_loss')
plt.title('Training Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

# plot training accuracy
plt.plot(training_acc, label = 'training_acc')
plt.title('Training Accuracy')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

