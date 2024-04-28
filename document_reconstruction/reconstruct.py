import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os

class PairedStripDataset(Dataset):
    def __init__(self, positive_dir, negative_dir, transform=None):
        self.positive_dir = positive_dir
        self.negative_dir = negative_dir
        self.transform = transform
        self.pairs = []
        self.labels = []

        # Load and pair images
        for filename in os.listdir(positive_dir):
            image_path = os.path.join(positive_dir, filename)
            image = Image.open(image_path).convert('RGB')  # Ensure images are RGB
            if self.transform:
                image = self.transform(image)
            self.pairs.append((image, 1))  # Positive label

        for filename in os.listdir(negative_dir):
            image_path = os.path.join(negative_dir, filename)
            image = Image.open(image_path).convert('RGB')  # Ensure images are RGB
            if self.transform:
                image = self.transform(image)
            self.pairs.append((image, 0))  # Negative label

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 128)
        self.fc = nn.Linear(256, 1)

    def forward(self, input1, input2):
        output1 = self.backbone(input1)
        output2 = self.backbone(input2)
        combined = torch.cat((output1, output2), dim=1)
        similarity = torch.sigmoid(self.fc(combined))
        return similarity

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

positive_path = os.path.join(os.getcwd(), 'samples', 'positive')
negative_path = os.path.join(os.getcwd(), 'samples', 'negative')
dataset = PairedStripDataset(positive_path, negative_path, transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, device, train_loader, optimizer, epochs=10):
    model.train()
    criterion = nn.BCELoss()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device).view(-1, 1)
            optimizer.zero_grad()
            output = model(data[:, 0], data[:, 1])
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)} / {len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Example usage:
train(model, device, train_loader, optimizer)
