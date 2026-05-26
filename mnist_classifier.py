import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# 1. Convert images to tensors
transform = transforms.ToTensor()

# 2. Download training and testing datasets
train_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

# 3. Load data in batches
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 4. Build a simple neural network
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()

        self.model = nn.Sequential(
            nn.Flatten(),              # 28x28 image becomes 784 values
            nn.Linear(28 * 28, 128),   # input layer to hidden layer
            nn.ReLU(),                 # activation function
            nn.Linear(128, 64),        # hidden layer
            nn.ReLU(),
            nn.Linear(64, 10)          # output layer: 10 classes
        )

    def forward(self, x):
        return self.model(x)


# 5. Create model, loss function, and optimizer
model = DigitClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 6. Train the model
epochs = 5

for epoch in range(epochs):
    total_loss = 0

    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss:.4f}")


# 7. Test the model
correct = 0
total = 0

model.eval()

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")


# 8. Show one prediction example
image, label = test_dataset[0]

with torch.no_grad():
    output = model(image.unsqueeze(0))
    _, predicted = torch.max(output, 1)

plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Actual: {label}, Predicted: {predicted.item()}")
plt.axis("off")
plt.show()


# 9. Save the trained model
torch.save(model.state_dict(), "mnist_model.pth")
print("Model saved as mnist_model.pth")