import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Convert images to tensors
transform = transforms.ToTensor()

# Load MNIST dataset
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

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class CNNDigitClassifier(nn.Module):
    def __init__(self):
        super(CNNDigitClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = CNNDigitClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5

train_losses = []
test_accuracies = []
all_labels = []
all_predictions = []

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    train_losses.append(average_loss)

    # Evaluate after each epoch
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if epoch == epochs - 1:
                all_labels.extend(labels.numpy())
                all_predictions.extend(predicted.numpy())

    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {average_loss:.4f}, Test Accuracy: {accuracy:.2f}%")


# Final accuracy
print(f"Final CNN Test Accuracy: {test_accuracies[-1]:.2f}%")


# Plot training loss
plt.figure()
plt.plot(range(1, epochs + 1), train_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("CNN Training Loss")
plt.grid(True)
plt.savefig("cnn_training_loss.png")
plt.show()


# Plot test accuracy
plt.figure()
plt.plot(range(1, epochs + 1), test_accuracies, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy (%)")
plt.title("CNN Test Accuracy")
plt.grid(True)
plt.savefig("cnn_test_accuracy.png")
plt.show()


# Show one prediction example
image, label = test_dataset[0]

with torch.no_grad():
    output = model(image.unsqueeze(0))
    _, predicted = torch.max(output, 1)

plt.figure()
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Actual: {label}, Predicted: {predicted.item()}")
plt.axis("off")
plt.savefig("cnn_prediction_example.png")
plt.show()

# Plot confusion matrix
cm = confusion_matrix(all_labels, all_predictions)
display = ConfusionMatrixDisplay(confusion_matrix=cm)

plt.figure(figsize=(8, 8))
display.plot()
plt.title("CNN Confusion Matrix")
plt.savefig("cnn_confusion_matrix.png")
plt.close()

print("Confusion matrix saved as cnn_confusion_matrix.png")

# Save CNN model
torch.save(model.state_dict(), "mnist_cnn_model.pth")
print("CNN model saved as mnist_cnn_model.pth")
print("Plots saved as cnn_training_loss.png, cnn_test_accuracy.png, and cnn_prediction_example.png")