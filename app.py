import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps, UnidentifiedImageError
from io import BytesIO
import torchvision.transforms as transforms


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


st.title("Handwritten Digit Classifier")
st.write("Upload a handwritten digit image and the CNN model will predict the digit.")

model = CNNDigitClassifier()
model.load_state_dict(torch.load("mnist_cnn_model.pth", map_location=torch.device("cpu")))
model.eval()

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(BytesIO(image_bytes)).convert("L")

        st.subheader("Uploaded Image")
        st.image(image, width=200)

        # Convert image to MNIST-like format
        image = ImageOps.invert(image)
        image = image.resize((28, 28))

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        st.subheader("Prediction")
        st.success(f"Predicted Digit: {predicted.item()}")
        st.write(f"Confidence: **{confidence.item() * 100:.2f}%**")

    except UnidentifiedImageError:
        st.error("The uploaded file could not be recognized as an image. Please upload a valid PNG, JPG, or JPEG file.")

    except Exception as e:
        st.error(f"Something went wrong: {e}")