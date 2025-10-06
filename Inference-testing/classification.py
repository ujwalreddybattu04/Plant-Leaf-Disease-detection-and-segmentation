import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import os
import random

# -------------------------------
# ✅ Model Definition
# -------------------------------
class PlantVillageCNN(nn.Module):
    def __init__(self, num_classes=38):
        super(PlantVillageCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x)); x = self.pool1(x)
        x = F.relu(self.conv2(x)); x = self.pool2(x)
        x = F.relu(self.conv3(x)); x = self.pool3(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# -------------------------------
# ✅ Custom Preprocessing (same as training)
# -------------------------------
def custom_preprocess(img_bgr, size=(256,256)):
    img = cv2.GaussianBlur(img_bgr, (3,3), 0)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img


# -------------------------------
# ✅ Load trained model
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PlantVillageCNN(num_classes=38).to(device)
model.load_state_dict(torch.load("/kaggle/working/best_plantvillage_cnn.pth", map_location=device))
model.eval()

# ✅ Class names
class_names = sorted(os.listdir("/kaggle/input/mission1111/Mission/test"))

# -------------------------------
# ✅ Preprocess single image
# -------------------------------
def preprocess_image(img_path):
    img_pil = Image.open(img_path).convert("RGB")
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    img_cv = custom_preprocess(img_cv)
    img_rgb = cv2.cvtColor((img_cv*255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    img_tensor = transform(image=img_rgb)["image"].unsqueeze(0)
    return img_tensor.to(device), img_rgb


# -------------------------------
# ✅ Predict function
# -------------------------------
def predict_image(img_path):
    img_tensor, img_rgb = preprocess_image(img_path)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        pred_class = class_names[predicted.item()]
    plt.imshow(img_rgb)
    plt.title(f"Predicted Class: {pred_class}")
    plt.axis("off")
    plt.show()
    return pred_class


# -------------------------------
# ✅ Test Example
# -------------------------------
test_image_path = "/kaggle/input/mission1111/Mission/test/Potato___Late_blight/01ad74ce-eb28-42c7-9204-778d17cfd45c___RS_LB_2669.jpg"  # change to your image path
predicted_class = predict_image(test_image_path)
print(f"✅ The model predicts: {predicted_class}")
