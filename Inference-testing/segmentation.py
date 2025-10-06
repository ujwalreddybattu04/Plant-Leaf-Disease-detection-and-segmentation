import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load trained model
model = load_model('Leaf_segmentation.h5')

# Load image
image_path = "/kaggle/input/plantdisease/PlantVillage/Potato___Late_blight/053c5330-129d-4515-84da-82a701710723___RS_LB 4576.JPG"   # change to your image path
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize to training size (128x128)
img_resized = cv2.resize(image, (128, 128))

# ⚠️ Do NOT divide by 255 — you trained on raw 0–255 pixels
img_input = np.expand_dims(img_resized, axis=0)

# Predict mask
pred_mask = model.predict(img_input)[0]

# Convert to binary mask
pred_mask = (pred_mask > 0.5).astype(np.uint8)

# Resize mask to original image size
pred_mask = cv2.resize(pred_mask, (image.shape[1], image.shape[0]))

# Show results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Predicted Segmentation Mask")
plt.imshow(pred_mask, cmap='gray')
plt.axis("off")

plt.show()
